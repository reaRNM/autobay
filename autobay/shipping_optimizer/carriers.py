"""
Carrier API integration for the Shipping Optimization Module.

This module provides client classes for integrating with major shipping carriers.
"""

import os
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from shipping_optimizer.models import Package, Address, ShippingRate


logger = logging.getLogger(__name__)


class BaseCarrierClient:
    """Base class for carrier API clients."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the carrier client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize session
        self.session = requests.Session()
        
        # Set up authentication headers
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            requests.RequestException: If the request fails after retries
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                return response.json()
            
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def get_rates(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> List[ShippingRate]:
        """
        Get shipping rates.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            List of shipping rates
        """
        raise NotImplementedError("Subclasses must implement get_rates")
    
    def track_shipment(self, tracking_number: str) -> Dict[str, Any]:
        """
        Track a shipment.
        
        Args:
            tracking_number: Tracking number
            
        Returns:
            Tracking information
        """
        raise NotImplementedError("Subclasses must implement track_shipment")


class USPSClient(BaseCarrierClient):
    """USPS API client."""
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the USPS client.
        
        Args:
            api_key: USPS API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="https://secure.shippingapis.com/ShippingAPI.dll",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Override headers for USPS
        self.headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    
    def get_rates(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> List[ShippingRate]:
        """
        Get USPS shipping rates.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            List of shipping rates
        """
        try:
            # Prepare request data
            request_data = self._prepare_rate_request(package, origin, destination, services)
            
            # Make API request
            response_data = self._make_usps_request("RateV4", request_data)
            
            # Parse response
            rates = self._parse_rate_response(response_data)
            
            return rates
        except Exception as e:
            logger.error(f"Error getting USPS rates: {e}")
            return []
    
    def _prepare_rate_request(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare USPS rate request data.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            Request data
        """
        # Map services to USPS service codes
        service_map = {
            "Priority": "PRIORITY",
            "First Class": "FIRST CLASS",
            "Ground": "RETAIL GROUND",
            "Express": "PRIORITY EXPRESS"
        }
        
        # Filter services if specified
        if services:
            service_codes = [service_map.get(s, s) for s in services if s in service_map]
        else:
            service_codes = list(service_map.values())
        
        # Prepare XML request
        xml_request = f"""
        <RateV4Request USERID="{self.api_key}">
            <Revision>2</Revision>
            <Package ID="1">
                <Service>{','.join(service_codes)}</Service>
                <ZipOrigination>{origin.postal_code}</ZipOrigination>
                <ZipDestination>{destination.postal_code}</ZipDestination>
                <Pounds>{int(package.weight_lb)}</Pounds>
                <Ounces>{(package.weight_lb % 1) * 16}</Ounces>
                <Container>VARIABLE</Container>
                <Size>REGULAR</Size>
                <Width>{package.width_in}</Width>
                <Length>{package.length_in}</Length>
                <Height>{package.height_in}</Height>
                <Girth>{2 * (package.width_in + package.height_in)}</Girth>
                <Value>{package.value}</Value>
                <Machinable>true</Machinable>
            </Package>
        </RateV4Request>
        """
        
        return {
            "API": "RateV4",
            "XML": xml_request
        }
    
    def _make_usps_request(self, api: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a USPS API request.
        
        Args:
            api: API name
            request_data: Request data
            
        Returns:
            Response data
        """
        # USPS uses a different request format
        params = {
            "API": request_data["API"],
            "XML": request_data["XML"]
        }
        
        # Make request
        response = self.session.get(
            self.base_url,
            params=params,
            timeout=self.timeout
        )
        
        # Parse XML response
        # Note: In a real implementation, use a proper XML parser
        # This is a simplified example
        response_text = response.text
        
        # Convert XML to dict (simplified)
        # In a real implementation, use xmltodict or similar
        response_data = {"RateV4Response": {"Package": {"Rate": []}}}
        
        # Extract rates from XML (simplified)
        import re
        rate_matches = re.findall(r'<Rate>(.*?)</Rate>', response_text)
        service_matches = re.findall(r'<MailService>(.*?)</MailService>', response_text)
        
        for i, (rate, service) in enumerate(zip(rate_matches, service_matches)):
            response_data["RateV4Response"]["Package"]["Rate"].append({
                "Rate": rate,
                "MailService": service
            })
        
        return response_data
    
    def _parse_rate_response(self, response_data: Dict[str, Any]) -> List[ShippingRate]:
        """
        Parse USPS rate response.
        
        Args:
            response_data: Response data
            
        Returns:
            List of shipping rates
        """
        rates = []
        
        # Extract rates from response
        rate_data = response_data.get("RateV4Response", {}).get("Package", {}).get("Rate", [])
        
        if not isinstance(rate_data, list):
            rate_data = [rate_data]
        
        for rate_info in rate_data:
            service = rate_info.get("MailService", "")
            rate = float(rate_info.get("Rate", 0))
            
            # Map delivery days based on service
            delivery_days = None
            if "Priority Express" in service:
                delivery_days = 1
            elif "Priority" in service:
                delivery_days = 2
            elif "First Class" in service:
                delivery_days = 3
            elif "Ground" in service:
                delivery_days = 5
            
            # Create shipping rate
            shipping_rate = ShippingRate(
                carrier="USPS",
                service=service,
                rate=rate,
                delivery_days=delivery_days,
                guaranteed="Express" in service,
                tracking_included=True,
                insurance_included=False,
                insurance_cost=0.0,
                signature_cost=0.0,
                fuel_surcharge=0.0
            )
            
            rates.append(shipping_rate)
        
        return rates
    
    def track_shipment(self, tracking_number: str) -> Dict[str, Any]:
        """
        Track a USPS shipment.
        
        Args:
            tracking_number: Tracking number
            
        Returns:
            Tracking information
        """
        try:
            # Prepare XML request
            xml_request = f"""
            <TrackFieldRequest USERID="{self.api_key}">
                <TrackID ID="{tracking_number}"></TrackID>
            </TrackFieldRequest>
            """
            
            # Make API request
            response = self.session.get(
                self.base_url,
                params={
                    "API": "TrackV2",
                    "XML": xml_request
                },
                timeout=self.timeout
            )
            
            # Parse XML response (simplified)
            # In a real implementation, use a proper XML parser
            response_text = response.text
            
            # Extract tracking info (simplified)
            import re
            status_match = re.search(r'<StatusDescription>(.*?)</StatusDescription>', response_text)
            status = status_match.group(1) if status_match else "Unknown"
            
            date_match = re.search(r'<EventDate>(.*?)</EventDate>', response_text)
            date_str = date_match.group(1) if date_match else ""
            
            time_match = re.search(r'<EventTime>(.*?)</EventTime>', response_text)
            time_str = time_match.group(1) if time_match else ""
            
            location_match = re.search(r'<EventCity>(.*?)</EventCity>', response_text)
            location = location_match.group(1) if location_match else "Unknown"
            
            # Create tracking info
            tracking_info = {
                "tracking_number": tracking_number,
                "status": status,
                "timestamp": f"{date_str} {time_str}",
                "location": location,
                "carrier": "USPS"
            }
            
            return tracking_info
        except Exception as e:
            logger.error(f"Error tracking USPS shipment: {e}")
            return {
                "tracking_number": tracking_number,
                "status": "Error",
                "error": str(e),
                "carrier": "USPS"
            }


class FedExClient(BaseCarrierClient):
    """FedEx API client."""
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the FedEx client.
        
        Args:
            api_key: FedEx API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="https://apis.fedex.com",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def get_rates(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> List[ShippingRate]:
        """
        Get FedEx shipping rates.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            List of shipping rates
        """
        try:
            # Prepare request data
            request_data = self._prepare_rate_request(package, origin, destination, services)
            
            # Make API request
            response_data = self._make_request("POST", "rate/v1/rates/quotes", data=request_data)
            
            # Parse response
            rates = self._parse_rate_response(response_data)
            
            return rates
        except Exception as e:
            logger.error(f"Error getting FedEx rates: {e}")
            return []
    
    def _prepare_rate_request(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare FedEx rate request data.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            Request data
        """
        # Map services to FedEx service codes
        service_map = {
            "Ground": "FEDEX_GROUND",
            "Express": "FEDEX_EXPRESS_SAVER",
            "2Day": "FEDEX_2_DAY",
            "Overnight": "STANDARD_OVERNIGHT"
        }
        
        # Filter services if specified
        if services:
            service_types = [service_map.get(s, s) for s in services if s in service_map]
        else:
            service_types = list(service_map.values())
        
        # Prepare request data
        request_data = {
            "accountNumber": {
                "value": "123456789"  # Replace with actual account number
            },
            "requestedShipment": {
                "shipper": {
                    "address": {
                        "streetLines": [origin.street1],
                        "city": origin.city,
                        "stateOrProvinceCode": origin.state,
                        "postalCode": origin.postal_code,
                        "countryCode": origin.country
                    }
                },
                "recipient": {
                    "address": {
                        "streetLines": [destination.street1],
                        "city": destination.city,
                        "stateOrProvinceCode": destination.state,
                        "postalCode": destination.postal_code,
                        "countryCode": destination.country,
                        "residential": destination.residential
                    }
                },
                "pickupType": "DROPOFF_AT_FEDEX_LOCATION",
                "rateRequestType": ["LIST", "ACCOUNT"],
                "requestedPackageLineItems": [{
                    "weight": {
                        "units": "LB",
                        "value": package.weight_lb
                    },
                    "dimensions": {
                        "length": package.length_in,
                        "width": package.width_in,
                        "height": package.height_in,
                        "units": "IN"
                    },
                    "packageSpecialServices": {
                        "specialServiceTypes": [
                            "SIGNATURE_OPTION" if package.requires_signature else None
                        ]
                    }
                }],
                "serviceType": service_types,
                "preferredCurrency": "USD"
            }
        }
        
        return request_data
    
    def _parse_rate_response(self, response_data: Dict[str, Any]) -> List[ShippingRate]:
        """
        Parse FedEx rate  response_data: Dict[str, Any]) -> List[ShippingRate]:
        """
        Parse FedEx rate response.
        
        Args:
            response_data: Response data
            
        Returns:
            List of shipping rates
        """
        rates = []
        
        # Extract rates from response
        rate_details = response_data.get("output", {}).get("rateReplyDetails", [])
        
        for rate_detail in rate_details:
            service_type = rate_detail.get("serviceType", "")
            
            # Get service name from service type
            service_name_map = {
                "FEDEX_GROUND": "Ground",
                "FEDEX_EXPRESS_SAVER": "Express Saver",
                "FEDEX_2_DAY": "2Day",
                "STANDARD_OVERNIGHT": "Standard Overnight",
                "PRIORITY_OVERNIGHT": "Priority Overnight"
            }
            
            service_name = service_name_map.get(service_type, service_type)
            
            # Get rate details
            rated_shipment_details = rate_detail.get("ratedShipmentDetails", [{}])[0]
            shipment_rate_detail = rated_shipment_details.get("shipmentRateDetail", {})
            
            # Get base rate
            total_base_charge = shipment_rate_detail.get("totalBaseCharge", {}).get("amount", 0)
            
            # Get surcharges
            surcharges = shipment_rate_detail.get("surcharges", [])
            fuel_surcharge = 0.0
            other_surcharges = {}
            
            for surcharge in surcharges:
                surcharge_type = surcharge.get("type", "")
                surcharge_amount = surcharge.get("amount", {}).get("amount", 0)
                
                if surcharge_type == "FUEL":
                    fuel_surcharge = surcharge_amount
                else:
                    other_surcharges[surcharge_type] = surcharge_amount
            
            # Get delivery details
            transit_time = rate_detail.get("commit", {}).get("transitTime", "")
            delivery_days = None
            
            if transit_time == "OVERNIGHT":
                delivery_days = 1
            elif transit_time == "2DAY":
                delivery_days = 2
            elif transit_time == "3DAY":
                delivery_days = 3
            elif transit_time == "4DAY":
                delivery_days = 4
            elif transit_time == "5DAY":
                delivery_days = 5
            
            # Create shipping rate
            shipping_rate = ShippingRate(
                carrier="FedEx",
                service=service_name,
                rate=float(total_base_charge),
                delivery_days=delivery_days,
                guaranteed=service_type != "FEDEX_GROUND",
                tracking_included=True,
                insurance_included=False,
                insurance_cost=0.0,
                signature_cost=0.0,
                fuel_surcharge=float(fuel_surcharge),
                other_surcharges=other_surcharges
            )
            
            rates.append(shipping_rate)
        
        return rates
    
    def track_shipment(self, tracking_number: str) -> Dict[str, Any]:
        """
        Track a FedEx shipment.
        
        Args:
            tracking_number: Tracking number
            
        Returns:
            Tracking information
        """
        try:
            # Prepare request data
            request_data = {
                "trackingInfo": [{
                    "trackingNumberInfo": {
                        "trackingNumber": tracking_number
                    }
                }],
                "includeDetailedScans": True
            }
            
            # Make API request
            response_data = self._make_request("POST", "track/v1/trackingnumbers", data=request_data)
            
            # Extract tracking info
            track_results = response_data.get("output", {}).get("completeTrackResults", [{}])[0]
            track_details = track_results.get("trackResults", [{}])[0]
            
            status = track_details.get("latestStatusDetail", {}).get("description", "Unknown")
            
            scan_events = track_details.get("scanEvents", [{}])[0]
            date = scan_events.get("date", "")
            time = scan_events.get("time", "")
            location = scan_events.get("scanLocation", {}).get("city", "Unknown")
            
            # Create tracking info
            tracking_info = {
                "tracking_number": tracking_number,
                "status": status,
                "timestamp": f"{date} {time}",
                "location": location,
                "carrier": "FedEx"
            }
            
            return tracking_info
        except Exception as e:
            logger.error(f"Error tracking FedEx shipment: {e}")
            return {
                "tracking_number": tracking_number,
                "status": "Error",
                "error": str(e),
                "carrier": "FedEx"
            }


class UPSClient(BaseCarrierClient):
    """UPS API client."""
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the UPS client.
        
        Args:
            api_key: UPS API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="https://onlinetools.ups.com/api",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def get_rates(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> List[ShippingRate]:
        """
        Get UPS shipping rates.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            List of shipping rates
        """
        try:
            # Prepare request data
            request_data = self._prepare_rate_request(package, origin, destination, services)
            
            # Make API request
            response_data = self._make_request("POST", "rating/v1/Rate", data=request_data)
            
            # Parse response
            rates = self._parse_rate_response(response_data)
            
            return rates
        except Exception as e:
            logger.error(f"Error getting UPS rates: {e}")
            return []
    
    def _prepare_rate_request(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare UPS rate request data.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            Request data
        """
        # Map services to UPS service codes
        service_map = {
            "Ground": "03",
            "3 Day Select": "12",
            "2nd Day Air": "02",
            "Next Day Air": "01"
        }
        
        # Prepare request data
        request_data = {
            "RateRequest": {
                "Request": {
                    "RequestOption": "Shop",
                    "TransactionReference": {
                        "CustomerContext": "Rate Request"
                    }
                },
                "Shipment": {
                    "Shipper": {
                        "Address": {
                            "AddressLine": [origin.street1],
                            "City": origin.city,
                            "StateProvinceCode": origin.state,
                            "PostalCode": origin.postal_code,
                            "CountryCode": origin.country
                        }
                    },
                    "ShipTo": {
                        "Address": {
                            "AddressLine": [destination.street1],
                            "City": destination.city,
                            "StateProvinceCode": destination.state,
                            "PostalCode": destination.postal_code,
                            "CountryCode": destination.country,
                            "ResidentialAddressIndicator": destination.residential
                        }
                    },
                    "Package": {
                        "PackagingType": {
                            "Code": "02"  # Customer packaging
                        },
                        "Dimensions": {
                            "UnitOfMeasurement": {
                                "Code": "IN"
                            },
                            "Length": str(package.length_in),
                            "Width": str(package.width_in),
                            "Height": str(package.height_in)
                        },
                        "PackageWeight": {
                            "UnitOfMeasurement": {
                                "Code": "LBS"
                            },
                            "Weight": str(package.weight_lb)
                        }
                    }
                }
            }
        }
        
        # Add service codes if specified
        if services:
            service_codes = [service_map.get(s, s) for s in services if s in service_map]
            for code in service_codes:
                request_data["RateRequest"]["Shipment"]["Service"] = {
                    "Code": code
                }
        
        return request_data
    
    def _parse_rate_response(self, response_data: Dict[str, Any]) -> List[ShippingRate]:
        """
        Parse UPS rate response.
        
        Args:
            response_data: Response data
            
        Returns:
            List of shipping rates
        """
        rates = []
        
        # Extract rates from response
        rate_response = response_data.get("RateResponse", {})
        rated_shipments = rate_response.get("RatedShipment", [])
        
        if not isinstance(rated_shipments, list):
            rated_shipments = [rated_shipments]
        
        for rated_shipment in rated_shipments:
            service_code = rated_shipment.get("Service", {}).get("Code", "")
            
            # Map service code to service name
            service_name_map = {
                "01": "Next Day Air",
                "02": "2nd Day Air",
                "03": "Ground",
                "12": "3 Day Select",
                "13": "Next Day Air Saver",
                "14": "UPS Next Day Air Early"
            }
            
            service_name = service_name_map.get(service_code, f"Service {service_code}")
            
            # Get rate details
            total_charges = rated_shipment.get("TotalCharges", {}).get("MonetaryValue", "0")
            
            # Get guaranteed delivery
            guaranteed_indicator = rated_shipment.get("GuaranteedDelivery", {}).get("Indicator", "")
            guaranteed = guaranteed_indicator == "Y"
            
            # Get delivery days
            delivery_days = rated_shipment.get("GuaranteedDelivery", {}).get("BusinessDaysInTransit", "")
            if delivery_days:
                delivery_days = int(delivery_days)
            else:
                # Estimate based on service
                if service_code == "01":
                    delivery_days = 1
                elif service_code == "02":
                    delivery_days = 2
                elif service_code == "12":
                    delivery_days = 3
                elif service_code == "03":
                    delivery_days = 5
                else:
                    delivery_days = None
            
            # Create shipping rate
            shipping_rate = ShippingRate(
                carrier="UPS",
                service=service_name,
                rate=float(total_charges),
                delivery_days=delivery_days,
                guaranteed=guaranteed,
                tracking_included=True,
                insurance_included=False,
                insurance_cost=0.0,
                signature_cost=0.0,
                fuel_surcharge=0.0  # UPS includes fuel surcharge in total
            )
            
            rates.append(shipping_rate)
        
        return rates
    
    def track_shipment(self, tracking_number: str) -> Dict[str, Any]:
        """
        Track a UPS shipment.
        
        Args:
            tracking_number: Tracking number
            
        Returns:
            Tracking information
        """
        try:
            # Prepare request data
            request_data = {
                "TrackRequest": {
                    "Request": {
                        "RequestOption": "1",
                        "TransactionReference": {
                            "CustomerContext": "Track Request"
                        }
                    },
                    "InquiryNumber": tracking_number
                }
            }
            
            # Make API request
            response_data = self._make_request("POST", "track/v1/details", data=request_data)
            
            # Extract tracking info
            track_response = response_data.get("TrackResponse", {})
            shipment = track_response.get("Shipment", [{}])[0]
            
            status = shipment.get("CurrentStatus", {}).get("Description", "Unknown")
            
            activity = shipment.get("Activity", [{}])[0]
            date = activity.get("Date", "")
            time = activity.get("Time", "")
            location = activity.get("ActivityLocation", {}).get("City", "Unknown")
            
            # Create tracking info
            tracking_info = {
                "tracking_number": tracking_number,
                "status": status,
                "timestamp": f"{date} {time}",
                "location": location,
                "carrier": "UPS"
            }
            
            return tracking_info
        except Exception as e:
            logger.error(f"Error tracking UPS shipment: {e}")
            return {
                "tracking_number": tracking_number,
                "status": "Error",
                "error": str(e),
                "carrier": "UPS"
            }


class DHLClient(BaseCarrierClient):
    """DHL API client."""
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the DHL client.
        
        Args:
            api_key: DHL API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api-mock.dhl.com/mydhlapi",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def get_rates(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> List[ShippingRate]:
        """
        Get DHL shipping rates.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            List of shipping rates
        """
        try:
            # Prepare request data
            request_data = self._prepare_rate_request(package, origin, destination, services)
            
            # Make API request
            response_data = self._make_request("POST", "rates", data=request_data)
            
            # Parse response
            rates = self._parse_rate_response(response_data)
            
            return rates
        except Exception as e:
            logger.error(f"Error getting DHL rates: {e}")
            return []
    
    def _prepare_rate_request(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare DHL rate request data.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            Request data
        """
        # Map services to DHL service codes
        service_map = {
            "Express": "P",
            "Express 9:00": "K",
            "Express 10:30": "T",
            "Express 12:00": "Y"
        }
        
        # Prepare request data
        request_data = {
            "customerDetails": {
                "shipperDetails": {
                    "postalCode": origin.postal_code,
                    "cityName": origin.city,
                    "countryCode": origin.country
                },
                "receiverDetails": {
                    "postalCode": destination.postal_code,
                    "cityName": destination.city,
                    "countryCode": destination.country
                }
            },
            "plannedShippingDateAndTime": datetime.now().isoformat(),
            "unitOfMeasurement": "imperial",
            "packages": [{
                "weight": package.weight_lb,
                "dimensions": {
                    "length": package.length_in,
                    "width": package.width_in,
                    "height": package.height_in
                }
            }],
            "productCode": None
        }
        
        # Add service codes if specified
        if services:
            service_codes = [service_map.get(s, s) for s in services if s in service_map]
            if service_codes:
                request_data["productCode"] = service_codes[0]
        
        return request_data
    
    def _parse_rate_response(self, response_data: Dict[str, Any]) -> List[ShippingRate]:
        """
        Parse DHL rate response.
        
        Args:
            response_data: Response data
            
        Returns:
            List of shipping rates
        """
        rates = []
        
        # Extract rates from response
        products = response_data.get("products", [])
        
        for product in products:
            product_name = product.get("productName", "")
            product_code = product.get("productCode", "")
            
            # Get rate details
            total_price = product.get("totalPrice", [{}])[0]
            price = total_price.get("price", 0)
            currency = total_price.get("currencyType", "USD")
            
            # Get delivery details
            delivery_time = product.get("deliveryTime", {})
            delivery_days = delivery_time.get("days", None)
            
            # Get additional services
            additional_services = product.get("serviceBreakdown", [])
            
            # Create shipping rate
            shipping_rate = ShippingRate(
                carrier="DHL",
                service=product_name,
                rate=float(price),
                delivery_days=delivery_days,
                guaranteed=True,  # DHL services are typically guaranteed
                tracking_included=True,
                insurance_included=False,
                insurance_cost=0.0,
                signature_cost=0.0,
                fuel_surcharge=0.0  # DHL includes fuel surcharge in total
            )
            
            rates.append(shipping_rate)
        
        return rates
    
    def track_shipment(self, tracking_number: str) -> Dict[str, Any]:
        """
        Track a DHL shipment.
        
        Args:
            tracking_number: Tracking number
            
        Returns:
            Tracking information
        """
        try:
            # Make API request
            response_data = self._make_request(
                "GET",
                f"shipments/{tracking_number}/tracking",
                params={"trackingView": "all-checkpoints"}
            )
            
            # Extract tracking info
            shipments = response_data.get("shipments", [{}])[0]
            
            status = shipments.get("status", {}).get("description", "Unknown")
            
            events = shipments.get("events", [{}])[0]
            timestamp = events.get("timestamp", "")
            location = events.get("location", {}).get("address", {}).get("city", "Unknown")
            
            # Create tracking info
            tracking_info = {
                "tracking_number": tracking_number,
                "status": status,
                "timestamp": timestamp,
                "location": location,
                "carrier": "DHL"
            }
            
            return tracking_info
        except Exception as e:
            logger.error(f"Error tracking DHL shipment: {e}")
            return {
                "tracking_number": tracking_number,
                "status": "Error",
                "error": str(e),
                "carrier": "DHL"
            }