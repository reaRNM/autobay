"""
Mobile notification service for the Dashboard & Mobile Alerts module.

This module provides functionality for sending mobile notifications
via Telegram, email, or other channels.
"""

import logging
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .models import AlertConfig, AlertHistory, db

logger = logging.getLogger(__name__)


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the notification channel."""
        self.config = config
    
    def send(self, user_id: int, message: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a notification.
        
        Args:
            user_id: ID of the user to notify
            message: Notification message
            data: Additional data for the notification
            
        Returns:
            Dict[str, Any]: Notification result
        """
        raise NotImplementedError("Subclasses must implement send()")


class TelegramNotifier(NotificationChannel):
    """Telegram notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Telegram notifier."""
        super().__init__(config)
        self.bot_token = config.get('bot_token')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if not self.bot_token:
            raise ValueError("Telegram bot token is required")
    
    def send(self, user_id: int, message: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a Telegram notification.
        
        Args:
            user_id: ID of the user to notify
            message: Notification message
            data: Additional data for the notification
            
        Returns:
            Dict[str, Any]: Notification result
        """
        # Get the Telegram chat ID for the user
        chat_id = self._get_chat_id(user_id)
        
        if not chat_id:
            logger.warning(f"No Telegram chat ID found for user {user_id}")
            return {
                'success': False,
                'error': 'No Telegram chat ID found for user'
            }
        
        # Format the message
        formatted_message = self._format_message(message, data)
        
        # Send the message
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    'chat_id': chat_id,
                    'text': formatted_message,
                    'parse_mode': 'HTML'
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('ok'):
                logger.info(f"Telegram notification sent to user {user_id}")
                return {
                    'success': True,
                    'message_id': result.get('result', {}).get('message_id')
                }
            else:
                logger.error(f"Failed to send Telegram notification: {result}")
                return {
                    'success': False,
                    'error': result.get('description', 'Unknown error')
                }
        
        except Exception as e:
            logger.exception(f"Error sending Telegram notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_chat_id(self, user_id: int) -> Optional[str]:
        """
        Get the Telegram chat ID for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Optional[str]: Telegram chat ID, or None if not found
        """
        # In a real implementation, this would query a database
        # to get the Telegram chat ID for the user
        # For this example, we'll use a simple mapping
        chat_id_mapping = self.config.get('chat_id_mapping', {})
        return chat_id_mapping.get(str(user_id))
    
    def _format_message(self, message: str, data: Dict[str, Any]) -> str:
        """
        Format a notification message for Telegram.
        
        Args:
            message: Base message
            data: Additional data for the notification
            
        Returns:
            str: Formatted message
        """
        # Basic HTML formatting for Telegram
        formatted_message = f"<b>🔔 Auction Alert</b>\n\n{message}\n"
        
        # Add item details if available
        if 'item' in data:
            item = data['item']
            formatted_message += f"\n<b>Item:</b> {item.get('title', 'N/A')}"
            
            if 'current_bid' in item:
                formatted_message += f"\n<b>Current Bid:</b> ${item['current_bid']:.2f}"
            
            if 'estimated_profit' in item:
                formatted_message += f"\n<b>Est. Profit:</b> ${item['estimated_profit']:.2f}"
            
            if 'risk_score' in item:
                formatted_message += f"\n<b>Risk Score:</b> {item['risk_score']:.2f}"
            
            if 'auction_end_time' in item and item['auction_end_time']:
                # Format the end time
                try:
                    end_time = datetime.fromisoformat(item['auction_end_time'])
                    formatted_message += f"\n<b>Ends:</b> {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                except (ValueError, TypeError):
                    pass
        
        # Add action buttons if available
        if 'actions' in data:
            formatted_message += "\n\n<b>Actions:</b>"
            for action in data['actions']:
                formatted_message += f"\n• {action}"
        
        return formatted_message


class EmailNotifier(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the email notifier."""
        super().__init__(config)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_username = config.get('smtp_username')
        self.smtp_password = config.get('smtp_password')
        self.from_email = config.get('from_email')
        
        if not all([self.smtp_server, self.smtp_username, self.smtp_password, self.from_email]):
            raise ValueError("SMTP configuration is incomplete")
    
    def send(self, user_id: int, message: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an email notification.
        
        Args:
            user_id: ID of the user to notify
            message: Notification message
            data: Additional data for the notification
            
        Returns:
            Dict[str, Any]: Notification result
        """
        # Get the email address for the user
        to_email = self._get_email(user_id)
        
        if not to_email:
            logger.warning(f"No email address found for user {user_id}")
            return {
                'success': False,
                'error': 'No email address found for user'
            }
        
        # Create the email
        email = MIMEMultipart('alternative')
        email['Subject'] = f"Auction Alert: {data.get('alert_type', 'Notification')}"
        email['From'] = self.from_email
        email['To'] = to_email
        
        # Format the message
        text_content = self._format_text_message(message, data)
        html_content = self._format_html_message(message, data)
        
        # Attach the content
        email.attach(MIMEText(text_content, 'plain'))
        email.attach(MIMEText(html_content, 'html'))
        
        # Send the email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(email)
            
            logger.info(f"Email notification sent to user {user_id}")
            return {
                'success': True
            }
        
        except Exception as e:
            logger.exception(f"Error sending email notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_email(self, user_id: int) -> Optional[str]:
        """
        Get the email address for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Optional[str]: Email address, or None if not found
        """
        # In a real implementation, this would query a database
        # to get the email address for the user
        # For this example, we'll use a simple mapping
        email_mapping = self.config.get('email_mapping', {})
        return email_mapping.get(str(user_id))
    
    def _format_text_message(self, message: str, data: Dict[str, Any]) -> str:
        """
        Format a notification message as plain text.
        
        Args:
            message: Base message
            data: Additional data for the notification
            
        Returns:
            str: Formatted message
        """
        formatted_message = f"Auction Alert\n\n{message}\n"
        
        # Add item details if available
        if 'item' in data:
            item = data['item']
            formatted_message += f"\nItem: {item.get('title', 'N/A')}"
            
            if 'current_bid' in item:
                formatted_message += f"\nCurrent Bid: ${item['current_bid']:.2f}"
            
            if 'estimated_profit' in item:
                formatted_message += f"\nEst. Profit: ${item['estimated_profit']:.2f}"
            
            if 'risk_score' in item:
                formatted_message += f"\nRisk Score: {item['risk_score']:.2f}"
            
            if 'auction_end_time' in item and item['auction_end_time']:
                # Format the end time
                try:
                    end_time = datetime.fromisoformat(item['auction_end_time'])
                    formatted_message += f"\nEnds: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                except (ValueError, TypeError):
                    pass
        
        return formatted_message
    
    def _format_html_message(self, message: str, data: Dict[str, Any]) -> str:
        """
        Format a notification message as HTML.
        
        Args:
            message: Base message
            data: Additional data for the notification
            
        Returns:
            str: Formatted message
        """
        formatted_message = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
                .content {{ padding: 20px; }}
                .item-details {{ background-color: #f9f9f9; padding: 15px; margin-top: 20px; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #777; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Auction Alert</h2>
                </div>
                <div class="content">
                    <p>{message}</p>
        """
        
        # Add item details if available
        if 'item' in data:
            item = data['item']
            formatted_message += f"""
                    <div class="item-details">
                        <h3>Item Details</h3>
                        <p><strong>Title:</strong> {item.get('title', 'N/A')}</p>
            """
            
            if 'current_bid' in item:
                formatted_message += f'<p><strong>Current Bid:</strong> ${item["current_bid"]:.2f}</p>'
            
            if 'estimated_profit' in item:
                formatted_message += f'<p><strong>Est. Profit:</strong> ${item["estimated_profit"]:.2f}</p>'
            
            if 'risk_score' in item:
                formatted_message += f'<p><strong>Risk Score:</strong> {item["risk_score"]:.2f}</p>'
            
            if 'auction_end_time' in item and item['auction_end_time']:
                # Format the end time
                try:
                    end_time = datetime.fromisoformat(item['auction_end_time'])
                    formatted_message += f'<p><strong>Ends:</strong> {end_time.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>'
                except (ValueError, TypeError):
                    pass
            
            formatted_message += """
                    </div>
            """
        
        # Add action buttons if available
        if 'actions' in data:
            formatted_message += """
                    <div style="margin-top: 20px;">
                        <h3>Actions</h3>
                        <ul>
            """
            
            for action in data['actions']:
                formatted_message += f'<li>{action}</li>'
            
            formatted_message += """
                        </ul>
                    </div>
            """
        
        # Close the HTML
        formatted_message += """
                </div>
                <div class="footer">
                    <p>This is an automated notification from your Auction Dashboard.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return formatted_message


class NotificationService:
    """Service for sending notifications via various channels."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the notification service."""
        self.config = config or {}
        self.channels = {}
        
        # Initialize notification channels
        self._init_channels()
    
    def _init_channels(self):
        """Initialize notification channels from config."""
        # Telegram
        telegram_config = self.config.get('telegram', {})
        if telegram_config.get('enabled', False):
            try:
                self.channels['telegram'] = TelegramNotifier(telegram_config)
                logger.info("Telegram notification channel initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram notification channel: {e}")
        
        # Email
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False):
            try:
                self.channels['email'] = EmailNotifier(email_config)
                logger.info("Email notification channel initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Email notification channel: {e}")
    
    def send_notification(self, user_id: int, alert_type: str, message: str, 
                         data: Dict[str, Any] = None, 
                         channels: List[str] = None) -> Dict[str, Any]:
        """
        Send a notification to a user.
        
        Args:
            user_id: ID of the user to notify
            alert_type: Type of alert
            message: Notification message
            data: Additional data for the notification
            channels: List of channels to use (if None, use all available channels)
            
        Returns:
            Dict[str, Any]: Notification results by channel
        """
        data = data or {}
        data['alert_type'] = alert_type
        
        # Determine which channels to use
        if channels is None:
            channels = list(self.channels.keys())
        
        # Send notifications
        results = {}
        for channel_name in channels:
            if channel_name in self.channels:
                try:
                    channel = self.channels[channel_name]
                    results[channel_name] = channel.send(user_id, message, data)
                except Exception as e:
                    logger.exception(f"Error sending notification via {channel_name}: {e}")
                    results[channel_name] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                logger.warning(f"Notification channel {channel_name} not available")
                results[channel_name] = {
                    'success': False,
                    'error': 'Channel not available'
                }
        
        # Record the notification in the database
        try:
            alert_history = AlertHistory(
                user_id=user_id,
                alert_type=alert_type,
                message=message,
                data=data,
                notification_channels=channels,
                delivery_status=results
            )
            db.session.add(alert_history)
            db.session.commit()
            logger.info(f"Alert history recorded for user {user_id}")
        except Exception as e:
            logger.exception(f"Error recording alert history: {e}")
        
        return results
    
    def send_alert_by_config(self, alert_config_id: int, message: str, 
                           data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a notification based on an alert configuration.
        
        Args:
            alert_config_id: ID of the alert configuration
            message: Notification message
            data: Additional data for the notification
            
        Returns:
            Dict[str, Any]: Notification results by channel
        """
        # Get the alert configuration
        alert_config = AlertConfig.query.get(alert_config_id)
        
        if not alert_config:
            logger.error(f"Alert configuration {alert_config_id} not found")
            return {
                'success': False,
                'error': 'Alert configuration not found'
            }
        
        if not alert_config.is_active:
            logger.info(f"Alert configuration {alert_config_id} is inactive")
            return {
                'success': False,
                'error': 'Alert configuration is inactive'
            }
        
        # Send the notification
        return self.send_notification(
            user_id=alert_config.user_id,
            alert_type=alert_config.alert_type,
            message=message,
            data=data,
            channels=alert_config.notification_channels
        )
    
    def process_alert_trigger(self, alert_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an alert trigger and send notifications to matching configurations.
        
        Args:
            alert_type: Type of alert
            data: Alert data
            
        Returns:
            List[Dict[str, Any]]: Notification results
        """
        # Find matching alert configurations
        alert_configs = AlertConfig.query.filter_by(
            alert_type=alert_type,
            is_active=True
        ).all()
        
        results = []
        for config in alert_configs:
            # Check if the alert conditions match
            if self._check_alert_conditions(config.conditions, data):
                # Generate message
                message = self._generate_alert_message(config, data)
                
                # Send notification
                result = self.send_alert_by_config(
                    alert_config_id=config.id,
                    message=message,
                    data=data
                )
                
                results.append({
                    'alert_config_id': config.id,
                    'user_id': config.user_id,
                    'result': result
                })
        
        return results
    
    def _check_alert_conditions(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Check if alert conditions match the data.
        
        Args:
            conditions: Alert conditions
            data: Alert data
            
        Returns:
            bool: True if conditions match, False otherwise
        """
        # Simple condition matching
        for key, condition in conditions.items():
            if key not in data:
                return False
            
            value = data[key]
            
            # Handle different condition types
            if isinstance(condition, dict):
                operator = condition.get('operator', '==')
                threshold = condition.get('value')
                
                if operator == '==' and value != threshold:
                    return False
                elif operator == '!=' and value == threshold:
                    return False
                elif operator == '>' and value <= threshold:
                    return False
                elif operator == '>=' and value < threshold:
                    return False
                elif operator == '<' and value >= threshold:
                    return False
                elif operator == '<=' and value > threshold:
                    return False
            else:
                # Direct comparison
                if value != condition:
                    return False
        
        return True
    
    def _generate_alert_message(self, config: AlertConfig, data: Dict[str, Any]) -> str:
        """
        Generate an alert message based on the configuration and data.
        
        Args:
            config: Alert configuration
            data: Alert data
            
        Returns:
            str: Alert message
        """
        # Use a template if available
        template = config.conditions.get('message_template')
        
        if template:
            # Simple template substitution
            message = template
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                if placeholder in message:
                    message = message.replace(placeholder, str(value))
            
            return message
        
        # Default messages by alert type
        if config.alert_type == 'auction_opportunity':
            return "New auction opportunity detected!"
        elif config.alert_type == 'bid_spike':
            return "Significant bid increase detected!"
        elif config.alert_type == 'auction_ending':
            return "Auction ending soon!"
        elif config.alert_type == 'price_drop':
            return "Price drop detected!"
        else:
            return f"Alert: {config.alert_type}"