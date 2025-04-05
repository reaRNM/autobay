"""
Image processing module for the Listing Generator.

This module provides functionality for processing product images,
generating alt text, captions, and enhancement suggestions.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
import random
from openai import AsyncOpenAI

from listing_generator.models import Product, ImageMetadata


logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Processor for product images.
    
    This class provides methods for processing product images,
    generating alt text, captions, and enhancement suggestions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ImageProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.generate_alt_text = config.get('generate_alt_text', True)
        self.generate_captions = config.get('generate_captions', True)
        self.suggest_enhancements = config.get('suggest_enhancements', True)
        self.max_tags = config.get('max_tags', 10)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        logger.info("ImageProcessor initialized")
    
    async def process_image(
        self,
        product: Product,
        image_url: str,
        is_primary: bool = False
    ) -> ImageMetadata:
        """
        Process a product image.
        
        Args:
            product: Product the image belongs to
            image_url: URL of the image
            is_primary: Whether this is the primary product image
            
        Returns:
            Image metadata
        """
        logger.info(f"Processing image for product {product.id}: {image_url}")
        
        try:
            # Generate alt text
            alt_text = await self._generate_alt_text(product, image_url, is_primary) if self.generate_alt_text else ""
            
            # Generate caption
            caption = await self._generate_caption(product, image_url, is_primary) if self.generate_captions else None
            
            # Generate tags
            tags = await self._generate_tags(product, image_url, is_primary)
            
            # Generate enhancement suggestions
            enhancement_suggestions = await self._suggest_enhancements(image_url) if self.suggest_enhancements else {}
            
            # Create image metadata
            metadata = ImageMetadata(
                id=str(random.randint(1000000, 9999999)),
                product_id=product.id,
                image_url=image_url,
                alt_text=alt_text,
                caption=caption,
                tags=tags,
                enhancement_suggestions=enhancement_suggestions,
                is_primary=is_primary
            )
            
            logger.info(f"Generated metadata for image: {image_url}")
            return metadata
        
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            # Fallback to basic metadata
            return self._generate_fallback_metadata(product, image_url, is_primary)
    
    async def _generate_alt_text(
        self,
        product: Product,
        image_url: str,
        is_primary: bool
    ) -> str:
        """
        Generate alt text for a product image.
        
        Args:
            product: Product the image belongs to
            image_url: URL of the image
            is_primary: Whether this is the primary product image
            
        Returns:
            Alt text
        """
        try:
            # In a real implementation, this would use computer vision or OpenAI's vision API
            # For this example, we'll use product information to generate alt text
            
            # Prepare prompt for OpenAI
            prompt = f"""
            Generate a concise, descriptive alt text for a product image.
            
            Product details:
            - Name: {product.name}
            - Brand: {product.brand or 'N/A'}
            - Category: {product.category or 'N/A'}
            - Primary image: {"Yes" if is_primary else "No"}
            
            The alt text should:
            - Be concise (10-15 words maximum)
            - Be descriptive and specific
            - Include key product details
            - Not include phrases like "image of" or "picture of"
            - Focus on the product's visual appearance
            
            Return ONLY the alt text with no additional commentary or explanation.
            """
            
            # Generate alt text using OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller model for alt text
                messages=[
                    {"role": "system", "content": "You are an expert in writing accessible and SEO-friendly alt text for e-commerce product images."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50,
                n=1
            )
            
            # Extract alt text from response
            alt_text = response.choices[0].message.content.strip()
            
            # Clean up alt text
            alt_text = alt_text.strip('"\'')
            
            # Ensure it's not too long
            if len(alt_text) > 125:
                alt_text = alt_text[:125].rsplit(' ', 1)[0] + '...'
            
            return alt_text
        
        except Exception as e:
            logger.error(f"Error generating alt text: {e}")
            # Fallback to basic alt text
            return self._generate_fallback_alt_text(product, is_primary)
    
    async def _generate_caption(
        self,
        product: Product,
        image_url: str,
        is_primary: bool
    ) -> Optional[str]:
        """
        Generate a caption for a product image.
        
        Args:
            product: Product the image belongs to
            image_url: URL of the image
            is_primary: Whether this is the primary product image
            
        Returns:
            Caption or None
        """
        try:
            # In a real implementation, this would use computer vision or OpenAI's vision API
            # For this example, we'll use product information to generate a caption
            
            # Prepare prompt for OpenAI
            prompt = f"""
            Generate a compelling, SEO-friendly caption for a product image.
            
            Product details:
            - Name: {product.name}
            - Brand: {product.brand or 'N/A'}
            - Category: {product.category or 'N/A'}
            - Primary image: {"Yes" if is_primary else "No"}
            
            The caption should:
            - Be 1-2 sentences (20-30 words)
            - Highlight key selling points
            - Include relevant keywords
            - Be engaging and persuasive
            - Focus on benefits to the customer
            
            Return ONLY the caption with no additional commentary or explanation.
            """
            
            # Generate caption using OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller model for captions
                messages=[
                    {"role": "system", "content": "You are an expert in writing compelling, SEO-friendly captions for e-commerce product images."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=75,
                n=1
            )
            
            # Extract caption from response
            caption = response.choices[0].message.content.strip()
            
            # Clean up caption
            caption = caption.strip('"\'')
            
            # Ensure it's not too long
            if len(caption) > 200:
                caption = caption[:200].rsplit('.', 1)[0] + '.'
            
            return caption
        
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            # Fallback to no caption
            return None
    
    async def _generate_tags(
        self,
        product: Product,
        image_url: str,
        is_primary: bool
    ) -> List[str]:
        """
        Generate tags for a product image.
        
        Args:
            product: Product the image belongs to
            image_url: URL of the image
            is_primary: Whether this is the primary product image
            
        Returns:
            List of tags
        """
        try:
            # In a real implementation, this would use computer vision or OpenAI's vision API
            # For this example, we'll use product information to generate tags
            
            # Start with basic tags from product info
            tags = []
            
            if product.brand:
                tags.append(product.brand.lower())
            
            if product.category:
                tags.append(product.category.lower())
            
            if product.condition:
                tags.append(product.condition.lower())
            
            # Add features as tags
            for feature in product.features[:3]:
                # Extract key terms from feature
                terms = feature.lower().split()
                for term in terms:
                    if len(term) >= 4 and term not in tags:
                        tags.append(term)
            
            # If we have enough tags, return them
            if len(tags) >= self.max_tags:
                return tags[:self.max_tags]
            
            # Otherwise, generate more tags using OpenAI
            prompt = f"""
            Generate relevant tags for a product image.
            
            Product details:
            - Name: {product.name}
            - Brand: {product.brand or 'N/A'}
            - Category: {product.category or 'N/A'}
            
            The tags should:
            - Be single words or short phrases
            - Be relevant to the product
            - Include visual attributes
            - Include use cases
            - Include target audience
            
            Return ONLY a comma-separated list of tags with no additional commentary or explanation.
            """
            
            # Generate tags using OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller model for tags
                messages=[
                    {"role": "system", "content": "You are an expert in generating relevant tags for e-commerce product images."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=100,
                n=1
            )
            
            # Extract tags from response
            content = response.choices[0].message.content.strip()
            
            # Split by commas and clean
            ai_tags = [tag.strip().lower() for tag in content.split(',')]
            
            # Add AI-generated tags
            for tag in ai_tags:
                if tag and tag not in tags:
                    tags.append(tag)
            
            # Limit to max tags
            return tags[:self.max_tags]
        
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            # Fallback to basic tags
            return self._generate_fallback_tags(product)
    
    async def _suggest_enhancements(self, image_url: str) -> Dict[str, Any]:
        """
        Suggest enhancements for a product image.
        
        Args:
            image_url: URL of the image
            
        Returns:
            Dictionary of enhancement suggestions
        """
        # In a real implementation, this would analyze the image
        # For this example, we'll return generic suggestions
        
        # Randomly select a few suggestions
        suggestions = {}
        
        all_suggestions = {
            "lighting": {
                "issue": "Image appears too dark",
                "suggestion": "Increase brightness and contrast for better visibility",
                "severity": "medium"
            },
            "background": {
                "issue": "Background is distracting",
                "suggestion": "Use a plain white or contextually appropriate background",
                "severity": "low"
            },
            "angle": {
                "issue": "Product features not clearly visible",
                "suggestion": "Capture from multiple angles to showcase key features",
                "severity": "medium"
            },
            "resolution": {
                "issue": "Image resolution could be improved",
                "suggestion": "Use higher resolution images (at least 1500x1500 pixels)",
                "severity": "high"
            },
            "focus": {
                "issue": "Product appears slightly out of focus",
                "suggestion": "Ensure product is in sharp focus",
                "severity": "medium"
            },
            "cropping": {
                "issue": "Product not centered in frame",
                "suggestion": "Crop image to center the product with appropriate margins",
                "severity": "low"
            }
        }
        
        # Randomly select 2-3 suggestions
        num_suggestions = random.randint(2, 3)
        suggestion_keys = random.sample(list(all_suggestions.keys()), num_suggestions)
        
        for key in suggestion_keys:
            suggestions[key] = all_suggestions[key]
        
        return suggestions
    
    def _generate_fallback_alt_text(self, product: Product, is_primary: bool) -> str:
        """
        Generate fallback alt text if the API fails.
        
        Args:
            product: Product to generate alt text for
            is_primary: Whether this is the primary product image
            
        Returns:
            Fallback alt text
        """
        if is_primary:
            return f"{product.brand} {product.name} {product.condition} {product.category}".strip()
        else:
            return f"{product.brand} {product.name} - additional view".strip()
    
    def _generate_fallback_tags(self, product: Product) -> List[str]:
        """
        Generate fallback tags if the API fails.
        
        Args:
            product: Product to generate tags for
            
        Returns:
            Fallback tags
        """
        tags = []
        
        if product.brand:
            tags.append(product.brand.lower())
        
        if product.category:
            tags.append(product.category.lower())
        
        if product.condition:
            tags.append(product.condition.lower())
        
        # Add some generic tags
        generic_tags = ["quality", "product", "online", "shopping", "deal", "value"]
        tags.extend(generic_tags)
        
        return tags[:self.max_tags]
    
    def _generate_fallback_metadata(
        self,
        product: Product,
        image_url: str,
        is_primary: bool
    ) -> ImageMetadata:
        """
        Generate fallback metadata if processing fails.
        
        Args:
            product: Product the image belongs to
            image_url: URL of the image
            is_primary: Whether this is the primary product image
            
        Returns:
            Fallback metadata
        """
        alt_text = self._generate_fallback_alt_text(product, is_primary)
        tags = self._generate_fallback_tags(product)
        
        return ImageMetadata(
            id=str(random.randint(1000000, 9999999)),
            product_id=product.id,
            image_url=image_url,
            alt_text=alt_text,
            caption=None,
            tags=tags,
            enhancement_suggestions={},
            is_primary=is_primary
        )