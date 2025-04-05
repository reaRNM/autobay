"""
NLP module for the Listing Generator.

This module provides functionality for generating optimized titles,
descriptions, and keywords using NLP techniques.
"""

import os
import logging
import re
import asyncio
from typing import Dict, List, Optional, Any, Set
import spacy
from openai import AsyncOpenAI
import numpy as np

from listing_generator.models import Product, Marketplace, ListingPerformance


logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    Generator for optimized listing titles.
    
    This class provides methods for generating optimized titles
    for product listings on various marketplaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TitleGenerator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_length = config.get('max_length', 80)
        self.num_variations = config.get('num_variations', 3)
        self.model = config.get('model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.7)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Load marketplace-specific title templates
        self.title_templates = {
            Marketplace.AMAZON: "{brand} {model} {key_feature} {category} {condition}",
            Marketplace.EBAY: "{brand} {model} {key_feature} {category} {condition} {color} {size}",
            Marketplace.ETSY: "{handmade} {vintage} {custom} {material} {category} {style}",
            Marketplace.WALMART: "{brand} {model} {category} {key_feature} {size}"
        }
        
        # Load marketplace-specific title length limits
        self.title_length_limits = {
            Marketplace.AMAZON: 200,
            Marketplace.EBAY: 80,
            Marketplace.ETSY: 140,
            Marketplace.WALMART: 200
        }
        
        logger.info("TitleGenerator initialized")
    
    async def generate_title(
        self,
        product: Product,
        marketplace: Marketplace,
        use_performance_data: bool = False,
        performance_data: Optional[ListingPerformance] = None
    ) -> str:
        """
        Generate an optimized title for a product.
        
        Args:
            product: Product to generate title for
            marketplace: Target marketplace
            use_performance_data: Whether to use performance data
            performance_data: Performance data to use
            
        Returns:
            Optimized title
        """
        logger.info(f"Generating title for product {product.id} on {marketplace}")
        
        try:
            # Get marketplace-specific length limit
            length_limit = self.title_length_limits.get(marketplace, self.max_length)
            
            # Prepare prompt for OpenAI
            prompt = self._create_title_prompt(product, marketplace, use_performance_data, performance_data)
            
            # Generate title using OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert e-commerce listing optimizer specializing in creating high-converting, SEO-friendly product titles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=100,
                n=1
            )
            
            # Extract title from response
            title = response.choices[0].message.content.strip()
            
            # Clean up title
            title = self._clean_title(title, length_limit)
            
            logger.info(f"Generated title: {title}")
            return title
        
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            # Fallback to basic title
            return self._generate_fallback_title(product, marketplace)
    
    async def generate_variations(
        self,
        product: Product,
        marketplace: Marketplace,
        num_variations: int = 3,
        existing_title: Optional[str] = None
    ) -> List[str]:
        """
        Generate variations of a title for A/B testing.
        
        Args:
            product: Product to generate title for
            marketplace: Target marketplace
            num_variations: Number of variations to generate
            existing_title: Existing title to use as a base
            
        Returns:
            List of title variations
        """
        logger.info(f"Generating {num_variations} title variations for product {product.id}")
        
        try:
            # Get marketplace-specific length limit
            length_limit = self.title_length_limits.get(marketplace, self.max_length)
            
            # Prepare prompt for OpenAI
            prompt = self._create_variations_prompt(product, marketplace, num_variations, existing_title)
            
            # Generate variations using OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert e-commerce listing optimizer specializing in creating high-converting, SEO-friendly product titles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature + 0.1,  # Slightly higher temperature for more variation
                max_tokens=200,
                n=1
            )
            
            # Extract variations from response
            content = response.choices[0].message.content.strip()
            
            # Parse variations (assuming they're numbered or on separate lines)
            variations = []
            for line in content.split('\n'):
                # Remove numbering and other formatting
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                if clean_line and "Title" not in clean_line and len(clean_line) > 10:
                    variations.append(self._clean_title(clean_line, length_limit))
            
            # Ensure we have the requested number of variations
            while len(variations) < num_variations:
                # Generate a fallback variation
                fallback = self._generate_fallback_title(product, marketplace)
                if fallback not in variations:
                    variations.append(fallback)
            
            # Limit to requested number
            variations = variations[:num_variations]
            
            logger.info(f"Generated {len(variations)} title variations")
            return variations
        
        except Exception as e:
            logger.error(f"Error generating title variations: {e}")
            # Fallback to basic variations
            return [self._generate_fallback_title(product, marketplace) for _ in range(num_variations)]
    
    def _create_title_prompt(
        self,
        product: Product,
        marketplace: Marketplace,
        use_performance_data: bool,
        performance_data: Optional[ListingPerformance]
    ) -> str:
        """
        Create a prompt for title generation.
        
        Args:
            product: Product to generate title for
            marketplace: Target marketplace
            use_performance_data: Whether to use performance data
            performance_data: Performance data to use
            
        Returns:
            Prompt for OpenAI
        """
        # Get marketplace-specific length limit
        length_limit = self.title_length_limits.get(marketplace, self.max_length)
        
        prompt = f"""
        Create an optimized product title for {marketplace.value} marketplace.
        
        Product details:
        - Name: {product.name}
        - Brand: {product.brand or 'N/A'}
        - Model: {product.model or 'N/A'}
        - Category: {product.category or 'N/A'}
        - Condition: {product.condition}
        
        Key features:
        {', '.join(product.features[:5]) if product.features else 'N/A'}
        
        Title requirements:
        - Maximum length: {length_limit} characters
        - Include important keywords for search visibility
        - Follow {marketplace.value} best practices
        - Be compelling and descriptive
        - Include brand and model if available
        """
        
        # Add marketplace-specific guidance
        if marketplace == Marketplace.AMAZON:
            prompt += """
            Amazon title best practices:
            - Include brand, model number, product type, color, size, quantity
            - Capitalize first letter of each word
            - Use numerals instead of spelling out numbers
            - Don't include promotional phrases like "sale" or "free shipping"
            - Don't include seller information
            """
        elif marketplace == Marketplace.EBAY:
            prompt += """
            eBay title best practices:
            - Include specific details like brand, model, size, color
            - Use all 80 characters if possible
            - Include popular search terms
            - Avoid ALL CAPS and excessive punctuation
            - Don't use promotional phrases like "L@@K" or "WOW"
            """
        
        # Add performance data if available
        if use_performance_data and performance_data:
            prompt += f"""
            Performance data to consider:
            - Click-through rate: {performance_data.ctr:.4f}
            - Conversion rate: {performance_data.conversion_rate:.4f}
            - Top search terms: {', '.join(performance_data.search_terms[:5]) if performance_data.search_terms else 'N/A'}
            
            Optimize the title to improve these metrics.
            """
        
        prompt += """
        Return ONLY the optimized title text with no additional commentary or explanation.
        """
        
        return prompt
    
    def _create_variations_prompt(
        self,
        product: Product,
        marketplace: Marketplace,
        num_variations: int,
        existing_title: Optional[str]
    ) -> str:
        """
        Create a prompt for generating title variations.
        
        Args:
            product: Product to generate title for
            marketplace: Target marketplace
            num_variations: Number of variations to generate
            existing_title: Existing title to use as a base
            
        Returns:
            Prompt for OpenAI
        """
        # Get marketplace-specific length limit
        length_limit = self.title_length_limits.get(marketplace, self.max_length)
        
        prompt = f"""
        Create {num_variations} different variations of a product title for {marketplace.value} marketplace.
        
        Product details:
        - Name: {product.name}
        - Brand: {product.brand or 'N/A'}
        - Model: {product.model or 'N/A'}
        - Category: {product.category or 'N/A'}
        - Condition: {product.condition}
        
        Key features:
        {', '.join(product.features[:5]) if product.features else 'N/A'}
        """
        
        if existing_title:
            prompt += f"""
            Current title: {existing_title}
            
            Create variations that are significantly different from the current title while maintaining accuracy.
            Try different approaches such as:
            - Different word order
            - Alternative keywords
            - Emphasizing different features
            - Different tone or style
            """
        
        prompt += f"""
        Title requirements:
        - Maximum length: {length_limit} characters
        - Include important keywords for search visibility
        - Follow {marketplace.value} best practices
        - Be compelling and descriptive
        - Each variation should be distinctly different
        
        Return {num_variations} numbered title variations, one per line.
        """
        
        return prompt
    
    def _clean_title(self, title: str, max_length: int) -> str:
        """
        Clean and format a title.
        
        Args:
            title: Title to clean
            max_length: Maximum length
            
        Returns:
            Cleaned title
        """
        # Remove quotes if present
        title = title.strip('"\'')
        
        # Remove any markdown formatting
        title = re.sub(r'[*_#]', '', title)
        
        # Ensure proper capitalization (first letter of each word)
        title = ' '.join(word.capitalize() if not word.isupper() else word for word in title.split())
        
        # Truncate if too long
        if len(title) > max_length:
            title = title[:max_length].rsplit(' ', 1)[0]
        
        return title
    
    def _generate_fallback_title(self, product: Product, marketplace: Marketplace) -> str:
        """
        Generate a fallback title if the API fails.
        
        Args:
            product: Product to generate title for
            marketplace: Target marketplace
            
        Returns:
            Fallback title
        """
        # Get template for marketplace
        template = self.title_templates.get(marketplace, "{brand} {model} {category}")
        
        # Fill in template
        title = template
        title = title.replace("{brand}", product.brand or "")
        title = title.replace("{model}", product.model or "")
        title = title.replace("{category}", product.category or "")
        title = title.replace("{condition}", product.condition or "")
        
        # Add a key feature if available
        key_feature = product.features[0] if product.features else ""
        title = title.replace("{key_feature}", key_feature)
        
        # Clean up multiple spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Get marketplace-specific length limit
        length_limit = self.title_length_limits.get(marketplace, self.max_length)
        
        # Truncate if too long
        if len(title) > length_limit:
            title = title[:length_limit].rsplit(' ', 1)[0]
        
        return title


class DescriptionGenerator:
    """
    Generator for optimized listing descriptions.
    
    This class provides methods for generating optimized descriptions
    for product listings on various marketplaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DescriptionGenerator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_length = config.get('max_length', 2000)
        self.model = config.get('model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.7)
        self.include_bullets = config.get('include_bullets', True)
        self.include_specifications = config.get('include_specifications', True)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Load marketplace-specific description templates
        self.description_templates = {
            Marketplace.AMAZON: {
                "intro": "Introducing the {brand} {model} - {short_description}",
                "bullets": "Key Features:\n{bullets}",
                "details": "{detailed_description}",
                "specs": "Specifications:\n{specs}",
                "outro": "Buy with confidence from {brand}."
            },
            Marketplace.EBAY: {
                "intro": "# {brand} {model} - {short_description}",
                "bullets": "## Key Features\n{bullets}",
                "details": "{detailed_description}",
                "specs": "## Specifications\n{specs}",
                "outro": "## Shipping & Returns\nFast shipping and hassle-free returns. Buy with confidence!"
            }
        }
        
        logger.info("DescriptionGenerator initialized")
    
    async def generate_description(
        self,
        product: Product,
        marketplace: Marketplace,
        use_performance_data: bool = False,
        performance_data: Optional[ListingPerformance] = None
    ) -> str:
        """
        Generate an optimized description for a product.
        
        Args:
            product: Product to generate description for
            marketplace: Target marketplace
            use_performance_data: Whether to use performance data
            performance_data: Performance data to use
            
        Returns:
            Optimized description
        """
        logger.info(f"Generating description for product {product.id} on {marketplace}")
        
        try:
            # Prepare prompt for OpenAI
            prompt = self._create_description_prompt(product, marketplace, use_performance_data, performance_data)
            
            # Generate description using OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert e-commerce copywriter specializing in creating high-converting, SEO-friendly product descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                n=1
            )
            
            # Extract description from response
            description = response.choices[0].message.content.strip()
            
            # Clean up description
            description = self._clean_description(description, marketplace)
            
            logger.info(f"Generated description of {len(description)} characters")
            return description
        
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            # Fallback to template-based description
            return self._generate_fallback_description(product, marketplace)
    
    def _create_description_prompt(
        self,
        product: Product,
        marketplace: Marketplace,
        use_performance_data: bool,
        performance_data: Optional[ListingPerformance]
    ) -> str:
        """
        Create a prompt for description generation.
        
        Args:
            product: Product to generate description for
            marketplace: Target marketplace
            use_performance_data: Whether to use performance data
            performance_data: Performance data to use
            
        Returns:
            Prompt for OpenAI
        """
        prompt = f"""
        Create an optimized product description for {marketplace.value} marketplace.
        
        Product details:
        - Name: {product.name}
        - Brand: {product.brand or 'N/A'}
        - Model: {product.model or 'N/A'}
        - Category: {product.category or 'N/A'}
        - Condition: {product.condition}
        
        Features:
        {', '.join(product.features) if product.features else 'N/A'}
        
        Specifications:
        {json.dumps(product.specifications, indent=2) if product.specifications else 'N/A'}
        
        Existing description (if available):
        {product.description or 'N/A'}
        
        Description requirements:
        - Maximum length: {self.max_length} characters
        - Include important keywords for search visibility
        - Follow {marketplace.value} best practices
        - Be compelling and benefit-focused
        - Use proper formatting for readability
        """
        
        # Add marketplace-specific guidance
        if marketplace == Marketplace.AMAZON:
            prompt += """
            Amazon description best practices:
            - Use HTML formatting for structure
            - Include 5-7 bullet points highlighting key features
            - Focus on benefits, not just features
            - Include technical specifications in a structured format
            - Avoid promotional language or time-sensitive content
            """
        elif marketplace == Marketplace.EBAY:
            prompt += """
            eBay description best practices:
            - Use HTML or simple markdown for formatting
            - Include detailed information about condition
            - Highlight shipping and return policies
            - Use bullet points for key features
            - Include measurements and specifications
            """
        
        # Add performance data if available
        if use_performance_data and performance_data:
            prompt += f"""
            Performance data to consider:
            - Conversion rate: {performance_data.conversion_rate:.4f}
            - Add-to-cart rate: {performance_data.add_to_cart_rate:.4f}
            - Top search terms: {', '.join(performance_data.search_terms[:5]) if performance_data.search_terms else 'N/A'}
            
            Optimize the description to improve these metrics.
            """
        
        prompt += """
        Structure the description with:
        1. An engaging introduction
        2. Bullet points for key features
        3. Detailed description with benefits
        4. Technical specifications
        5. A call to action
        
        Return the formatted description ready for use on the marketplace.
        """
        
        return prompt
    
    def _clean_description(self, description: str, marketplace: Marketplace) -> str:
        """
        Clean and format a description.
        
        Args:
            description: Description to clean
            marketplace: Target marketplace
            
        Returns:
            Cleaned description
        """
        # Remove any unwanted formatting
        if marketplace == Marketplace.AMAZON:
            # Amazon allows HTML
            pass
        elif marketplace == Marketplace.EBAY:
            # eBay allows HTML and markdown
            pass
        else:
            # Remove HTML for other marketplaces
            description = re.sub(r'<[^>]+>', '', description)
        
        # Ensure proper spacing
        description = re.sub(r'\n{3,}', '\n\n', description)
        
        # Truncate if too long
        if len(description) > self.max_length:
            # Try to truncate at a paragraph break
            truncated = description[:self.max_length]
            last_para = truncated.rfind('\n\n')
            if last_para > 0:
                description = description[:last_para] + '\n\nSee product details for more information.'
            else:
                # If no paragraph break, truncate at a sentence
                last_sentence = truncated.rfind('. ')
                if last_sentence > 0:
                    description = description[:last_sentence+1] + ' See product details for more information.'
                else:
                    # Last resort: truncate at a word boundary
                    description = truncated.rsplit(' ', 1)[0] + '... See product details for more information.'
        
        return description
    
    def _generate_fallback_description(self, product: Product, marketplace: Marketplace) -> str:
        """
        Generate a fallback description if the API fails.
        
        Args:
            product: Product to generate description for
            marketplace: Target marketplace
            
        Returns:
            Fallback description
        """
        # Get template for marketplace
        template = self.description_templates.get(
            marketplace, 
            self.description_templates.get(Marketplace.AMAZON)
        )
        
        # Fill in template
        description = []
        
        # Intro
        intro = template["intro"]
        intro = intro.replace("{brand}", product.brand or "")
        intro = intro.replace("{model}", product.model or "")
        intro = intro.replace("{short_description}", product.name)
        description.append(intro)
        
        # Bullets
        if self.include_bullets and product.features:
            bullets = template["bullets"]
            bullet_points = "\n".join([f"- {feature}" for feature in product.features])
            bullets = bullets.replace("{bullets}", bullet_points)
            description.append(bullets)
        
        # Details
        details = template["details"]
        if product.description:
            detailed_description = product.description
        else:
            detailed_description = f"The {product.brand} {product.model} is a high-quality {product.category} designed to meet your needs."
        details = details.replace("{detailed_description}", detailed_description)
        description.append(details)
        
        # Specifications
        if self.include_specifications and product.specifications:
            specs = template["specs"]
            spec_points = "\n".join([f"- {key}: {value}" for key, value in product.specifications.items()])
            specs = specs.replace("{specs}", spec_points)
            description.append(specs)
        
        # Outro
        outro = template["outro"]
        outro = outro.replace("{brand}", product.brand or "us")
        description.append(outro)
        
        # Join all sections
        full_description = "\n\n".join(description)
        
        # Truncate if too long
        if len(full_description) > self.max_length:
            full_description = full_description[:self.max_length].rsplit('\n', 1)[0]
        
        return full_description


class KeywordOptimizer:
    """
    Optimizer for product listing keywords.
    
    This class provides methods for extracting and optimizing keywords
    for product listings on various marketplaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the KeywordOptimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_keywords = config.get('max_keywords', 20)
        self.min_keyword_length = config.get('min_keyword_length', 3)
        self.use_marketplace_data = config.get('use_marketplace_data', True)
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            # Fallback to smaller model if medium not available
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("Could not load spaCy model. Installing en_core_web_sm...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Load marketplace-specific keyword data
        self.marketplace_keywords = {
            Marketplace.AMAZON: set(["prime", "bestseller", "deal", "quality", "authentic"]),
            Marketplace.EBAY: set(["new", "fast shipping", "warranty", "authentic", "genuine"]),
            Marketplace.ETSY: set(["handmade", "custom", "vintage", "unique", "personalized"]),
            Marketplace.WALMART: set(["value", "quality", "affordable", "brand", "authentic"])
        }
        
        logger.info("KeywordOptimizer initialized")
    
    async def extract_keywords(
        self,
        product: Product,
        title: str,
        description: str,
        marketplace: Marketplace
    ) -> List[str]:
        """
        Extract and optimize keywords for a product listing.
        
        Args:
            product: Product to extract keywords for
            title: Listing title
            description: Listing description
            marketplace: Target marketplace
            
        Returns:
            List of optimized keywords
        """
        logger.info(f"Extracting keywords for product {product.id} on {marketplace}")
        
        try:
            # Combine text for analysis
            text = f"{product.name} {title} {description}"
            if product.brand:
                text += f" {product.brand}"
            if product.model:
                text += f" {product.model}"
            if product.category:
                text += f" {product.category}"
            if product.features:
                text += f" {' '.join(product.features)}"
            
            # Extract keywords using spaCy
            keywords_spacy = self._extract_keywords_spacy(text)
            
            # Extract keywords using OpenAI
            keywords_openai = await self._extract_keywords_openai(product, title, description, marketplace)
            
            # Combine keywords from both sources
            all_keywords = set(keywords_spacy + keywords_openai)
            
            # Add marketplace-specific keywords if applicable
            if self.use_marketplace_data and marketplace in self.marketplace_keywords:
                marketplace_specific = self.marketplace_keywords[marketplace]
                relevant_marketplace = [kw for kw in marketplace_specific if self._is_relevant(kw, product)]
                all_keywords.update(relevant_marketplace)
            
            # Filter and rank keywords
            ranked_keywords = self._rank_keywords(list(all_keywords), product, marketplace)
            
            # Limit to max keywords
            final_keywords = ranked_keywords[:self.max_keywords]
            
            logger.info(f"Extracted {len(final_keywords)} keywords")
            return final_keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Fallback to basic keywords
            return self._generate_fallback_keywords(product, marketplace)
    
    def _extract_keywords_spacy(self, text: str) -> List[str]:
        """
        Extract keywords using spaCy.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract noun phrases and named entities
        keywords = []
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) >= self.min_keyword_length:
                keywords.append(chunk.text.lower())
        
        # Add named entities
        for ent in doc.ents:
            if len(ent.text) >= self.min_keyword_length:
                keywords.append(ent.text.lower())
        
        # Add important single tokens (nouns, adjectives)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and len(token.text) >= self.min_keyword_length:
                keywords.append(token.text.lower())
        
        # Remove duplicates and clean
        clean_keywords = []
        seen = set()
        for kw in keywords:
            # Clean keyword
            clean_kw = re.sub(r'[^\w\s]', '', kw).strip()
            if clean_kw and clean_kw not in seen and len(clean_kw) >= self.min_keyword_length:
                clean_keywords.append(clean_kw)
                seen.add(clean_kw)
        
        return clean_keywords
    
    async def _extract_keywords_openai(
        self,
        product: Product,
        title: str,
        description: str,
        marketplace: Marketplace
    ) -> List[str]:
        """
        Extract keywords using OpenAI.
        
        Args:
            product: Product to extract keywords for
            title: Listing title
            description: Listing description
            marketplace: Target marketplace
            
        Returns:
            List of keywords
        """
        try:
            # Prepare prompt for OpenAI
            prompt = f"""
            Extract the most relevant SEO keywords for this product listing on {marketplace.value}.
            
            Product: {product.name}
            Brand: {product.brand or 'N/A'}
            Category: {product.category or 'N/A'}
            Title: {title}
            Description excerpt: {description[:500]}...
            
            Return ONLY a comma-separated list of the top 15-20 keywords or short phrases that shoppers might use to find this product.
            Focus on specific, relevant terms that would drive targeted traffic.
            """
            
            # Generate keywords using OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller model for keyword extraction
                messages=[
                    {"role": "system", "content": "You are an e-commerce SEO specialist who extracts relevant keywords from product listings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                n=1
            )
            
            # Extract keywords from response
            content = response.choices[0].message.content.strip()
            
            # Split by commas and clean
            keywords = [kw.strip().lower() for kw in content.split(',')]
            
            # Filter out keywords that are too short
            keywords = [kw for kw in keywords if len(kw) >= self.min_keyword_length]
            
            return keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords with OpenAI: {e}")
            return []
    
    def _rank_keywords(
        self,
        keywords: List[str],
        product: Product,
        marketplace: Marketplace
    ) -> List[str]:
        """
        Rank keywords by relevance.
        
        Args:
            keywords: List of keywords to rank
            product: Product to rank keywords for
            marketplace: Target marketplace
            
        Returns:
            Ranked list of keywords
        """
        # Create a scoring function
        def score_keyword(keyword: str) -> float:
            score = 0.0
            
            # Length score (prefer medium length keywords)
            length = len(keyword)
            if 5 <= length <= 15:
                score += 0.2
            elif 16 <= length <= 25:
                score += 0.1
            
            # Relevance to product
            if product.brand and product.brand.lower() in keyword.lower():
                score += 0.5
            if product.model and product.model.lower() in keyword.lower():
                score += 0.4
            if product.category and product.category.lower() in keyword.lower():
                score += 0.3
            
            # Check if keyword is in features
            for feature in product.features:
                if keyword.lower() in feature.lower():
                    score += 0.2
                    break
            
            # Marketplace-specific bonus
            if marketplace in self.marketplace_keywords and keyword in self.marketplace_keywords[marketplace]:
                score += 0.3
            
            # Add a small random factor to break ties
            score += np.random.uniform(0, 0.05)
            
            return score
        
        # Score and sort keywords
        scored_keywords = [(kw, score_keyword(kw)) for kw in keywords]
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked keywords
        return [kw for kw, _ in scored_keywords]
    
    def _is_relevant(self, keyword: str, product: Product) -> bool:
        """
        Check if a marketplace keyword is relevant to the product.
        
        Args:
            keyword: Keyword to check
            product: Product to check relevance for
            
        Returns:
            True if relevant, False otherwise
        """
        # Check if the keyword is relevant to the product category
        if keyword == "handmade" and product.category not in ["Handmade", "Crafts", "Art"]:
            return False
        if keyword == "vintage" and product.condition != "Used":
            return False
        
        # Most marketplace keywords are generally applicable
        return True
    
    def _generate_fallback_keywords(self, product: Product, marketplace: Marketplace) -> List[str]:
        """
        Generate fallback keywords if extraction fails.
        
        Args:
            product: Product to generate keywords for
            marketplace: Target marketplace
            
        Returns:
            List of fallback keywords
        """
        keywords = []
        
        # Add basic product info
        if product.brand:
            keywords.append(product.brand.lower())
        if product.model:
            keywords.append(product.model.lower())
        if product.category:
            keywords.append(product.category.lower())
        
        # Add condition
        keywords.append(product.condition.lower())
        
        # Add features
        for feature in product.features[:5]:
            # Extract key terms from feature
            terms = feature.lower().split()
            for term in terms:
                if len(term) >= self.min_keyword_length and term not in keywords:
                    keywords.append(term)
        
        # Add marketplace-specific keywords
        if marketplace in self.marketplace_keywords:
            keywords.extend(list(self.marketplace_keywords[marketplace])[:5])
        
        return keywords