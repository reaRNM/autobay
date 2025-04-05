"""
User agent rotation manager for web scraping.
"""

import random
from typing import List, Optional

# Common user agents for different browsers and platforms
DEFAULT_USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55",
]


class UserAgentManager:
    """
    Manages a pool of user agents for web scraping, with automatic rotation.
    """

    def __init__(self, user_agents: Optional[List[str]] = None):
        """
        Initialize the user agent manager.

        Args:
            user_agents: List of user agent strings. If None, default user agents will be used.
        """
        self.user_agents = user_agents or DEFAULT_USER_AGENTS
        self._last_used: Optional[str] = None

    def add_user_agent(self, user_agent: str) -> None:
        """
        Add a new user agent to the pool.

        Args:
            user_agent: User agent string to add
        """
        if user_agent not in self.user_agents:
            self.user_agents.append(user_agent)

    def remove_user_agent(self, user_agent: str) -> None:
        """
        Remove a user agent from the pool.

        Args:
            user_agent: User agent string to remove
        """
        if user_agent in self.user_agents:
            self.user_agents.remove(user_agent)
            if self._last_used == user_agent:
                self._last_used = None

    def get_random_user_agent(self) -> str:
        """
        Get a random user agent from the pool.

        Returns:
            str: A random user agent string
        """
        if not self.user_agents:
            raise ValueError("No user agents available")
            
        # Avoid using the same user agent twice in a row if possible
        if len(self.user_agents) > 1 and self._last_used:
            available = [ua for ua in self.user_agents if ua != self._last_used]
            user_agent = random.choice(available)
        else:
            user_agent = random.choice(self.user_agents)
            
        self._last_used = user_agent
        return user_agent

    def get_user_agents(self) -> List[str]:
        """
        Get all user agents in the pool.

        Returns:
            List[str]: All user agent strings
        """
        return self.user_agents.copy()