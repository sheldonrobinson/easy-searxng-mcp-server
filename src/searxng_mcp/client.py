"""MCP client types - SearchError, SearchParameters, SearchPort, SearchQuery."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


class SearchError(Exception):
    """Raised when a search operation fails."""

    pass


@dataclass(frozen=True)
class SearchQuery:
    """Value object representing a search query."""

    text: str

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Search query cannot be empty")


@dataclass(frozen=True)
class SearchParameters:
    """Value object representing search parameters."""

    categories: Tuple[str, ...] = ("general",)
    engines: Tuple[str, ...] = ("google", "bing", "duckduckgo")
    language: str = "auto"
    max_results: int = 10
    time_range: Optional[str] = None


class SearchPort:
    """Abstract interface for search operations."""

    async def search(self, query: SearchQuery, parameters: SearchParameters):
        """Perform a search and return results."""
        raise NotImplementedError

    async def close(self):
        """Close any resources held by the search port."""
        pass
