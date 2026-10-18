"""SearXNG MCP compatibility modules for searxng-engine package.

Provides SearchError, SearchParameters, SearchPort, SearchQuery, HttpSearchAdapter,
and serve() that were previously from the searxng PyPI package.
"""

from .client import SearchError, SearchParameters, SearchPort, SearchQuery
from .adapters import HttpSearchAdapter
from .server import serve

__all__ = [
    "SearchError",
    "SearchParameters",
    "SearchPort",
    "SearchQuery",
    "HttpSearchAdapter",
    "serve",
]
