"""HTTP search adapter using aiohttp."""

import aiohttp
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from .client import SearchError, SearchParameters, SearchQuery, SearchPort


class SearchResult:
    """Represents a single search result."""

    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content


class SearchResultCollection:
    """Collection of search results."""

    def __init__(self, results: List[SearchResult]):
        self.results = results


class HttpSearchAdapter(SearchPort):
    """HTTP-based search adapter that delegates to a SearXNG instance."""

    def __init__(self, instance_url: str, timeout: int = 10):
        self.instance_url = instance_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def search(self, query: SearchQuery, parameters: SearchParameters):
        """Perform a search via HTTP to a SearXNG instance."""
        params = {
            "q": query.text,
            "categories": ",".join(parameters.categories),
            "engines": ",".join(parameters.engines),
            "language": parameters.language,
            "max_results": parameters.max_results,
        }
        if parameters.time_range:
            params["time_range"] = parameters.time_range

        url = f"{self.instance_url}/search?{urlencode(params)}"

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise SearchError(
                        f"Search request failed with status {resp.status}: {text}"
                    )
                data = await resp.json()
        except aiohttp.ClientError as exc:
            raise SearchError(f"Search request failed: {exc}") from exc

        results = []
        for item in data.get("results", [])[: parameters.max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                )
            )

        return SearchResultCollection(results=results)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
