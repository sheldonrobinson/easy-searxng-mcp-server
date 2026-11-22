"""HTTP search adapter using aiohttp."""

import json
from typing import List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
from bs4 import BeautifulSoup

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

    def _search_url(
        self,
        query: SearchQuery,
        parameters: SearchParameters,
        json_output: bool,
    ) -> str:
        params = {
            "q": query.text,
            "categories": ",".join(parameters.categories),
            "engines": ",".join(parameters.engines),
            "language": parameters.language,
            "max_results": parameters.max_results,
            "safesearch": parameters.safe_search,
        }
        if parameters.time_range:
            params["time_range"] = parameters.time_range
        if json_output:
            params["format"] = "json"
        else:
            # The HTML parser supports SearXNG's simple theme.
            params["theme"] = "simple"
        return f"{self.instance_url}/search?{urlencode(params)}"

    @staticmethod
    async def _fetch(
        session: aiohttp.ClientSession, url: str
    ) -> Tuple[int, str, str]:
        async with session.get(url) as resp:
            content_type = resp.headers.get(
                "Content-Type", "unknown content type"
            )
            return resp.status, content_type, await resp.text()

    @staticmethod
    def _parse_html_results(
        body: str, max_results: int
    ) -> Optional[List[SearchResult]]:
        """Extract results from SearXNG's simple-theme HTML response."""
        soup = BeautifulSoup(body, "html.parser")
        if soup.find(id="results") is None:
            return None

        results = []
        for article in soup.select("#urls article.result"):
            title_link = article.select_one("h3 a[href]")
            if title_link is None:
                continue
            content_element = article.select_one("p.content")
            results.append(
                SearchResult(
                    title=title_link.get_text(" ", strip=True),
                    url=str(title_link.get("href", "")),
                    content=(
                        content_element.get_text(" ", strip=True)
                        if content_element is not None
                        else ""
                    ),
                )
            )
            if len(results) >= max_results:
                break
        return results

    def _anubis_error(self) -> SearchError:
        return SearchError(
            f"SearXNG instance {self.instance_url} returned an "
            "Anubis anti-bot challenge instead of search results. This "
            "instance blocks unattended API requests. Run with --embedded "
            "or set SEARXNG_URL to an instance that permits automated API "
            "access."
        )

    async def search(self, query: SearchQuery, parameters: SearchParameters):
        """Perform a search via HTTP to a SearXNG instance."""
        url = self._search_url(query, parameters, json_output=True)

        try:
            session = await self._get_session()
            status, content_type, body = await self._fetch(session, url)

            # SearXNG returns a plain 403 when JSON is not enabled. In that
            # case, retry once as the equivalent browser-style HTML search.
            if status == 403 and "403 forbidden" in body.lower():
                html_url = self._search_url(
                    query, parameters, json_output=False
                )
                status, content_type, body = await self._fetch(
                    session, html_url
                )
        except aiohttp.ClientError as exc:
            raise SearchError(f"Search request failed: {exc}") from exc

        if "anubis_challenge" in body.lower():
            raise self._anubis_error()
        if status == 429:
            raise SearchError(
                f"SearXNG instance {self.instance_url} rate limited the "
                "request (HTTP 429). The request was not retried. Try again "
                "later, use --embedded, or configure another instance."
            )
        if status != 200:
            raise SearchError(
                f"Search request failed with status {status}: {body}"
            )

        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type == "application/json":
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                raise SearchError(
                    f"SearXNG instance {self.instance_url} returned invalid "
                    "JSON."
                ) from exc
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

        if media_type == "text/html":
            results = self._parse_html_results(body, parameters.max_results)
            if results is not None:
                return SearchResultCollection(results=results)

        raise SearchError(
            f"SearXNG instance {self.instance_url} returned {content_type} "
            "instead of JSON or recognizable SearXNG HTML results. Enable "
            "'json' under search.formats in the instance settings and ensure "
            "automated API access is allowed, or run with --embedded."
        )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
