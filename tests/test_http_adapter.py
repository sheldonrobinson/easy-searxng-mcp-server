"""Tests for SearXNG HTTP response handling."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searxng_mcp.adapters import HttpSearchAdapter  # noqa: E402
from searxng_mcp.client import (  # noqa: E402
    SearchError,
    SearchParameters,
    SearchQuery,
)


RESULT_HTML = """
<html><body>
  <div id="results">
    <div id="urls" role="main">
      <article class="result result-default category-general">
        <h3><a href="https://example.test/result">Example <em>result</em></a></h3>
        <p class="content">A <span>useful</span> description.</p>
      </article>
    </div>
  </div>
</body></html>
"""


class FakeResponse:
    def __init__(self, content_type, body="", status=200):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self.body

    async def json(self):
        content_type = self.headers["Content-Type"]
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise aiohttp.ContentTypeError(
                request_info=SimpleNamespace(
                    real_url="https://baresearch.org/search"
                ),
                history=(),
                status=self.status,
                message=(
                    "Attempt to decode JSON with unexpected mimetype: "
                    f"{content_type}"
                ),
            )
        return json.loads(self.body)


class FakeSession:
    def __init__(self, *responses):
        self.responses = responses
        self.requested_urls = []

    def get(self, url):
        self.requested_urls.append(url)
        return self.responses[len(self.requested_urls) - 1]


class HttpSearchAdapterTests(IsolatedAsyncioTestCase):
    async def search_with(self, *responses, parameters=None):
        adapter = HttpSearchAdapter("https://baresearch.org/")
        session = FakeSession(*responses)
        self.session = session
        with patch.object(
            adapter,
            "_get_session",
            new=AsyncMock(return_value=session),
        ):
            return await adapter.search(
                SearchQuery("test"),
                parameters or SearchParameters(max_results=5),
            )

    async def test_anubis_challenge_has_actionable_error(self):
        response = FakeResponse(
            "text/html; charset=utf-8",
            body=(
                "<title>Making sure you&#39;re not a bot!</title>"
                '<script id="anubis_challenge">{}</script>'
            ),
        )

        with self.assertRaises(SearchError) as raised:
            await self.search_with(response)

        message = str(raised.exception)
        self.assertIn("Anubis anti-bot challenge", message)
        self.assertIn("https://baresearch.org", message)
        self.assertIn("--embedded", message)
        self.assertEqual(len(self.session.requested_urls), 1)

    async def test_unrecognized_html_explains_json_requirement(self):
        response = FakeResponse(
            "text/html; charset=utf-8",
            body="<html><body>Not a SearXNG result page</body></html>",
        )

        with self.assertRaises(SearchError) as raised:
            await self.search_with(response)

        message = str(raised.exception)
        self.assertIn("instead of JSON", message)
        self.assertIn("search.formats", message)
        self.assertIn("--embedded", message)

    async def test_json_content_type_with_charset_is_accepted(self):
        payload = {
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.test/result",
                    "content": "Content",
                }
            ]
        }
        response = FakeResponse(
            "application/json; charset=utf-8",
            body=json.dumps(payload),
        )

        collection = await self.search_with(
            response,
            parameters=SearchParameters(max_results=5, safe_search=0),
        )

        self.assertEqual(len(collection.results), 1)
        self.assertEqual(collection.results[0].title, "Result")
        query = parse_qs(urlsplit(self.session.requested_urls[0]).query)
        self.assertEqual(query["safesearch"], ["0"])

    async def test_searxng_html_results_are_supported(self):
        response = FakeResponse("text/html; charset=utf-8", body=RESULT_HTML)

        collection = await self.search_with(response)

        self.assertEqual(len(collection.results), 1)
        result = collection.results[0]
        self.assertEqual(result.title, "Example result")
        self.assertEqual(result.url, "https://example.test/result")
        self.assertEqual(result.content, "A useful description.")

    async def test_json_forbidden_retries_browser_style_html_search(self):
        forbidden = FakeResponse(
            "text/html; charset=utf-8",
            body="<title>403 Forbidden</title>",
            status=403,
        )
        results = FakeResponse("text/html; charset=utf-8", body=RESULT_HTML)

        collection = await self.search_with(forbidden, results)

        self.assertEqual(len(collection.results), 1)
        self.assertEqual(len(self.session.requested_urls), 2)
        json_query = parse_qs(urlsplit(self.session.requested_urls[0]).query)
        html_query = parse_qs(urlsplit(self.session.requested_urls[1]).query)
        self.assertEqual(json_query["format"], ["json"])
        self.assertNotIn("format", html_query)
        self.assertEqual(html_query["theme"], ["simple"])

    async def test_rate_limit_is_not_retried(self):
        response = FakeResponse(
            "text/html; charset=utf-8",
            body="<title>429 Too Many Requests</title>",
            status=429,
        )

        with self.assertRaises(SearchError) as raised:
            await self.search_with(response)

        self.assertIn("rate limit", str(raised.exception).lower())
        self.assertEqual(len(self.session.requested_urls), 1)
