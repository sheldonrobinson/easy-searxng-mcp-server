"""Regression tests for the Tavily adapter and its content extraction."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tavily.adapter as adapter  # noqa: E402


class FakeResponse:
    def __init__(self, body, content_type, status=200):
        self.body = body
        self.headers = {"Content-Type": content_type}
        self.status = status
        self.charset = "utf-8"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def read(self):
        return self.body

    async def text(self):
        return self.body.decode(self.charset)


class FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, *_args, **_kwargs):
        return self.response


class FakeStreamInfo:
    def __init__(self, mimetype=None, charset=None, url=None):
        self.mimetype = mimetype
        self.charset = charset
        self.url = url


class TavilyAdapterImportTests(unittest.TestCase):
    def test_aiohttp_is_bound_for_annotations_and_runtime_use(self):
        self.assertIs(adapter.aiohttp, aiohttp)


class CrawledContentTests(IsolatedAsyncioTestCase):
    @staticmethod
    async def fetch(body, content_type, include_raw=False, content_length=2500):
        session = FakeSession(FakeResponse(body, content_type))
        return await adapter.TavilyAdapter._fetch_raw_content(
            session,
            "https://example.test/page",
            include_raw,
            content_length,
        )

    @staticmethod
    def markitdown_double(markdown):
        converter = Mock()
        converter.convert_stream.return_value = SimpleNamespace(
            markdown=markdown
        )
        return Mock(return_value=converter), converter

    async def test_html_uses_trafilatura_and_preserves_raw_content(self):
        html = b"<html><body><article>Article body</article></body></html>"
        extractor = SimpleNamespace(extract=Mock(return_value="Article body"))
        factory, _ = self.markitdown_double("Unused")

        with (
            patch.object(adapter, "trafilatura", extractor, create=True),
            patch.object(adapter, "MarkItDown", factory),
        ):
            content = await self.fetch(
                html, "text/html; charset=utf-8", include_raw=True
            )

        self.assertEqual(content["text"], "Article body")
        self.assertEqual(content["raw"], html.decode("utf-8"))
        extractor.extract.assert_called_once_with(
            html.decode("utf-8"),
            url="https://example.test/page",
            include_comments=False,
        )
        factory.assert_not_called()

    async def test_non_html_document_uses_markitdown_convert_stream(self):
        document = b"%PDF-1.7 test document"
        extractor = Mock()
        factory, converter = self.markitdown_double("# Converted document")

        with (
            patch.object(adapter, "trafilatura", extractor, create=True),
            patch.object(adapter, "MarkItDown", factory),
            patch.object(adapter, "StreamInfo", FakeStreamInfo, create=True),
        ):
            content = await self.fetch(document, "application/pdf")

        self.assertEqual(content["text"], "# Converted document")
        self.assertEqual(content["raw"], "")
        extractor.extract.assert_not_called()
        args, kwargs = converter.convert_stream.call_args
        self.assertEqual(args[0].read(), document)
        self.assertEqual(kwargs["stream_info"].mimetype, "application/pdf")
        self.assertEqual(kwargs["stream_info"].url, "https://example.test/page")

    async def test_html_falls_back_to_markitdown(self):
        html = b"<html><body><h1>Fallback</h1></body></html>"
        extractor = SimpleNamespace(extract=Mock(return_value=None))
        factory, converter = self.markitdown_double("# Fallback")

        with (
            patch.object(adapter, "trafilatura", extractor, create=True),
            patch.object(adapter, "MarkItDown", factory),
            patch.object(adapter, "StreamInfo", FakeStreamInfo, create=True),
        ):
            content = await self.fetch(html, "text/html")

        self.assertEqual(content["text"], "# Fallback")
        converter.convert_stream.assert_called_once()

    async def test_beautifulsoup_remains_final_html_fallback(self):
        html = (
            b"<html><body><script>ignore()</script><nav>Menu</nav>"
            b"<main>Main article text</main></body></html>"
        )

        with (
            patch.object(adapter, "trafilatura", None, create=True),
            patch.object(adapter, "MarkItDown", None),
        ):
            content = await self.fetch(html, "text/html")

        self.assertEqual(content["text"], "Main article text")

    async def test_extracted_text_is_truncated(self):
        html = b"<html><body>Article body</body></html>"
        extractor = SimpleNamespace(extract=Mock(return_value="Article body"))

        with (
            patch.object(adapter, "trafilatura", extractor, create=True),
            patch.object(adapter, "MarkItDown", None),
        ):
            content = await self.fetch(
                html, "text/html", content_length=4
            )

        self.assertEqual(content["text"], "Arti...")


if __name__ == "__main__":
    unittest.main()
