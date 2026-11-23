"""Regression tests for local and remote VectorRAG embeddings."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402
import tavily.adapter as adapter  # noqa: E402
from searxng_mcp.adapters import SearchResult, SearchResultCollection  # noqa: E402
from searxng_mcp.client import SearchParameters, SearchQuery  # noqa: E402
from shared.types import TavilyResponse, TavilyResult  # noqa: E402


class FakeEmbeddingResponse:
    status = 200

    def __init__(self, embeddings):
        self._embeddings = embeddings

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return {
            "data": [
                {"index": index, "embedding": embedding}
                for index, embedding in enumerate(self._embeddings)
            ]
        }

    async def text(self):
        return "embedding error"


class FakeEmbeddingSession:
    def __init__(self, embeddings):
        self.response = FakeEmbeddingResponse(embeddings)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


class FakeTokenizer:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {
            "input_ids": self.input_ids,
            "token_type_ids": [
                [0] * len(sequence) for sequence in self.input_ids
            ],
        }


class VectorRAGRemoteTests(IsolatedAsyncioTestCase):
    async def test_openai_alias_key_and_default_dimension(self):
        session = FakeEmbeddingSession([[0.25] * 384])
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}),
            patch.object(
                adapter.aiohttp, "ClientSession", return_value=session
            ),
        ):
            rag = adapter._VectorRAG(
                embeddings_url="openai",
                model="text-embedding-3-small",
            )
            vectors = await rag._embed(["hello"])

        self.assertEqual(vectors, [[0.25] * 384])
        self.assertEqual(len(session.requests), 1)
        url, kwargs = session.requests[0]
        self.assertEqual(url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(
            kwargs["headers"]["Authorization"], "Bearer test-secret"
        )
        self.assertEqual(kwargs["json"]["model"], "text-embedding-3-small")
        self.assertEqual(kwargs["json"]["dimensions"], 384)

    async def test_full_endpoint_is_not_suffixed_and_empty_key_is_allowed(self):
        session = FakeEmbeddingSession([[0.5]])
        environment = dict(os.environ)
        environment.pop("OPENAI_API_KEY", None)
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                adapter.aiohttp, "ClientSession", return_value=session
            ),
        ):
            rag = adapter._VectorRAG(
                embeddings_url="https://example.test/v1/embeddings/",
                model="remote-model",
                dim=1,
            )
            await rag._embed(["hello"])

        url, kwargs = session.requests[0]
        self.assertEqual(url, "https://example.test/v1/embeddings")
        self.assertNotIn("Authorization", kwargs["headers"])


class VectorRAGLocalTests(IsolatedAsyncioTestCase):
    async def test_context_cleanup_uses_owned_temporary_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            tmpdir = os.path.join(parent, "rag")
            os.mkdir(tmpdir)
            Path(tmpdir, "marker").write_text("data", encoding="utf-8")
            fake_db = SimpleNamespace()  # Deliberately has no path attribute.
            fake_lancedb = SimpleNamespace(connect=Mock(return_value=fake_db))

            with (
                patch.dict(sys.modules, {"lancedb": fake_lancedb}),
                patch.object(
                    adapter.tempfile, "mkdtemp", return_value=tmpdir
                ),
            ):
                async with adapter._VectorRAG() as rag:
                    self.assertIs(rag._db, fake_db)
                    self.assertTrue(os.path.exists(tmpdir))

            self.assertFalse(os.path.exists(tmpdir))
            self.assertIsNone(rag._db)
            self.assertIsNone(rag._tmpdir)

    @staticmethod
    def local_modules(model_path, encoder, tokenizer):
        snapshot_download = Mock(return_value=model_path)
        from_pretrained = Mock(return_value=tokenizer)
        modules = {
            "ctranslate2": SimpleNamespace(
                Encoder=Mock(return_value=encoder)
            ),
            "huggingface_hub": SimpleNamespace(
                snapshot_download=snapshot_download
            ),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)
            ),
        }
        return modules, snapshot_download, from_pretrained

    async def test_embed_dispatches_to_local_backend_without_http(self):
        rag = adapter._VectorRAG(
            embeddings_url=None,
            model="keisuke-miyako/all-MiniLM-L6-v2-ct2-int8",
            dim=384,
        )
        with (
            patch.object(
                rag, "_embed_local", Mock(return_value=[[0.0] * 384])
            ) as local_embed,
            patch.object(adapter.aiohttp, "ClientSession") as http_session,
        ):
            vectors = await rag._embed(["local text"])

        self.assertEqual(vectors, [[0.0] * 384])
        local_embed.assert_called_once_with(["local text"])
        http_session.assert_not_called()

    async def test_local_snapshot_mean_pooling_and_normalization(self):
        tokenizer = FakeTokenizer([[101, 5, 102], [101, 102]])
        hidden = np.array(
            [
                [[1.0, 1.0], [3.0, 0.0], [5.0, 0.0]],
                [[0.0, 2.0], [0.0, 4.0], [100.0, 100.0]],
            ],
            dtype=np.float32,
        )
        encoder = Mock()
        encoder.forward_batch.return_value = SimpleNamespace(
            last_hidden_state=hidden
        )
        modules, download, from_pretrained = self.local_modules(
            "C:/models/minilm", encoder, tokenizer
        )
        rag = adapter._VectorRAG(
            embeddings_url=None,
            model="keisuke-miyako/all-MiniLM-L6-v2-ct2-int8",
            dim=2,
        )
        with patch.dict(sys.modules, modules):
            vectors = await rag._embed(["first", "second"])

        download.assert_called_once_with(
            repo_id="keisuke-miyako/all-MiniLM-L6-v2-ct2-int8"
        )
        from_pretrained.assert_called_once_with("C:/models/minilm")
        modules["ctranslate2"].Encoder.assert_called_once_with(
            "C:/models/minilm", device="cpu"
        )
        tokenizer.calls[0][1]["truncation"] = tokenizer.calls[0][1][
            "truncation"
        ]
        self.assertTrue(tokenizer.calls[0][1]["truncation"])
        self.assertFalse(tokenizer.calls[0][1]["padding"])
        expected_first = np.array([3.0, 1.0 / 3.0])
        expected_first /= np.linalg.norm(expected_first)
        np.testing.assert_allclose(vectors[0], expected_first, rtol=1e-5)
        np.testing.assert_allclose(vectors[1], [0.0, 1.0], rtol=1e-5)
        self.assertEqual(
            encoder.forward_batch.call_args.kwargs["token_type_ids"],
            [[0, 0, 0], [0, 0]],
        )

    async def test_bge_auto_pooling_uses_cls(self):
        tokenizer = FakeTokenizer([[101, 5, 102]])
        hidden = np.array(
            [[[1.0, 1.0], [3.0, 0.0], [100.0, 100.0]]],
            dtype=np.float32,
        )
        encoder = Mock()
        encoder.forward_batch.return_value = SimpleNamespace(
            last_hidden_state=hidden
        )
        modules, _, _ = self.local_modules(
            "C:/models/bge", encoder, tokenizer
        )
        rag = adapter._VectorRAG(
            embeddings_url=None,
            model="jncraton/bge-small-en-ct2-int8",
            dim=2,
        )
        with patch.dict(sys.modules, modules):
            vectors = await rag._embed(["document"])

        np.testing.assert_allclose(
            vectors[0], [2**-0.5, 2**-0.5], rtol=1e-5
        )

    async def test_local_dimension_mismatch_is_actionable(self):
        tokenizer = FakeTokenizer([[101, 102]])
        encoder = Mock()
        encoder.forward_batch.return_value = SimpleNamespace(
            last_hidden_state=np.ones((1, 2, 2), dtype=np.float32)
        )
        modules, _, _ = self.local_modules(
            "C:/models/minilm", encoder, tokenizer
        )
        rag = adapter._VectorRAG(
            embeddings_url=None,
            model="keisuke-miyako/all-MiniLM-L6-v2-ct2-int8",
            dim=384,
        )
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "EMBEDDINGS_DIM=384.*actual.*2"),
        ):
            await rag._embed(["document"])

    async def test_smart_crawl_constructs_local_rag_without_a_url(self):
        rag = Mock()
        rag.__aenter__ = AsyncMock(return_value=rag)
        rag.__aexit__ = AsyncMock(return_value=None)
        rag.add_results = AsyncMock()
        rag.rerank = AsyncMock(
            return_value=[
                {
                    "url": "https://example.test/result",
                    "title": "Result",
                    "content": "Content",
                    "score": 0.0,
                }
            ]
        )
        with patch.object(adapter, "_VectorRAG", return_value=rag) as rag_type:
            response = await adapter.TavilyAdapter.crawl(
                search_results=[
                    {
                        "url": "https://example.test/result",
                        "title": "Result",
                        "content": "Content",
                    }
                ],
                query="query",
                smart=True,
                embeddings_model="org/local-ct2-model",
            )

        self.assertEqual(response.results[0].url, "https://example.test/result")
        self.assertIsNone(rag_type.call_args.kwargs["embeddings_url"])
        self.assertEqual(rag_type.call_args.kwargs["model"], "org/local-ct2-model")
        self.assertEqual(rag_type.call_args.kwargs["dim"], 384)
        rag.add_results.assert_awaited_once()


class VectorRAGConfigurationTests(IsolatedAsyncioTestCase):
    @staticmethod
    def invoke_main(arguments, environment):
        with (
            patch.object(
                main.sys, "argv", ["easy-searxng-mcp-server", *arguments]
            ),
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                main.TavilyAdapter, "configure_tavily_adapter"
            ) as configure,
            patch.object(main, "KonnekSearchPort"),
            patch.object(main, "KonnekMCPServer"),
            patch.object(
                main.asyncio,
                "run",
                side_effect=lambda coroutine: coroutine.close(),
            ),
        ):
            main.main()
        return configure.call_args.kwargs["tavily_config"]

    async def test_smart_cli_defaults_to_local_model_and_384_dimensions(self):
        config = self.invoke_main(["--smart"], {})
        self.assertTrue(config.smart)
        self.assertIsNone(config.embeddings_url)
        self.assertEqual(
            config.embeddings_model,
            "keisuke-miyako/all-MiniLM-L6-v2-ct2-int8",
        )
        self.assertEqual(config.embeddings_dim, 384)

    async def test_smart_cli_uses_explicit_remote_environment(self):
        config = self.invoke_main(
            ["--smart"],
            {
                "EMBEDDINGS_URL": "openai",
                "EMBEDDINGS_MODEL": "text-embedding-3-small",
                "EMBEDDINGS_DIM": "768",
            },
        )
        self.assertTrue(config.smart)
        self.assertEqual(config.embeddings_url, "openai")
        self.assertEqual(config.embeddings_model, "text-embedding-3-small")
        self.assertEqual(config.embeddings_dim, 768)

    async def test_primary_search_port_applies_configured_smart_reranking(self):
        original_results = SearchResultCollection(
            results=[
                SearchResult("First", "https://example.test/first", "one"),
                SearchResult("Second", "https://example.test/second", "two"),
            ]
        )
        http_adapter = Mock()
        http_adapter.search = AsyncMock(return_value=original_results)
        reranked = TavilyResponse(
            results=[
                TavilyResult(
                    title="Second",
                    url="https://example.test/second",
                    content="two",
                )
            ]
        )
        config = adapter.TavilyAdapterConfig(
            smart=True,
            embeddings_model="org/local-ct2-model",
            embeddings_dim=384,
        )
        adapter.TavilyAdapter.configure_tavily_adapter(config)
        try:
            with (
                patch.object(main, "HttpSearchAdapter", return_value=http_adapter),
                patch.object(
                    adapter.TavilyAdapter,
                    "crawl",
                    new=AsyncMock(return_value=reranked),
                ) as crawl,
            ):
                port = main.KonnekSearchPort("https://search.example.test")
                result = await port.search(
                    SearchQuery("query"), SearchParameters(max_results=2)
                )
        finally:
            adapter.TavilyAdapter.configure_tavily_adapter(
                adapter.TavilyAdapterConfig()
            )

        self.assertEqual(
            [item.title for item in result.results], ["Second"]
        )
        self.assertTrue(crawl.await_args.kwargs["smart"])
        self.assertEqual(
            crawl.await_args.kwargs["embeddings_model"],
            "org/local-ct2-model",
        )


if __name__ == "__main__":
    unittest.main()
