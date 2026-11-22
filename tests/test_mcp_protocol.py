"""Protocol-level regression tests for the low-level MCP request handlers."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import KonnekMCPServer  # noqa: E402
import searxng_mcp.server as compatibility_server  # noqa: E402
from searxng_mcp.adapters import SearchResult, SearchResultCollection  # noqa: E402
from searxng_mcp.client import SearchPort  # noqa: E402


class FakeSearchPort(SearchPort):
    """Return a deterministic result without requiring a SearXNG instance."""

    def __init__(self):
        self.calls = []

    async def search(self, query, parameters):
        self.calls.append((query, parameters))
        return SearchResultCollection(
            results=[
                SearchResult(
                    title="Riemann Hypothesis",
                    url="https://example.test/riemann",
                    content="A deterministic test search result.",
                )
            ]
        )


class MCPRequestHandlerTests(IsolatedAsyncioTestCase):
    async def _assert_list_and_call(self, client_streams):
        async with ClientSession(*client_streams) as client:
            await client.initialize()

            listed = await client.list_tools()
            self.assertEqual([tool.name for tool in listed.tools], ["web_search"])

            called = await client.call_tool(
                "web_search",
                {
                    "query": "Riemann Hypothesis",
                    "categories": ["general"],
                    "max_results": 1,
                },
            )
            self.assertFalse(called.is_error)
            self.assertIn("Riemann Hypothesis", called.content[0].text)
            self.assertIn("https://example.test/riemann", called.content[0].text)

    async def test_primary_server_accepts_list_and_call_params(self):
        search_port = FakeSearchPort()
        server = KonnekMCPServer(search_port=search_port).server

        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    server.run,
                    *server_streams,
                    server.create_initialization_options(),
                )
                try:
                    await self._assert_list_and_call(client_streams)
                finally:
                    task_group.cancel_scope.cancel()

        self.assertEqual(len(search_port.calls), 1)
        query, parameters = search_port.calls[0]
        self.assertEqual(query.text, "Riemann Hypothesis")
        self.assertEqual(parameters.categories, ("general",))
        self.assertEqual(parameters.max_results, 1)

    async def test_compatibility_stdio_server_accepts_list_and_call_params(self):
        search_port = FakeSearchPort()

        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):

            @asynccontextmanager
            async def fake_stdio_server():
                yield server_streams

            with (
                patch.object(
                    compatibility_server, "stdio_server", fake_stdio_server
                ),
                patch(
                    "searxng_mcp.adapters.HttpSearchAdapter",
                    return_value=search_port,
                ),
            ):
                async with anyio.create_task_group() as task_group:
                    task_group.start_soon(
                        compatibility_server.serve,
                        "https://unused.example.test",
                        30,
                        "stdio",
                    )
                    try:
                        await self._assert_list_and_call(client_streams)
                    finally:
                        task_group.cancel_scope.cancel()

        self.assertEqual(len(search_port.calls), 1)
        self.assertEqual(search_port.calls[0][0].text, "Riemann Hypothesis")
