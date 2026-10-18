# Konnek SearXNG MCP Server

This project implements an MCP (Model Context Protocol) server for SearXNG integration,
providing web search functionality for AI agents.

## Purpose

This server is developed to serve as a dedicated, open, and reliable metasearch
solution for AI agents. Its goal is to provide a strong, high-quality search
backend that can replace or supplement commercial search APIs like Tavily, Bing
Search, and Brave Search.

## Features

- **MCP Compliance**: Implements the MCP protocol for search operations.
- **SearXNG Integration**: Uses the `searxng` Python library to interact with
  SearXNG instances.
- **Dual Search Methods**:
  - `search`: Basic search via SearXNG client.
  - `web_search`: Full-featured web search with categories, engines, language,
    time range, safe search, and result limits.
- **Tavily-Compatible Adapter**: Provides a Tavily-compatible interface
  (`TavilyAdapter`) that wraps SearXNG results in Tavily's response format.
- **Embedded SearXNG**: Optional embedded local SearXNG MCP server via
  `--embedded` flag — starts searxng's MCP server in-process (background
  thread) and connects via `HttpSearchAdapter` on localhost. No separate
  Flask server required.
- **Dual Transport Support**:
  - **stdio**: MCP over Standard Input/Output (default).
  - **streamable-http**: MCP over Streamable HTTP (Starlette + uvicorn).
- **PEP 8 compliant**: Code follows Python's style guide.
- **RealPython Project Layout**: Source code is organised under the `src`
  directory.

## Directory Structure

```
easy-searxng-mcp-server/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Core server implementation
│   ├── common/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration and logging
│   ├── shared/
│   │   ├── __init__.py
│   │   └── types.py            # Shared data types (TavilyResult, etc.)
│   ├── tavily/
│   │   ├── __init__.py
│   │   ├── adapter.py          # Tavily-compatible adapter
│   │   └── settings/
│   │       ├── __init__.py
│   │       └── tavily_defaults.yml
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── simplexng.py        # Embedded SearXNG server
│   │   └── settings/
│   │       ├── __init__.py
│   │       └── settings_template.yml
│   └── mcp_server/
│       ├── __init__.py
│       └── server.py           # MCP server with web_search tool
├── pyproject.toml              # Project configuration and dependencies
├── server_entry.sh             # Entry script
└── README.md                   # This file
```

## Usage

### Via uvx (recommended)

```bash
uvx --from easy-searxng-mcp-server easy-searxng-mcp-server [OPTIONS]
```

### Direct execution from source

```bash
cd /resources/Projects/easy-searxng-mcp-server
./server_entry.sh [OPTIONS]
```

### Supported Options

| Flag | Description |
| :--- | :--- |
| `--transport` | MCP transport: `stdio` (default) or `streamable-http`. |
| `--embedded` | Start searxng MCP server in-process on localhost, then connect via HttpSearchAdapter. No separate Flask server required. |
| `--http` | Connect to a remote SearXNG instance via HTTP (requires `--searxng-url` or `SEARXNG_URL` env var). |
| `--searxng-url` | URL of the remote SearXNG instance (e.g., `http://127.0.0.1:11888`). |
| `--host` | Host to bind for streamable-http transport (default: 127.0.0.1). |
| `--port` | Port to bind for streamable-http transport (default: 8000). |
| `--sxng-port` | Port for embedded SearXNG server (default: 11888). |
| `--sxng-host` | Host for embedded SearXNG server (default: 127.0.0.1). |
| `--embeddings-url` | OpenAI-compatible embeddings endpoint URL for VectorRAG reranking. |
| `--embeddings-model` | Embedding model name (omit for llama-server to use loaded model). |
| `--embeddings-dim` | Embedding dimension (omit for llama-server to use model's n_embd). |

> Note: `--embedded` is the recommended way to run locally. It starts the
> SearXNG MCP server in-process and connects automatically. Using
> `--searxng-url=http://127.0.0.1:11888` achieves the same result
> but requires you to start the SearXNG server separately.

### VectorRAG (Optional)

When the `EMBEDDINGS_URL` environment variable is set, the server uses **VectorRAG** to rerank search results by semantic similarity using LanceDB + OpenAI-compatible embeddings.

| Env Var | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDINGS_URL` | — | OpenAI-compatible embeddings endpoint URL (required for VectorRAG) |
| `EMBEDDINGS_MODEL` | — | Embedding model name (omit for llama-server to use loaded model) |
| `EMBEDDINGS_DIM` | — | Embedding dimension (omit for llama-server to use model's n_embd) |
| `RERANK_TOP_K` | `5` | Number of top results after reranking |

Example:
```bash
export EMBEDDINGS_URL=http://localhost:8080/v1
uvx --from easy-searxng-mcp-server easy-searxng-mcp-server --embedded
```

### Examples

**Embedded SearXNG with stdio transport (recommended for local use):**
```bash
./server_entry.sh --embedded --transport stdio
```
Then enter MCP requests in JSON format:
```json
{"method": "search", "query": "python programming"}
```

**Embedded SearXNG with streamable-http transport:**
```bash
./server_entry.sh --embedded --transport streamable-http --host 127.0.0.1 --port 8000
```
MCP endpoint: `http://127.0.0.1:8000/mcp`

**Web Search with parameters:**
```json
{
  "method": "web_search",
  "query": "artificial intelligence",
  "categories": ["general", "news"],
  "engines": ["google", "bing", "duckduckgo"],
  "language": "en",
  "max_results": 10,
  "time_range": "week",
  "safe_search": 1
}
```

**Remote Mode with streamable-http:**
```bash
export SEARXNG_URL="http://your-searxng-instance:8080"
./server_entry.sh --transport streamable-http --host 0.0.0.0 --port 8000
```

Or with explicit URL:
```bash
./server_entry.sh --searxng-url "http://your-searxng-instance:8080" --transport streamable-http
```

### Environment Variables

| Variable | Description |
| :--- | :--- |
| `SEARXNG_URL` | URL of the remote SearXNG instance. |
| `TAVILY_TIMEOUT` | Tavily adapter HTTP timeout (default: 10). |
| `TAVILY_USER_AGENT` | User agent for Tavily adapter. |
| `TAVILY_MAX_RESULTS` | Max results for Tavily adapter (default: 10). |
| `TAVILY_CONTENT_LENGTH` | Content length for Tavily adapter (default: 2500). |

## Supported File Formats

The server uses [markitdown](https://github.com/microsoft/markitdown) for converting various file formats to text. When enabled, the following formats are supported:

| Format | Installation | Description |
| :--- | :--- | :--- |
| `[pptx]` | `pip install 'easy-searxng-mcp-server[pptx]'` | PowerPoint files |
| `[docx]` | `pip install 'easy-searxng-mcp-server[docx]'` | Word files |
| `[xlsx]` | `pip install 'easy-searxng-mcp-server[xlsx]'` | Excel files (newer) |
| `[xls]` | `pip install 'easy-searxng-mcp-server[xls]'` | Excel files (older) |
| `[pdf]` | `pip install 'easy-searxng-mcp-server[pdf]'` | PDF files |
| `[outlook]` | `pip install 'easy-searxng-mcp-server[outlook]'` | Outlook messages |
| `[audio-transcription]` | `pip install 'easy-searxng-mcp-server[audio-transcription]'` | Audio transcription (wav, mp3) using FasterWhisper |
| `[youtube-transcription]` | `pip install 'easy-searxng-mcp-server[youtube-transcription]'` | YouTube video transcription |

To install all optional dependencies:
```bash
pip install 'easy-searxng-mcp-server[all-formats]'
```

Or via uvx:
```bash
uvx --from 'easy-searxng-mcp-server[all-formats]' easy-searxng-mcp-server
```

Audio transcription uses [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) for high-performance speech recognition. The Whisper base model is bundled with the package, so no internet download is required. The model size can be configured via the `WHISPER_MODEL` environment variable (default: `base`, falls back to downloaded model if bundled not found).

## Dependencies

- Python 3.8+
- `searxng` - Core SearXNG client and MCP server
- `uvicorn`, `starlette`, `sse-starlette`, `anyio` - Streamable HTTP transport
- `flask`, `waitress` - Web server for embedded mode
- `aiohttp`, `beautifulsoup4` - Web scraping for Tavily adapter
- `pydantic` - Data validation
- `pyyaml`, `platformdirs` - Configuration
- `mcp-utils`, `strif`, `queuelib` - MCP and queue utilities

Install with:
```bash
pip install -r requirements.txt
```

## MCP Tools

### `web_search`

Performs a web search using SearXNG with customizable parameters.

**Parameters:**
- `query` (required): Search query string
- `categories`: Search categories (e.g., `["general", "news"]`)
- `engines`: Search engines (e.g., `["google", "bing", "duckduckgo"]`)
- `language`: Language code (default: `"auto"`)
- `max_results`: Maximum results (default: 5, range: 1-50)
- `time_range`: Time filter (`day`, `week`, `month`, `year`, `all`)
- `safe_search`: Safe search level (0=normal, 1=moderate, 2=strict)

## Development

The project follows PEP 8 guidelines and the RealPython project layout
with source code in the `src` directory.
