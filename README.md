# Konnek SearXNG MCP Server

<!-- mcp-name: io.github.sheldonrobinson/easy-searxng-mcp-server -->

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
- **Web Content Extraction**: Uses Trafilatura for HTML article extraction and
  optional MarkItDown conversion for supported documents and HTML fallback.
- **Smart VectorRAG Reranking**: Optional `--smart` semantic reranking with
  local CTranslate2 Hugging Face models or OpenAI-compatible embeddings.
- **Embedded SearXNG**: Optional embedded local SearXNG web app via
  `--embedded` — starts SearXNG in-process (background thread) and connects
  through `HttpSearchAdapter` on localhost. No separate server required.
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
cd <path-to-source>
./server_entry.sh [OPTIONS]
```

### Supported Options

| Flag | Description |
| :--- | :--- |
| `--version` | Print the package version and exit. The version is also included in `-h`/`--help` output. |
| `--transport` | MCP transport: `stdio` (default) or `streamable-http`. |
| `--embedded` | Start the SearXNG web app in-process on localhost, then connect via HttpSearchAdapter. No separate server required. |
| `--http` | Connect to a remote SearXNG instance via HTTP (requires `--searxng-url` or `SEARXNG_URL` env var). |
| `--searxng-url` | URL of the remote SearXNG instance (e.g., `http://127.0.0.1:11888`). |
| `--host` | Host to bind for streamable-http transport (default: 127.0.0.1). |
| `--port` | Port to bind for streamable-http transport (default: 8000). |
| `--sxng-port` | Port for embedded SearXNG server (default: 11888). |
| `--sxng-host` | Host for embedded SearXNG server (default: 127.0.0.1). |
| `--smart` | Enable VectorRAG reranking with local CTranslate2 embeddings, or remote embeddings when a URL is configured. |
| `--embeddings-url` | OpenAI-compatible embeddings endpoint URL. Use `openai` for the official OpenAI endpoint. |
| `--embeddings-model` | Remote model name or local pre-converted CTranslate2 Hugging Face slug. |
| `--embeddings-dim` | Requested/expected embedding dimension (default: 384). |

> Note: `--embedded` is the recommended way to run locally. It starts the
> SearXNG web app in-process and connects automatically. Using
> `--searxng-url=http://127.0.0.1:11888` achieves the same result
> but requires you to start the SearXNG server separately.

### Remote SearXNG requirements

A remote instance must permit automated search requests. The client prefers
JSON output and also supports results rendered with SearXNG's simple HTML
theme. If an instance rejects JSON with SearXNG's standard `403 Forbidden`
response because JSON is disabled, the client retries once as an equivalent
browser-style HTML search. To enable JSON explicitly, use:

```yaml
search:
  formats:
    - html
    - json
```

The MCP `safe_search` argument is forwarded to SearXNG as `safesearch`.
Browser access alone does not guarantee unattended access: Anubis or similar
anti-bot systems can return an HTML challenge with HTTP 200, and a rate-limited
instance can return HTTP 429. The client reports these responses explicitly;
it does not parse challenges as results, bypass access controls, or retry
rate-limited requests. Choose an instance that permits automated access or use
`--embedded`.

### VectorRAG (Optional)

Pass `--smart` to rerank search results by semantic similarity using LanceDB.
Install the `embeddings` extra (or the `all` extra for every optional feature):

```bash
uvx --from 'easy-searxng-mcp-server[embeddings]' easy-searxng-mcp-server --embedded --smart
```

Without `EMBEDDINGS_URL`, smart mode uses a local CTranslate2 encoder. The
default is `keisuke-miyako/all-MiniLM-L6-v2-ct2-int8`; override it with a
Hugging Face slug for any compatible, pre-converted encoder repository that
includes its tokenizer files. The model is downloaded into the Hugging Face
cache on first use, so initial use requires network access unless it is already
cached. Runtime model conversion and PyTorch are not required.

Examples of compatible 384-dimensional models include:

- `keisuke-miyako/all-MiniLM-L6-v2-ct2-int8` (mean pooling)
- `jncraton/bge-small-en-ct2-int8` (CLS pooling)
- `jncraton/granite-embedding-107m-multilingual-ct2-int8` (CLS pooling)

`EMBEDDINGS_POOLING=auto` recognizes those families. Set it to `mean` or `cls`
for another model when needed. Local vectors are L2-normalized. A local model's
actual output size must match `EMBEDDINGS_DIM` (384 by default).

| Env Var | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDINGS_URL` | unset (local CTranslate2) | OpenAI-compatible base URL or full `/embeddings` URL; `openai` selects `https://api.openai.com/v1/embeddings` |
| `EMBEDDINGS_MODEL` | `keisuke-miyako/all-MiniLM-L6-v2-ct2-int8` locally; server-selected remotely | Remote model name or local pre-converted Hugging Face slug |
| `EMBEDDINGS_DIM` | `384` | Remote requested dimension or local required output dimension |
| `EMBEDDINGS_POOLING` | `auto` | Local pooling: `auto`, `mean`, or `cls` |
| `OPENAI_API_KEY` | empty string | Optional Bearer token for remote embeddings |
| `RERANK_TOP_K` | `5` | Number of top results after reranking |

Remote OpenAI example:

```bash
export EMBEDDINGS_URL=openai
export EMBEDDINGS_MODEL=text-embedding-3-small
export OPENAI_API_KEY=your-api-key
uvx --from 'easy-searxng-mcp-server[embeddings]' easy-searxng-mcp-server --embedded --smart
```

For another OpenAI-compatible server, either a base URL or a full embeddings
path works; the client does not append `/embeddings` twice:

```bash
export EMBEDDINGS_URL=http://localhost:8080/v1
export EMBEDDINGS_MODEL=text-embedding-3-small
uvx --from 'easy-searxng-mcp-server[embeddings]' easy-searxng-mcp-server --embedded --smart
```

For compatibility with earlier releases, configuring `EMBEDDINGS_URL` also
enables remote reranking without `--smart`. Local reranking requires `--smart`.
If embedding or reranking fails, the adapter logs a warning and returns the
original SearXNG ranking.

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

## Crawled Content and Supported File Formats

When result crawling is enabled, the server uses
[Trafilatura](https://pypi.org/project/trafilatura/) to extract the main text
from HTML pages. [MarkItDown](https://pypi.org/project/markitdown/) is the
fallback HTML converter and converts supported non-HTML documents when the
`markitdown` extra is installed. BeautifulSoup remains a final HTML fallback.
Responses that cannot be converted return the search snippet without failing
the overall search.

The following MarkItDown format extras are available:

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
- `aiohttp` - Asynchronous HTTP fetching for the Tavily adapter
- `trafilatura` - Main-text extraction from crawled HTML
- `beautifulsoup4` - Final fallback for HTML extraction
- `markitdown` (optional) - Markdown conversion for documents and HTML
- `ctranslate2`, `huggingface-hub`, `transformers` (optional) - Local smart embeddings
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
