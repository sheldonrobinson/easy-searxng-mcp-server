"""
Tavily-compatible client for SearXNG.
Provides same interface as tavily-python package but uses SearXNG backend.
See https://docs.tavily.com/documentation/api-reference/introduction
"""

import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import numpy as np
import yaml
from bs4 import BeautifulSoup
try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None
from pydantic import BaseModel

from shared.types import TavilyResult, TavilyResponse


class TavilyAdapterConfig(BaseModel):
    """Configuration loader for Tavily adapter."""

    timeout: Optional[int] = 10
    user_agent: Optional[str] = "Mozilla/5.0 (compatible; SearchBot/1.0)"
    settings: Optional[str] = None
    # VectorRAG / embeddings options
    embeddings_url: Optional[str] = None
    embeddings_model: Optional[str] = None
    embeddings_dim: Optional[int] = 0
    rerank_top_k: Optional[int] = 5


def get_tavily_config(
    cfg: TavilyAdapterConfig = TavilyAdapterConfig(),
) -> TavilyAdapterConfig:
    """Load configuration from unified YAML file."""
    default_settings = (
        Path(__file__).parent / "settings" / "tavily_defaults.yml"
    )
    defaults = yaml.safe_load(default_settings.read_text())
    if cfg.settings:
        config_path = Path(cfg.settings)
        if config_path.exists():
            settings = yaml.safe_load(config_path.read_text())
            app_settings = (
                settings["crawler"] if settings.get("crawler") else defaults["crawler"]
            )
            return TavilyAdapterConfig(
                timeout=int(
                    cfg.timeout if cfg.timeout else app_settings["timeout"]
                ),
                user_agent=cfg.user_agent
                if cfg.user_agent
                else app_settings["user_agent"],
                settings=str(config_path),
                embeddings_url=cfg.embeddings_url or os.getenv("EMBEDDINGS_URL"),
                embeddings_model=cfg.embeddings_model or os.getenv(
                    "EMBEDDINGS_MODEL", "text-embedding-3-small"
                ),
                embeddings_dim=cfg.embeddings_dim or int(
                    os.getenv("EMBEDDINGS_DIM", "1536")
                ),
                rerank_top_k=cfg.rerank_top_k or int(
                    os.getenv("RERANK_TOP_K", "5")
                ),
            )

    return TavilyAdapterConfig(
        timeout=int(os.getenv("TAVILY_TIMEOUT", str(cfg.timeout))),
        user_agent=str(
            os.getenv("TAVILY_USER_AGENT", cfg.user_agent or "")
        ),
        settings="",
        embeddings_url=cfg.embeddings_url or os.getenv("EMBEDDINGS_URL"),
        embeddings_model=os.getenv("EMBEDDINGS_MODEL") or cfg.embeddings_model,
        embeddings_dim=cfg.embeddings_dim or int(os.getenv("EMBEDDINGS_DIM", "0")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", str(cfg.rerank_top_k or 5))),
    )




class _FasterWhisperAudioConverter:
    """Custom audio converter for markitdown using FasterWhisper."""
    
    # Bundled model path (relative to package)
    BUNDLED_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "..", 
        "models", 
        "models--Systran--faster-whisper-base"
    )
    
    ACCEPTED_FILE_EXTENSIONS = [".wav", ".mp3", ".m4a", ".mp4"]
    ACCEPTED_MIME_TYPE_PREFIXES = [
        "audio/x-wav",
        "audio/mpeg",
        "video/mp4",
    ]
    
    def __init__(self):
        self._model = None
        self._model_size = os.getenv("WHISPER_MODEL", "base")
        self._model_path = os.getenv("WHISPER_MODEL_PATH", self.BUNDLED_MODEL_PATH)
    
    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            
            # Try bundled model first, fallback to download
            if os.path.exists(self._model_path):
                model_path = self._model_path
            else:
                model_path = self._model_size
            
            self._model = WhisperModel(
                model_path, 
                device="cpu", 
                compute_type="int8"
            )
        return self._model
    
    def accepts(self, file_stream, stream_info, **kwargs):
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()
        
        if extension in self.ACCEPTED_FILE_EXTENSIONS:
            return True
        
        for prefix in self.ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True
        
        return False
    
    def convert(self, file_stream, stream_info, **kwargs):
        from markitdown._base_converter import DocumentConverterResult
        
        try:
            file_stream.seek(0)
            model = self._get_model()
            segments, info = model.transcribe(
                file_stream, 
                language=kwargs.get("language", None)
            )
            text = ' '.join(s.text for s in segments if s.text.strip())
            
            result = f"### Audio Transcript:\n{text}" if text else "[No speech detected]"
            return DocumentConverterResult(markdown=result)
        except Exception as e:
            return DocumentConverterResult(markdown=f"[Audio transcription failed: {e}]")


class _VectorRAG:
    """Lightweight VectorRAG using LanceDB + OpenAI-compatible embeddings."""

    def __init__(
        self,
        embeddings_url: str,
        model: str = "",
        dim: int = 0,
        api_key: str = "",
    ):
        self._embeddings_url = embeddings_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._api_key = api_key
        self._db = None
        self._table = None

    async def __aenter__(self):
        import lancedb

        tmpdir = tempfile.mkdtemp(prefix="easy-searxng-rag_")
        self._db = lancedb.connect(tmpdir)
        return self

    async def __aexit__(self, *exc):
        if self._db:
            import shutil
            shutil.rmtree(self._db.path, ignore_errors=True)
        self._db = None
        self._table = None

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI-compatible embeddings endpoint."""
        url = f"{self._embeddings_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with aiohttp.ClientSession() as session:
            body = {
                "input": texts,
                "encoding_format": "float",
            }
            if self._model:
                body["model"] = self._model
            if self._dim > 0:
                body["dimensions"] = self._dim
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Embeddings endpoint returned {resp.status}: {text}"
                    )
                data = await resp.json()

        # OpenAI-compatible response format
        items = data.get("data", [])
        if len(items) != len(texts):
            raise RuntimeError(
                f"Embeddings count mismatch: got {len(items)}, expected {len(texts)}"
            )
        return [item["embedding"] for item in items]

    async def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Embed and store search results in the vector index."""
        if not results:
            return

        texts = [
            f"{r.get('title', '')} {r.get('content', '')} {r.get('url', '')}"
            for r in results
        ]
        vectors = await self._embed(texts)

        rows = []
        for i, r in enumerate(results):
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "text": texts[i],
                    "score": float(r.get("score", 0)),
                    "vector": vectors[i],
                }
            )

        self._table = self._db.create_table("results", rows)

    async def rerank(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Embed the query and return results sorted by cosine similarity."""
        q_vec = await self._embed([query])
        query_embedding = q_vec[0]

        # LanceDB expects numpy array for search
        query_array = np.array(query_embedding, dtype=np.float32)
        result = self._table.search(query_array).limit(top_k).to_list()
        
        # Sort by _distance ascending (closest first)
        result.sort(key=lambda r: r.get("_distance", 0.0))
        return result


class TavilyAdapter:
    """Tavily-compatible adapter that uses SearXNG as backend."""

    __instance: Optional["TavilyAdapter"] = None
    __adapter_config: Optional[TavilyAdapterConfig] = TavilyAdapterConfig()

    def __init__(
        self,
        api_key: str = "",
        search_func: Optional[Any] = None,
    ):
        self.api_key = api_key  # Not used, but kept for compatibility
        self._search_func = search_func

    @property
    def search_func(self) -> Optional[Any]:
        return self._search_func

    @search_func.setter
    def search_func(self, value: Any) -> None:
        self._search_func = value

    @search_func.deleter
    def search_func(self) -> None:
        del self._search_func

    @staticmethod
    async def _fetch_raw_content(
        session: aiohttp.ClientSession,
        url: str,
        include_raw_content: bool,
        content_length: int,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0",
    ) -> Optional[Dict[str, str]]:
        """Scrap a page and return the first N characters of text.
        
        Uses markitdown for HTML parsing when available, falling back to
        BeautifulSoup.
        """
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": user_agent},
            ) as response:
                if response.status != 200:
                    return None

                html = await response.text()

                # Try markitdown first
                text = ""
                if MarkItDown is not None:
                    try:
                        import io
                        md = MarkItDown()
                        # Enable plugins for additional format support
                        md.enable_plugins()
                        # Register custom FasterWhisper audio converter
                        try:
                            from faster_whisper import WhisperModel
                            audio_converter = _FasterWhisperAudioConverter()
                            md.register_converter(audio_converter)
                        except ImportError:
                            pass  # faster-whisper not available
                        result = md.convert(io.BytesIO(html.encode("utf-8")))
                        text = result.text_content if hasattr(result, "text_content") else str(result)
                    except Exception:
                        pass  # Fall through to BeautifulSoup

                # Fallback to BeautifulSoup
                if not text:
                    try:
                        soup = BeautifulSoup(html, "html.parser")
                        # Remove unnecessary elements
                        for tag in soup(
                            ["script", "style", "nav", "header", "footer", "aside"]
                        ):
                            tag.decompose()
                        text = soup.get_text(strip=True, separator=" ")
                    except Exception:
                        text = ""

                # Crop to specified size
                if len(text) > content_length:
                    text = text[:content_length] + "..."

                return {
                    "url": url,
                    "text": text,
                    "raw": html if include_raw_content else "",
                }
        except Exception:
            return None

    @staticmethod
    async def crawl(
        search_results: List[Dict[str, Any]],
        query: str,
        max_results: int = 10,
        scrape_content: bool = False,
        include_raw_content: bool = False,
        content_length: int = 2500,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0",
        embeddings_url: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        embeddings_dim: int = 0,
        rerank_top_k: int = 5,
    ) -> TavilyResponse:
        """
        Convert SearXNG results to Tavily-compatible response.

        Args:
            search_results: Raw search results from SearXNG.
            query: Original search query.
            max_results: Maximum number of results to return.
            scrape_content: Whether to scrape page content.
            include_raw_content: Whether to include raw HTML.
            content_length: Maximum content length per result.
            timeout: HTTP timeout for scraping.
            user_agent: User agent string for scraping.
            embeddings_url: OpenAI-compatible embeddings endpoint URL.
                When provided, VectorRAG re-ranking is applied.
            embeddings_model: Embedding model name (default: text-embedding-3-small).
            embeddings_dim: Embedding dimension (default: 1536).
            rerank_top_k: Number of top results to return after reranking.

        Returns:
            A TavilyResponse.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        accepted_results = search_results[:max_results]
        raw_contents: Optional[Dict[str, tuple]] = None

        if scrape_content and accepted_results:
            urls_to_scrape = [
                r.get("url", "") for r in accepted_results if r.get("url")
            ]
            async with aiohttp.ClientSession() as scrape_session:
                tasks = [
                    TavilyAdapter._fetch_raw_content(
                        scrape_session, url, include_raw_content, content_length,
                        timeout, user_agent,
                    )
                    for url in urls_to_scrape
                ]
                page_contents = await asyncio.gather(
                    *tasks, return_exceptions=True
                )
                raw_contents = {
                    content["url"]: (content["text"], content["raw"])
                    for content in page_contents
                    if content and isinstance(content, dict)
                }

        # --- VectorRAG re-ranking ------------------------------------------------
        if embeddings_url and accepted_results:
            try:
                async with _VectorRAG(
                    embeddings_url=embeddings_url,
                    model=embeddings_model,
                    dim=embeddings_dim,
                ) as rag:
                    await rag.add_results(accepted_results)
                    reranked = await rag.rerank(query, top_k=rerank_top_k)

                    # Map reranked LanceDB rows back to TavilyResult
                    ordered: List[Dict[str, Any]] = []
                    for row in reranked:
                        url = row.get("url", "")
                        # Build the same result dict shape as the non-rag path
                        text_result = ""
                        raw_result = ""
                        if raw_contents and raw_contents.get(url):
                            if len(raw_contents[url]) > 0:
                                text_result = raw_contents[url][0] or ""
                            if len(raw_contents[url]) > 1:
                                raw_result = raw_contents[url][1] or ""

                        ordered.append(
                            {
                                "url": url,
                                "title": row.get("title", ""),
                                "content": row.get("content", ""),
                                "score": row.get("score", 0.0),
                                "text": text_result,
                                "raw_content": raw_result,
                                "_distance": row.get("_distance", 0.0),
                            }
                        )
                    accepted_results = ordered
                    logger = __import__("logging").getLogger(__name__)
                    logger.info(
                        "VectorRAG reranked %d results (top_k=%d)",
                        len(accepted_results),
                        rerank_top_k,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                # Graceful fallback: proceed with original ranking
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    "VectorRAG reranking failed, falling back to original ranking: %s",
                    exc,
                    exc_info=True,
                )

        # --- Build results -------------------------------------------------------
        results = []
        n_results = max(len(accepted_results), 1)
        
        for i, result in enumerate(accepted_results):
            if not result or not result.get("url"):
                continue

            # In reranked path, result is a dict with text/raw_content already set
            if "text" in result:
                text_result = result.get("text", "")
                raw_result = result.get("raw_content", "")
            else:
                url = result["url"]
                text_result = ""
                raw_result = ""
                if raw_contents and raw_contents.get(url):
                    if len(raw_contents[url]) > 0:
                        text_result = raw_contents[url][0] or ""
                    if len(raw_contents[url]) > 1:
                        raw_result = raw_contents[url][1] or ""

            # Smart score calculation with multiple signals:
            # 1. Original SearXNG score (if available)
            # 2. VectorRAG cosine similarity (if reranked)
            # 3. Keyword relevance boost
            # 4. Positional decay
            
            base_score = 0.0
            original_score = result.get("score")
            distance = result.get("_distance")
            
            if original_score and original_score > 0:
                base_score = float(original_score)
            elif distance is not None and distance >= 0:
                # Convert LanceDB _distance to similarity: similarity = 1 - distance
                base_score = max(0.0, min(1.0, 1.0 - float(distance)))
            
            # Keyword relevance boost: count query word matches in title/content
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            text_to_check = f"{result.get('title', '')} {result.get('content', '')}".lower()
            keyword_matches = len(query_words & set(re.findall(r'\b\w+\b', text_to_check)))
            keyword_boost = min(keyword_matches * 0.05, 0.3)  # Max 30% boost
            
            # Positional decay: exponential falloff
            # Uses 0.85^i so position 0 = 1.0, position 5 ≈ 0.44, position 10 ≈ 0.20
            position_decay = 0.85 ** i
            
            # Content length bonus: longer content slightly boosts score
            content_len = len(result.get("content", ""))
            length_bonus = min(content_len / 5000, 0.1)  # Max 10% bonus at 5000+ chars
            
            # Final score combines all signals, normalized to [0, 1]
            score = min(1.0, max(0.0, (base_score * 0.5) + keyword_boost + position_decay * 0.3 + length_bonus))

            results.append(
                TavilyResult(
                    url=result["url"],
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    score=score,
                    text=text_result,
                    raw_content=raw_result,
                )
            )

        response_time = time.time() - start_time

        return TavilyResponse(
            query=query,
            follow_up_questions=None,
            answer=None,
            images=[],
            results=results,
            response_time=response_time,
            request_id=request_id,
        )

    @staticmethod
    def configure_tavily_adapter(
        tavily_config: TavilyAdapterConfig,
    ) -> None:
        TavilyAdapter.__adapter_config = tavily_config
        TavilyAdapter.__instance = TavilyAdapter()

    @staticmethod
    def get_tavily_adapter() -> "TavilyAdapter":
        if not TavilyAdapter.__instance:
            TavilyAdapter.__instance = TavilyAdapter()
        return TavilyAdapter.__instance

    @staticmethod
    def get_tavily_config() -> Optional[TavilyAdapterConfig]:
        return TavilyAdapter.__adapter_config
