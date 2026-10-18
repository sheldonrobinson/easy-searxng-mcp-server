"""Shared data types for Konnek SearXNG MCP Server."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SearxCategory(str, Enum):
    """SearXNG search categories."""

    general = "general"
    news = "news"
    images = "images"
    music = "music"
    videos = "videos"
    files = "files"
    weather = "weather"
    social_media = "social media"
    map = "map"
    books = "books"
    web = "web"
    it = "it"
    repos = "repos"
    software_wikis = "software wikis"
    packages = "packages"
    code = "code"
    currency = "currency"
    dictionaries = "dictionaries"
    shopping = "shopping"
    movies = "movies"
    q_and_a = "q&a"
    radio = "radio"
    science = "science"
    scientific_publications = "scientific publications"


class SearxTimeRange(str, Enum):
    """SearXNG time range options."""

    day = "day"
    week = "week"
    month = "month"
    year = "year"


class TavilyResult(BaseModel):
    """A single Tavily-compatible search result."""

    url: str = Field(description="Url used for result", default=None)
    title: str = Field(description="Title of result", default=None)
    content: str = Field(description="Content for result", default=None)
    score: float = Field(description="Score rank for result", default=0.0)
    text: Optional[str] = Field(
        description="Extracted text for result", default=None
    )
    raw_content: Optional[str] = Field(
        description="Raw html for result", default=None
    )


class TavilyResponse(BaseModel):
    """Tavily-compatible search response."""

    query: str = Field(description="Search query", default=None)
    follow_up_questions: Optional[list[str]] = Field(
        description="Follow-up questions", default=None
    )
    answer: Optional[str] = Field(description="Answer to query", default=None)
    images: Optional[list[str]] = Field(
        description="Image result provided to query", default=None
    )
    results: list[TavilyResult] = Field(
        description="List of results for query", default=[]
    )
    response_time: float = Field(
        description="List of results for query", default=None
    )
    request_id: str = Field(description="Request id for query", default=None)
