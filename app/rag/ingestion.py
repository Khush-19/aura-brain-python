# Ingestion engine — scrapes Sydney lifestyle and NSW gov sources into LangChain Documents.
#
# JS-heavy pages: swap Scraper._fetch() for a Playwright variant (see commented stub at bottom).

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import SCRAPER_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Sydney sources
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    url: str
    name: str
    source_type: str  # "lifestyle" | "gov" | "custom"
    # CSS selectors tried in order; first match wins
    content_selectors: List[str] = field(
        default_factory=lambda: ["article", "main", '[role="main"]', ".content", "#content", "body"]
    )


SYDNEY_SOURCES: List[SourceConfig] = [
    SourceConfig(
        url="https://www.broadsheet.com.au/sydney",
        name="Broadsheet Sydney",
        source_type="lifestyle",
    ),
    SourceConfig(
        url="https://www.timeout.com/sydney",
        name="TimeOut Sydney",
        source_type="lifestyle",
    ),
    SourceConfig(
        url="https://www.cityofsydney.nsw.gov.au/whats-on",
        name="City of Sydney — What's On",
        source_type="gov",
    ),
    SourceConfig(
        url="https://www.health.nsw.gov.au/news/Pages/default.aspx",
        name="NSW Health News",
        source_type="gov",
    ),
]

# ---------------------------------------------------------------------------
# HTML → clean text cleaner
# ---------------------------------------------------------------------------

class HTMLMarkdownCleaner:
    """
    Strips boilerplate (nav, footer, scripts) and extracts meaningful text blocks.
    Preserves heading hierarchy with simple prefixes so the LLM retains structure.
    """

    _STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "form"}
    _BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "td", "th"}

    def clean(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(_STRIP_TAGS := list(self._STRIP_TAGS)):
            tag.decompose()

        blocks: List[str] = []
        for tag in soup.find_all(self._BLOCK_TAGS):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) < 25:
                continue
            if tag.name in ("h1", "h2"):
                text = f"## {text}"
            elif tag.name in ("h3", "h4"):
                text = f"### {text}"
            blocks.append(text)

        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class Scraper:
    """Fetches URLs and returns one LangChain Document per source."""

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-AU,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, timeout: int = SCRAPER_TIMEOUT) -> None:
        self.timeout = timeout
        self.cleaner = HTMLMarkdownCleaner()
        self._session = requests.Session()
        self._session.headers.update(self._HEADERS)

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    def _fetch(self, url: str) -> str:
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def scrape(self, source: SourceConfig) -> Optional[Document]:
        try:
            html = self._fetch(source.url)
        except Exception as exc:
            logger.error("Fetch failed [%s]: %s", source.url, exc)
            return None

        text = self.cleaner.clean(html)
        if not text.strip():
            logger.warning("Empty body after cleaning [%s]", source.url)
            return None

        return Document(
            page_content=text,
            metadata={
                "source": source.url,
                "name": source.name,
                "type": source.source_type,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    def scrape_all(self, sources: Optional[List[SourceConfig]] = None) -> List[Document]:
        targets = sources if sources is not None else SYDNEY_SOURCES
        docs = [doc for src in targets if (doc := self.scrape(src)) is not None]
        logger.info("Scraped %d / %d sources successfully", len(docs), len(targets))
        return docs


# ---------------------------------------------------------------------------
# Playwright stub (uncomment + pip install playwright + playwright install chromium)
# ---------------------------------------------------------------------------
#
# async def _fetch_playwright(url: str) -> str:
#     from playwright.async_api import async_playwright
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         page = await browser.new_page()
#         await page.goto(url, wait_until="networkidle", timeout=15_000)
#         html = await page.content()
#         await browser.close()
#         return html
