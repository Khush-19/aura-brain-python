"""Background refresh jobs for live Sydney alerts and events."""

import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import ENABLE_SCHEDULER, REFRESH_INTERVAL_MINUTES
from app.rag.ingestion import Scraper, SourceConfig
from app.rag.retriever import ingest_documents

logger = logging.getLogger(__name__)

REALTIME_SOURCES = [
    SourceConfig(
        url="https://www.nationalparks.nsw.gov.au/alerts/alerts-list/",
        name="NSW National Parks Alerts",
        source_type="alerts",
        use_playwright=True,
    ),
    SourceConfig(
        url="https://whatson.cityofsydney.nsw.gov.au/",
        name="City of Sydney What's On",
        source_type="events",
        use_playwright=True,
    ),
]


def refresh_realtime_sources() -> int:
    """
    Poll real-time-ish Sydney sources and ingest only new content.

    Deduplication happens in the vector-store layer via content_hash metadata, so
    repeated scheduler runs do not keep appending duplicate FAISS vectors.
    """
    scraper = Scraper()
    docs = scraper.scrape_all(REALTIME_SOURCES)

    if not docs:
        logger.warning("Realtime refresh found no valid documents")
        return 0

    chunks = ingest_documents(docs)
    logger.info("Realtime refresh ingested %d new chunks from %d documents", chunks, len(docs))
    return chunks


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Australia/Sydney")
    scheduler.add_job(
        refresh_realtime_sources,
        trigger=IntervalTrigger(minutes=REFRESH_INTERVAL_MINUTES),
        id="realtime_source_refresh",
        name="Refresh NSW parks alerts and City of Sydney events",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


_scheduler: Optional[BackgroundScheduler] = None


def start_scheduler() -> BackgroundScheduler:
    global _scheduler

    if not ENABLE_SCHEDULER:
        logger.info("Realtime refresh scheduler is disabled")
        _scheduler = create_scheduler()
        return _scheduler

    if _scheduler and _scheduler.running:
        return _scheduler

    _scheduler = create_scheduler()
    _scheduler.start()
    logger.info("Started realtime refresh scheduler (%d minute interval)", REFRESH_INTERVAL_MINUTES)
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Stopped realtime refresh scheduler")
    _scheduler = None
