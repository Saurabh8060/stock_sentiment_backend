from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference.predictor import SentimentPredictor
import logging
from datetime import datetime, timezone, timedelta
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SentimentPredictor()
logger = logging.getLogger("api")
_LAST_GOOD_RESPONSE: dict[str, Any] | None = None
_LAST_GOOD_AT: datetime | None = None
DASHBOARD_MAX_RECORDS = 50
GDELT_BACKOFF_SECONDS = int(os.getenv("GDELT_BACKOFF_SECONDS", "300"))
_GDELT_BACKOFF_UNTIL: datetime | None = None

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(req: Request):
    return predictor.predict(req.text)


class DashboardResponse(BaseModel):
    updated_at: str
    kpis: dict
    sentimentDistribution: dict
    trend: list
    articles: list




class EmailRequest(BaseModel):
    keyword: str
    start_date: str  
    end_date: str   
    email: str
    max_records: int | None = 1000


@app.get("/dashboard", response_model=DashboardResponse)
async def dashboard(keyword: str = "AAPL"):
    now = datetime.now(timezone.utc)

    end_dt = now
    start_dt = now - timedelta(days=3)

    try:
        articles = _fetch_articles_with_fallback(
            keyword,
            start_dt,
            end_dt,
            max_records=DASHBOARD_MAX_RECORDS,
        )
    except Exception as exc:
        logger.exception("Failed to fetch any articles: %s", exc)
        return _fallback_or_empty(now)
    if not articles:
        logger.warning("No articles found for keyword=%s", keyword)
        return _fallback_or_empty(now)

    bucket_count = 6
    overall_counts = {"positive": 0, "neutral": 0, "negative": 0}
    trend_buckets, bucket_size = _init_trend_buckets(now, start_dt, bucket_count)
    article_rows = []

    for article in articles:
        text = f"{article['title']} {article.get('summary', '')}".strip()
        prediction = predictor.predict(text)
        sentiment = prediction["sentiment"]
        overall_counts[sentiment] += 1
        _bucket_trend(trend_buckets, article["published_at"], sentiment, start_dt, bucket_size)
        article_rows.append({
            "title": article["title"],
            "url": article["url"],
            "source": article["source"],
            "published_at": article["published_at"].isoformat() if article["published_at"] else None,
            "sentiment": sentiment.capitalize(),
            "confidence": prediction["confidence"],
        })

    total_articles = len(article_rows)
    bullish = overall_counts["positive"]
    bearish = overall_counts["negative"]
    neutral = overall_counts["neutral"]

    response = {
        "updated_at": now.isoformat(),
        "kpis": {
            "totalArticles": total_articles,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
        },
        "sentimentDistribution": overall_counts,
        "trend": trend_buckets,
        "articles": article_rows,
    }
    _store_last_good(response)
    return response


def _fetch_articles_with_fallback(
    keyword: str,
    start_dt: datetime,
    end_dt: datetime,
    max_records: int,
) -> list[dict[str, Any]]:
    global _GDELT_BACKOFF_UNTIL
    articles: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    gdelt_allowed = _GDELT_BACKOFF_UNTIL is None or now >= _GDELT_BACKOFF_UNTIL

    if gdelt_allowed:
        try:
            from data_sources.gdelt_client import fetch_gdelt_articles
            articles = fetch_gdelt_articles(keyword, start_dt, end_dt, max_records=max_records)
            logger.info("GDELT articles fetched: %s", len(articles))
            if articles:
                _GDELT_BACKOFF_UNTIL = None
        except Exception as exc:
            _GDELT_BACKOFF_UNTIL = now + timedelta(seconds=max(30, GDELT_BACKOFF_SECONDS))
            logger.warning("GDELT fetch failed, switching to RSS fallback: %s", exc)
    else:
        logger.info("Skipping GDELT fetch due to backoff until %s", _GDELT_BACKOFF_UNTIL.isoformat())

    if articles:
        return articles

    try:
        from data_sources.rss_client import fetch_google_news_articles
        articles = fetch_google_news_articles(keyword, max_records=max_records)
        logger.info("RSS fallback articles fetched: %s", len(articles))
        return articles
    except Exception as exc:
        logger.warning("RSS fallback failed: %s", exc)
        return []


def _init_trend_buckets(
    now: datetime,
    start_dt: datetime,
    bucket_count: int,
) -> tuple[list[dict[str, Any]], int]:
    total_seconds = max(1, int((now - start_dt).total_seconds()))
    bucket_size = max(1, total_seconds // bucket_count)
    buckets = []
    for i in range(bucket_count):
        bucket_start = start_dt + timedelta(seconds=i * bucket_size)
        buckets.append({
            "time": bucket_start.strftime("%Y-%m-%d"),
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        })
    return buckets, bucket_size


def _bucket_trend(
    buckets: list[dict[str, Any]],
    published_at: datetime | None,
    sentiment: str,
    start_dt: datetime,
    bucket_size: int,
) -> None:
    if sentiment not in ("positive", "neutral", "negative"):
        return
    if not buckets or published_at is None:
        return

    idx = int((published_at - start_dt).total_seconds() // bucket_size)
    idx = max(0, min(len(buckets) - 1, idx))
    buckets[idx][sentiment] += 1


def _empty_dashboard(now: datetime) -> dict[str, Any]:
    return {
        "updated_at": now.isoformat(),
        "kpis": {
            "totalArticles": 0,
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
        },
        "sentimentDistribution": {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        },
        "trend": [],
        "articles": [],
    }


def _store_last_good(payload: dict[str, Any]) -> None:
    global _LAST_GOOD_RESPONSE, _LAST_GOOD_AT
    if payload.get("kpis", {}).get("totalArticles", 0) > 0:
        _LAST_GOOD_RESPONSE = payload
        _LAST_GOOD_AT = datetime.now(timezone.utc)


def _fallback_or_empty(now: datetime) -> dict[str, Any]:
    if _LAST_GOOD_RESPONSE:
        cached = dict(_LAST_GOOD_RESPONSE)
        cached["updated_at"] = now.isoformat()
        cached["kpis"] = dict(cached.get("kpis", {}))
        cached["kpis"]["note"] = "Showing last cached results due to upstream outage."
        return cached
    return _empty_dashboard(now)


@app.post("/email/request")
def request_email_report(req: EmailRequest):
    if not req.keyword.strip():
        raise HTTPException(status_code=400, detail="Keyword is required")
    try:
        start_dt = datetime.fromisoformat(req.start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(req.end_date).replace(tzinfo=timezone.utc)
    except Exception:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD")
    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    max_records = min(max(req.max_records or 1000, 100), 1000)

    queue_dir = os.path.join(PROJECT_ROOT, "backend", "outputs")
    os.makedirs(queue_dir, exist_ok=True)
    queue_path = os.path.join(queue_dir, "email_queue.jsonl")

    record = {
        "keyword": req.keyword.strip(),
        "start_date": req.start_date,
        "end_date": req.end_date,
        "email": req.email.strip(),
        "max_records": max_records,
        "status": "queued",
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }

    import json

    with open(queue_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return {"ok": True, "queued": True, "max_records": max_records}
