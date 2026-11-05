"""
Interview History Store — Feature 3
Persists practice sessions (questions, answers, scores, dates) to a local
JSON file so users can track their progress across browser sessions.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_FILE = os.getenv("HISTORY_FILE", "./data/interview_history.json")


def _load_all() -> list[dict]:
    """Load the full history list from disk."""
    path = Path(HISTORY_FILE)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load history: {e}")
        return []


def _save_all(records: list[dict]) -> None:
    """Persist the history list to disk."""
    path = Path(HISTORY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Could not save history: {e}")


def save_session(record: dict) -> str:
    """
    Append a session record to history.

    Expected record shape:
    {
        "candidate_name": str,
        "job_title": str,
        "experience_level": str,
        "questions": [{"question":…, "category":…, "difficulty":…, …}, …],
        "evaluations": [{"question":…, "answer":…, "score":…, …}, …],  # may be []
        "timestamp": ISO str (auto-set if missing),
        "session_id": str (auto-generated if missing),
    }

    Returns the session_id.
    """
    import uuid
    records = _load_all()

    if not record.get("session_id"):
        record["session_id"] = str(uuid.uuid4())[:8]
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

    records.append(record)
    # Keep last 50 sessions to avoid unbounded growth
    records = records[-50:]
    _save_all(records)
    return record["session_id"]


def get_all_sessions() -> list[dict]:
    """Return all stored sessions, newest first."""
    records = _load_all()
    return list(reversed(records))


def get_session(session_id: str) -> dict | None:
    """Return a single session by its ID."""
    for r in _load_all():
        if r.get("session_id") == session_id:
            return r
    return None


def delete_session(session_id: str) -> bool:
    """Delete a session by ID. Returns True if found and deleted."""
    records = _load_all()
    original_len = len(records)
    records = [r for r in records if r.get("session_id") != session_id]
    if len(records) < original_len:
        _save_all(records)
        return True
    return False


def get_score_trend(candidate_name: str = None) -> list[dict]:
    """
    Return [{date, avg_score, session_id, job_title}, …] for charting.
    Optionally filtered by candidate name.
    """
    records = _load_all()
    trend = []
    for r in records:
        if candidate_name and r.get("candidate_name", "").lower() != candidate_name.lower():
            continue
        evals = r.get("evaluations", [])
        if not evals:
            continue
        scores = [e.get("score", 0) for e in evals if isinstance(e.get("score"), (int, float))]
        if not scores:
            continue
        trend.append({
            "session_id": r.get("session_id"),
            "date": r.get("timestamp", "")[:10],
            "avg_score": round(sum(scores) / len(scores), 1),
            "max_score": max(scores),
            "min_score": min(scores),
            "num_questions": len(evals),
            "job_title": r.get("job_title", ""),
        })
    return trend
