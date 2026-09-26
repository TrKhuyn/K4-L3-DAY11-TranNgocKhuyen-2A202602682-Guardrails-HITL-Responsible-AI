"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store request metadata and return the id used to correlate its output."""
        request_id = request_id or uuid.uuid4().hex
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "timestamp": utc_now_iso(),
            "started_at": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete a request record with its decision, output, and latency."""
        key = request_id
        if key is None:
            key = next(
                (rid for rid, item in reversed(list(self._open.items()))
                 if item["user_id"] == user_id),
                None,
            )
        pending = self._open.pop(key, None) if key else None
        pending = pending or {
            "request_id": key or uuid.uuid4().hex,
            "user_id": user_id,
            "input": "",
            "timestamp": utc_now_iso(),
            "started_at": time.perf_counter(),
        }
        started_at = pending.pop("started_at")
        pending.update({
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        })
        self.logs.append(pending)
        return pending

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
