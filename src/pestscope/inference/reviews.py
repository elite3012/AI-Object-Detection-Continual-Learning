from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    predicted_class_id INTEGER,
                    corrected_class_id INTEGER,
                    note TEXT,
                    image_consent INTEGER NOT NULL
                )
                """
            )

    def add(
        self,
        *,
        prediction_id: str,
        decision: str,
        predicted_class_id: int | None,
        corrected_class_id: int | None,
        note: str | None,
        image_consent: bool,
    ) -> dict:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO reviews (
                    created_at, prediction_id, decision, predicted_class_id,
                    corrected_class_id, note, image_consent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    prediction_id,
                    decision,
                    predicted_class_id,
                    corrected_class_id,
                    note,
                    int(image_consent),
                ),
            )
        return {
            "review_id": cursor.lastrowid,
            "created_at": created_at,
            "image_retained": False,
        }

    def summary(self) -> dict:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        return {"review_count": count, "image_retention": "disabled"}
