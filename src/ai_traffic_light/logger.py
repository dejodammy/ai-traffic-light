from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Logs every DQN phase decision to a SQLite database.

    Designed as an on_decision callback for run_live(). Each row records
    the timestamp, chosen action, per-approach vehicle counts and densities,
    and the raw 12-feature state vector.

    Usage:
        db = DecisionLogger("results/live_session.db")
        run_live(controller, on_decision=db)
        db.close()
    """

    def __init__(self, db_path: str = "results/live_session.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._create_table()
        logger.info("Decision logger opened: %s", self._path)

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                action      INTEGER NOT NULL,
                observations TEXT   NOT NULL,
                state        TEXT   NOT NULL
            )
        """)
        self._conn.commit()

    def __call__(self, decision) -> None:
        obs = {
            o.approach: {"count": float(o.vehicle_count), "density": float(o.density)}
            for o in decision.observations
        }
        self._conn.execute(
            "INSERT INTO decisions (timestamp, action, observations, state) VALUES (?,?,?,?)",
            (
                decision.timestamp,
                int(decision.action),
                json.dumps(obs),
                json.dumps(decision.state.tolist()),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
        logger.info("Decision logger closed: %s", self._path)
