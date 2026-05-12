"""Database utilities for VisionScan Global.

Handles secure, concurrent SQLite-based storage of prediction logs
with Write-Ahead Logging (WAL) enabled for high concurrent production safety.
Allows migration of legacy prediction history from CSV automatically.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
import pandas as pd
from utils.engine import Prediction

log = logging.getLogger(__name__)

DB_PATH = Path("results/prediction_history.db")
LEGACY_CSV_PATH = Path("results/prediction_history.csv")


def get_db_connection() -> sqlite3.Connection:
    """Create and return a SQLite connection with WAL mode enabled and timeout adjustments."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    # Enable WAL mode for safe multi-user concurrent writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    # Ensure rows are dict-accessible
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite prediction log schema and migrate existing CSV records if any."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk TEXT NOT NULL,
                    certainty TEXT NOT NULL
                );
            """)
        log.info("SQLite database schema verified successfully.")
    except Exception as e:
        log.error("Failed to initialise database table schema: %s", e)
        conn.close()
        return

    # Check for automatic legacy CSV migration
    if LEGACY_CSV_PATH.exists():
        try:
            df = pd.read_csv(LEGACY_CSV_PATH)
            if not df.empty:
                log.info("Legacy prediction_history.csv detected. Commencing migration...")
                with conn:
                    for _, row in df.iterrows():
                        # Prevent duplicate inserts during migrations
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT 1 FROM prediction_history WHERE timestamp = ? AND filename = ?",
                            (str(row["timestamp"]), str(row["filename"]))
                        )
                        if cursor.fetchone() is None:
                            conn.execute(
                                """
                                INSERT INTO prediction_history (timestamp, filename, prediction, confidence, risk, certainty)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    str(row["timestamp"]),
                                    str(row["filename"]),
                                    str(row["prediction"]),
                                    float(row["confidence"]),
                                    str(row["risk"]),
                                    str(row["certainty"]),
                                )
                            )
                log.info("Migration of %d history entries completed successfully.", len(df))
            # Rename CSV to avoid duplicate scanning loops
            LEGACY_CSV_PATH.rename(LEGACY_CSV_PATH.with_suffix(".csv.migrated"))
        except Exception as e:
            log.warning("Could not complete legacy CSV history migration: %s", e)

    conn.close()


def log_prediction_to_db(filename: str, pred: Prediction) -> None:
    """Securely write prediction indicators to sqlite database."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO prediction_history (timestamp, filename, prediction, confidence, risk, certainty)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pred.timestamp,
                    filename,
                    pred.label,
                    pred.confidence,
                    pred.risk.value,
                    pred.certainty.value,
                )
            )
        log.info("Successfully saved prediction for %s to sqlite storage.", filename)
    except Exception as e:
        log.error("SQLite write failed for %s: %s", filename, e)
    finally:
        conn.close()


def get_prediction_history_df() -> pd.DataFrame:
    """Load and return all log histories as a pandas dataframe."""
    init_db()  # Verify table is initialized
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT timestamp, filename, prediction, confidence, risk, certainty FROM prediction_history", conn)
        return df
    except Exception as e:
        log.error("Could not retrieve prediction history from database: %s", e)
        return pd.DataFrame()
    finally:
        conn.close()
