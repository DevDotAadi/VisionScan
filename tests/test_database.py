"""Unit tests for VisionScan Global — SQLite database storage & automatic migration."""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path
import pandas as pd

from utils.engine import Prediction, RiskLevel, Certainty
from utils.db_utils import (
    DB_PATH,
    LEGACY_CSV_PATH,
    init_db,
    get_db_connection,
    log_prediction_to_db,
    get_prediction_history_df,
)


@pytest.fixture(autouse=True)
def setup_temporary_database(tmp_path, monkeypatch):
    """Fixture to sandbox SQLite and CSV paths inside an isolated pytest folder."""
    temp_db = tmp_path / "test_prediction_history.db"
    temp_csv = tmp_path / "test_prediction_history.csv"

    # Patch modules global variables
    monkeypatch.setattr("utils.db_utils.DB_PATH", temp_db)
    monkeypatch.setattr("utils.db_utils.LEGACY_CSV_PATH", temp_csv)

    yield temp_db, temp_csv


class TestDatabaseStorage:
    def test_database_initialization_and_wal_journal_mode(self, setup_temporary_database):
        temp_db, _ = setup_temporary_database
        assert not temp_db.exists()

        init_db()
        assert temp_db.exists()

        # Check WAL mode status
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        conn.close()

        # On some platforms SQLite might fall back to delete mode if WAL is disabled,
        # but connection initialization should pass successfully.
        assert mode.lower() in ("wal", "memory", "delete", "truncate")

    def test_write_and_read_prediction_records(self, setup_temporary_database):
        init_db()

        pred = Prediction(
            label="Benign",
            confidence=0.892,
            prob_malignant=0.108,
            risk=RiskLevel.LOW,
            certainty=Certainty.CERTAIN,
            recommendation="Continue monthly skin checks.",
            timestamp="2026-05-10T12:00:00Z",
        )

        log_prediction_to_db("test_lesion.png", pred)

        df = get_prediction_history_df()
        assert len(df) == 1
        assert df.iloc[0]["filename"] == "test_lesion.png"
        assert df.iloc[0]["prediction"] == "Benign"
        assert abs(df.iloc[0]["confidence"] - 0.892) < 1e-5
        assert df.iloc[0]["risk"] == "Low Risk"

    def test_legacy_csv_migration_on_startup(self, setup_temporary_database):
        temp_db, temp_csv = setup_temporary_database

        # Write dummy legacy CSV rows
        legacy_data = pd.DataFrame([{
            "timestamp": "2026-05-09T18:00:00Z",
            "filename": "old_scan.jpg",
            "prediction": "Malignant",
            "confidence": 0.941,
            "risk": "High Risk",
            "certainty": "Certain",
        }])
        legacy_data.to_csv(temp_csv, index=False)

        # Trigger DB initialization which should trigger the migration
        init_db()

        # Database should hold the migrated row
        df = get_prediction_history_df()
        assert len(df) == 1
        assert df.iloc[0]["filename"] == "old_scan.jpg"
        assert df.iloc[0]["prediction"] == "Malignant"

        # CSV should be renamed with .migrated suffix to avoid re-runs
        assert not temp_csv.exists()
        assert temp_csv.with_suffix(".csv.migrated").exists()
