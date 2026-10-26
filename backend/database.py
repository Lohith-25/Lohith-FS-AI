"""
SolarMap-India — Database Access Layer (Phase 8 & Persistence).

Manages prediction history persistence across MongoDB (when configured)
and local embedded SQLite database fallback (for zero-config local operations).
Never exposes credentials or secrets in logs or responses.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from backend.config import (
    MONGODB_COLLECTION,
    MONGODB_DATABASE,
    MONGODB_SERVER_TIMEOUT_MS,
    MONGODB_URI,
    PROJECT_ROOT,
)

logger = logging.getLogger("solarmap.database")
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "outputs" / "predictions_history.db"


def sanitize_mongodb_uri(uri: str) -> str:
    """Masks username and password in connection strings for safe logging."""
    if not uri:
        return "<empty>"
    # Matches ://<user>:<password>@ and masks user:password
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", uri)


class DatabaseManager:
    """
    Database manager supporting both asynchronous MongoDB (Motor) and local persistent SQLite.
    Provides robust connection handling, schema indexing, and prediction persistence.
    """

    def __init__(
        self,
        uri: str = MONGODB_URI,
        database_name: str = MONGODB_DATABASE,
        collection_name: str = MONGODB_COLLECTION,
        server_timeout_ms: int = MONGODB_SERVER_TIMEOUT_MS,
        sqlite_path: Optional[Path] = None,
        enable_sqlite_fallback: bool = False,
    ):
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.server_timeout_ms = server_timeout_ms
        self.enable_sqlite_fallback = enable_sqlite_fallback
        self.sqlite_path = sqlite_path or DEFAULT_SQLITE_PATH

        self._backend_type: str = "none"  # "mongodb" | "sqlite" | "none"
        self._client: Any = None
        self._db: Any = None
        self._collection: Any = None
        self._is_connected: bool = False
        self._last_error: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        """Returns True if a MongoDB URI is configured or if SQLite fallback is enabled."""
        if bool(self.uri and self.uri.strip()):
            return True
        return self.enable_sqlite_fallback

    @property
    def is_connected(self) -> bool:
        """Returns True if the database is actively connected and ready for queries."""
        return self._is_connected

    def get_status(self) -> Dict[str, Any]:
        """Returns sanitized connection diagnostic information without exposing secrets."""
        db_name = None
        if self._backend_type == "mongodb":
            db_name = self.database_name
        elif self._backend_type == "sqlite":
            db_name = "sqlite_history"

        return {
            "configured": self.is_configured,
            "connected": self._is_connected,
            "backend": self._backend_type,
            "database": db_name,
            "collection": self.collection_name if self._backend_type == "mongodb" else "predictions",
            "error": self._last_error,
        }

    async def connect(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Initializes the database connection.
        Attempts MongoDB if configured; otherwise seamlessly uses local SQLite if fallback is enabled.
        Does not crash the application if databases are unreachable.
        """
        if uri is not None:
            self.uri = uri
        if database_name is not None:
            self.database_name = database_name
        if collection_name is not None:
            self.collection_name = collection_name

        # 1. Try MongoDB if configured
        if self.uri and self.uri.strip():
            try:
                from motor.motor_asyncio import AsyncIOMotorClient

                masked_uri = sanitize_mongodb_uri(self.uri)
                logger.info("Connecting to MongoDB at %s (database: %s)...", masked_uri, self.database_name)

                self._client = AsyncIOMotorClient(
                    self.uri,
                    serverSelectionTimeoutMS=self.server_timeout_ms,
                )
                self._db = self._client[self.database_name]
                self._collection = self._db[self.collection_name]

                # Issue lightweight ping to verify connection health
                await self._client.admin.command("ping")

                self._backend_type = "mongodb"
                self._is_connected = True
                self._last_error = None
                logger.info(
                    "Connected to MongoDB database '%s', collection '%s'.",
                    self.database_name,
                    self.collection_name,
                )

                await self.ensure_indexes()
                return True

            except Exception as exc:
                self._backend_type = "none"
                self._is_connected = False
                self._last_error = str(exc)
                logger.warning(
                    "Could not connect to MongoDB (%s): %s",
                    sanitize_mongodb_uri(self.uri),
                    exc,
                )
                if not self.enable_sqlite_fallback:
                    return False

        # 2. Seamless local SQLite persistent fallback
        if self.enable_sqlite_fallback:
            try:
                self._init_sqlite()
                self._backend_type = "sqlite"
                self._is_connected = True
                self._last_error = None
                logger.info("Connected to local persistent SQLite history at %s", self.sqlite_path)
                return True
            except Exception as exc:
                self._backend_type = "none"
                self._is_connected = False
                self._last_error = f"SQLite init error: {exc}"
                logger.error("Failed to initialize SQLite history fallback: %s", exc)
                return False

        logger.info("Database URI is not configured and local fallback disabled. Prediction history is disabled.")
        self._backend_type = "none"
        self._is_connected = False
        self._last_error = None
        return False

    def _init_sqlite(self) -> None:
        """Initializes the SQLite schema for local prediction history."""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.sqlite_path), timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    filename TEXT,
                    data_json TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_created 
                ON predictions (created_at_utc DESC)
                """
            )
            conn.commit()

    async def disconnect(self) -> None:
        """Closes MongoDB or SQLite database connections cleanly."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.debug("Error while closing MongoDB client: %s", exc)
            finally:
                self._client = None
                self._db = None
                self._collection = None

        self._backend_type = "none"
        self._is_connected = False
        logger.info("Database connection closed.")

    async def ensure_indexes(self) -> None:
        """Creates necessary indexes for query efficiency in MongoDB."""
        if not self._is_connected or self._collection is None:
            return

        try:
            await self._collection.create_index(
                [("prediction_id", 1)],
                unique=True,
                name="idx_prediction_id_unique",
                background=True,
            )
            await self._collection.create_index(
                [("created_at_utc", -1)],
                name="idx_created_at_desc",
                background=True,
            )
            logger.info("MongoDB indexes verified successfully.")
        except Exception as exc:
            logger.warning("Failed to create MongoDB indexes: %s", exc)

    async def save_prediction(self, record: Dict[str, Any]) -> bool:
        """
        Persists a prediction history document into MongoDB or SQLite.
        Returns True on successful insertion, False otherwise.
        """
        if not self._is_connected:
            logger.debug("Skipping prediction save: database is not connected.")
            return False

        if "prediction_id" not in record or not record["prediction_id"]:
            logger.error("Cannot persist prediction record: missing prediction_id.")
            return False

        doc = dict(record)
        if "persisted_at_utc" not in doc:
            doc["persisted_at_utc"] = datetime.now(timezone.utc).isoformat()

        if self._collection is not None:
            try:
                await self._collection.insert_one(doc)
                logger.info("Successfully persisted prediction '%s' to MongoDB.", record["prediction_id"])
                return True
            except Exception as exc:
                logger.error("Failed to persist prediction '%s' to MongoDB: %s", record.get("prediction_id"), exc)
                return False

        if self._backend_type == "sqlite":
            try:
                return await asyncio.to_thread(self._save_sqlite_sync, doc)
            except Exception as exc:
                logger.error("Failed to persist prediction '%s' to SQLite: %s", record.get("prediction_id"), exc)
                return False

        return False

    def _save_sqlite_sync(self, doc: Dict[str, Any]) -> bool:
        pred_id = str(doc["prediction_id"])
        created_at = str(doc.get("created_at_utc") or datetime.now(timezone.utc).isoformat())
        filename = str(doc.get("filename") or "")
        data_json = json.dumps(doc)

        with sqlite3.connect(str(self.sqlite_path), timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions (prediction_id, created_at_utc, filename, data_json)
                VALUES (?, ?, ?, ?)
                """,
                (pred_id, created_at, filename, data_json),
            )
            conn.commit()
        logger.info("Successfully persisted prediction '%s' to SQLite.", pred_id)
        return True

    async def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single prediction history record by its UUID prediction_id.
        Returns None if not found or if database is offline.
        """
        if not self._is_connected:
            return None

        if self._collection is not None:
            try:
                doc = await self._collection.find_one({"prediction_id": prediction_id})
                if doc:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                    return doc
                return None
            except Exception as exc:
                logger.error("Error retrieving prediction '%s' from MongoDB: %s", prediction_id, exc)
                return None

        if self._backend_type == "sqlite":
            try:
                return await asyncio.to_thread(self._get_sqlite_sync, prediction_id)
            except Exception as exc:
                logger.error("Error retrieving prediction '%s' from SQLite: %s", prediction_id, exc)
                return None

        return None

    def _get_sqlite_sync(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        if not self.sqlite_path.is_file():
            return None
        with sqlite3.connect(str(self.sqlite_path), timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM predictions WHERE prediction_id = ?", (prediction_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    async def list_predictions(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieves a paginated list of prediction history records sorted newest first.
        Returns a tuple of (items_list, total_count).
        """
        if not self._is_connected:
            return [], 0

        safe_skip = max(0, int(skip))
        safe_limit = max(1, min(100, int(limit)))

        if self._collection is not None:
            try:
                total_count = await self._collection.count_documents({})
                cursor = (
                    self._collection.find({})
                    .sort("created_at_utc", -1)
                    .skip(safe_skip)
                    .limit(safe_limit)
                )

                items: List[Dict[str, Any]] = []
                async for doc in cursor:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                    items.append(doc)

                return items, total_count
            except Exception as exc:
                logger.error("Error listing prediction history from MongoDB: %s", exc)
                return [], 0

        if self._backend_type == "sqlite":
            try:
                return await asyncio.to_thread(self._list_sqlite_sync, safe_skip, safe_limit)
            except Exception as exc:
                logger.error("Error listing prediction history from SQLite: %s", exc)
                return [], 0

        return [], 0

    def _list_sqlite_sync(self, skip: int, limit: int) -> Tuple[List[Dict[str, Any]], int]:
        if not self.sqlite_path.is_file():
            return [], 0
        with sqlite3.connect(str(self.sqlite_path), timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total = cursor.fetchone()[0]
            cursor.execute(
                "SELECT data_json FROM predictions ORDER BY created_at_utc DESC LIMIT ? OFFSET ?",
                (limit, skip),
            )
            rows = cursor.fetchall()
            items = [json.loads(r[0]) for r in rows]
            return items, total

    async def delete_prediction(self, prediction_id: str) -> bool:
        """Deletes a prediction record by its prediction_id. Returns True if deleted."""
        if not self._is_connected:
            return False

        if self._collection is not None:
            try:
                res = await self._collection.delete_one({"prediction_id": prediction_id})
                return res.deleted_count > 0
            except Exception as exc:
                logger.error("Error deleting prediction '%s' from MongoDB: %s", prediction_id, exc)
                return False

        if self._backend_type == "sqlite":
            try:
                return await asyncio.to_thread(self._delete_sqlite_sync, prediction_id)
            except Exception as exc:
                logger.error("Error deleting prediction '%s' from SQLite: %s", prediction_id, exc)
                return False

        return False

    def _delete_sqlite_sync(self, prediction_id: str) -> bool:
        if not self.sqlite_path.is_file():
            return False
        with sqlite3.connect(str(self.sqlite_path), timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE prediction_id = ?", (prediction_id,))
            conn.commit()
            return cursor.rowcount > 0


# Global singleton database manager instance:
# When MONGODB_URI is provided, connects to MongoDB.
# Otherwise, seamlessly enables local persistent SQLite storage so history is fully functional.
db_manager = DatabaseManager(enable_sqlite_fallback=True)
