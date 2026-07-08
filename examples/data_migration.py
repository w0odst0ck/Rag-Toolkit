"""
rag-toolkit / examples / data_migration.py

Template: migrate structured data from MySQL → Milvus.

Replace the column/field names and embedding logic for your domain.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from tqdm import tqdm

from rag_toolkit.pipelines.base_rag import BaseRAGPipeline
from rag_toolkit.storage.milvus_manager import MilvusManager
from rag_toolkit.storage.mysql_manager import MysqlManager

logger = logging.getLogger(__name__)


class DataMigrator:
    """Migrate data from a MySQL source table into a Milvus collection.

    Usage:

        migrator = DataMigrator(
            mysql_db="my_database",
            milvus_collection="my_collection",
        )
        migrator.migrate(
            sql="SELECT id, title, content, group_id FROM my_table WHERE status = 1",
            id_field="id",
            text_field="content",
            title_field="title",
        )
    """

    def __init__(
        self,
        mysql_db: str,
        milvus_collection: str,
        milvus_manager: MilvusManager | None = None,
        mysql_manager: MysqlManager | None = None,
        embedding_fn=None,  # callable(texts) → list[list[float]]
    ):
        self.milvus = milvus_manager or MilvusManager()
        self.mysql = mysql_manager or MysqlManager(database=mysql_db)
        self.collection = milvus_collection
        self._embedding_fn = embedding_fn

    def _ensure_collection(self) -> None:
        if not self.milvus.has_collection(self.collection):
            self.milvus.create_collection(self.collection)
            logger.info(f"Created collection: {self.collection}")

    def migrate(
        self,
        sql: str,
        id_field: str = "id",
        text_field: str = "content",
        title_field: str = "title",
        batch_size: int = 50,
    ) -> int:
        """Run migration.

        Returns total inserted count.
        """
        self._ensure_collection()

        rows = self.mysql.select(sql)
        if not rows:
            logger.warning("No data to migrate.")
            return 0

        logger.info(f"Migrating {len(rows)} rows → collection '{self.collection}' ...")

        inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            texts = [r[text_field] for r in batch]
            embeddings = self._embedding_fn(texts) if self._embedding_fn else [[0.0]] * len(texts)

            data = []
            for j, row in enumerate(batch):
                data.append({
                    "doc_id": str(row[id_field]),
                    "title": row.get(title_field, ""),
                    "text": row[text_field],
                    "sparse_text": row[text_field],
                    "vector": embeddings[j] if embeddings else [],
                })

            self.milvus.insert(self.collection, data)
            inserted += len(data)
            logger.info(f"  Batch {i // batch_size + 1}: {len(data)} rows inserted.")

        logger.info(f"Migration complete: {inserted} rows.")
        return inserted
