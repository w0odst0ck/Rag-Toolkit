"""
rag-toolkit / storage / mysql_manager.py

Simple MySQL connection manager with execute_sql convenience.
Uses ``pymysql`` or ``mysql-connector-python``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import pymysql
from pymysql.cursors import DictCursor

from rag_toolkit.core.config import Config

logger = logging.getLogger(__name__)


class MysqlManager:
    """Lightweight MySQL wrapper for RAG data sources.

    Supports read-only queries (SELECT) and write operations (INSERT/UPDATE/DELETE).
    Reconnects automatically on lost connections.
    """

    def __init__(self, config: Optional[Config] = None, database: Optional[str] = None):
        cfg = config or Config()
        self._host = cfg.MYSQL_HOST
        self._port = cfg.MYSQL_PORT
        self._user = cfg.MYSQL_USER
        self._password = cfg.MYSQL_PASSWORD
        self._database = database or cfg.MYSQL_DATABASE
        self._charset = cfg.MYSQL_CHARSET
        self._conn: Optional[pymysql.Connection] = None

    def _connect(self) -> None:
        if self._conn is not None and self._conn.open:
            try:
                self._conn.ping(reconnect=True)
                return
            except Exception:
                pass
        self._conn = pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset=self._charset,
            cursorclass=DictCursor,
        )
        logger.info(f"Connected to MySQL: {self._host}:{self._port}/{self._database}")

    def close(self) -> None:
        if self._conn and self._conn.open:
            self._conn.close()
            logger.info("MySQL connection closed.")

    def execute_sql(
        self,
        sql: str,
        params: Optional[Union[tuple, dict]] = None,
        fetch: bool = True,
        many: bool = False,
    ) -> Any:
        """Execute a SQL statement.

        Args:
            sql: SQL string with ``%s`` placeholders.
            params: Parameters to bind.
            fetch: If True (default), return fetched rows.
            many: If True, use ``executemany`` (INSERT multiple rows).

        Returns:
            Fetched rows (list of dicts) when *fetch* is True,
            or affected row count when *fetch* is False.
        """
        self._connect()
        cursor = self._conn.cursor()
        try:
            if many:
                cursor.executemany(sql, params or [])
            else:
                cursor.execute(sql, params or ())
            if fetch:
                result = cursor.fetchall()
                return result or []
            else:
                self._conn.commit()
                return cursor.rowcount
        except Exception as e:
            self._conn.rollback()
            logger.error(f"SQL execution error: {e}\nSQL: {sql}")
            raise
        finally:
            cursor.close()

    def select(self, sql: str, params: Optional[Union[tuple, dict]] = None) -> list[dict]:
        """Convenience wrapper for SELECT."""
        return self.execute_sql(sql, params, fetch=True)

    def insert(self, sql: str, params: Optional[Union[tuple, dict]] = None) -> int:
        """Execute INSERT and return lastrowid."""
        self._connect()
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params or ())
            self._conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Insert error: {e}")
            raise
        finally:
            cursor.close()

    def update(self, sql: str, params: Optional[Union[tuple, dict]] = None) -> int:
        """Execute UPDATE/DELETE and return affected rows."""
        return self.execute_sql(sql, params, fetch=False)

    def insert_many(self, sql: str, params_list: list[tuple]) -> int:
        """Execute batch INSERT and return row count."""
        return self.execute_sql(sql, params_list, fetch=False, many=True)
