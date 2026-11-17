# -*- coding: utf-8 -*-
"""向量仓库。两套实现，接口一样，改 config.STORE_BACKEND 就能切换。

SqliteStore  向量存成 BLOB，检索时全量读进内存做矩阵乘法。
             十万张以内这个最省事，而且完全不用装数据库。
PgStore      向量存在 PostgreSQL 的 vector 列里，检索交给 pgvector 的 HNSW 索引。
             简历上要写的就是这一套。
"""
from pathlib import Path

import numpy as np

import config

IMAGE_PLACEHOLDER = "<image>"


def get_store(backend=None):
    backend = backend or config.STORE_BACKEND
    if backend == "pg":
        return PgStore(config.PG_DSN)
    return SqliteStore(config.DB_PATH)


class SqliteStore:
    backend = "sqlite"

    def __init__(self, path=None):
        import sqlite3

        path = Path(path or config.DB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                image_id TEXT PRIMARY KEY,
                item_id  TEXT NOT NULL,
                file     TEXT NOT NULL,
                dim      INTEGER NOT NULL,
                vec      BLOB NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_item ON items(item_id)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value))
        )
        self.conn.commit()

    def get_meta(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else default

    def stored_dim(self):
        row = self.conn.execute("SELECT dim FROM items LIMIT 1").fetchone()
        return int(row[0]) if row else None

    def item_of_file(self, path):
        """这张图在库里属于哪个商品？不在库里就返回 None。"""
        target = str(Path(path).resolve()).lower()
        for item_id, f in self.conn.execute("SELECT item_id, file FROM items"):
            if str(Path(f).resolve()).lower() == target:
                return item_id
        return None

    def upsert_batch(self, rows):
        payload = [
            (r["image_id"], r["item_id"], r["file"], int(len(r["vec"])),
             np.asarray(r["vec"], dtype="float32").tobytes())
            for r in rows
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO items(image_id,item_id,file,dim,vec) "
            "VALUES(?,?,?,?,?)",
            payload,
        )
        self.conn.commit()

    def load_all(self):
        rows = self.conn.execute(
            "SELECT image_id, item_id, file, vec FROM items"
        ).fetchall()
        if not rows:
            return [], [], [], np.zeros((0, 0), dtype="float32")
        ids = [r[0] for r in rows]
        items = [r[1] for r in rows]
        files = [r[2] for r in rows]
        mat = np.vstack([np.frombuffer(r[3], dtype="float32") for r in rows])
        return ids, items, files, mat

    def search(self, vec, k=5):
        ids, items, files, mat = self.load_all()
        if not ids:
            return []
        q = np.asarray(vec, dtype="float32")
        scores = mat @ q  # 向量已归一化，点积即余弦相似度
        order = np.argsort(-scores)[:k]
        return [
            {"image_id": ids[i], "item_id": items[i], "file": files[i],
             "score": float(scores[i])}
            for i in order
        ]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def reset(self):
        self.conn.execute("DELETE FROM items")
        self.conn.execute("DELETE FROM meta")
        self.conn.commit()


class PgStore:
    backend = "pg"

    def __init__(self, dsn=None):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError as exc:
            raise SystemExit(
                "\n[错误] 缺少 psycopg2 或 pgvector 的 Python 包：\n"
                "  pip install psycopg2-binary pgvector "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                f"原始报错：{exc}"
            )
        self.dsn = dsn or config.PG_DSN
        try:
            self.conn = psycopg2.connect(self.dsn)
        except Exception as exc:
            raise SystemExit(
                "\n[错误] 连不上 PostgreSQL。检查 config.py 里的 PG_DSN，"
                "格式形如：\n"
                "  postgresql://用户名:密码@主机:5432/库名\n"
                f"原始报错：{exc}"
            )
        self.conn.autocommit = True
        self.dim = None
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'items' AND a.attname = 'vec'
                """
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    CREATE TABLE items (
                        image_id TEXT PRIMARY KEY,
                        item_id  TEXT NOT NULL,
                        file     TEXT NOT NULL,
                        vec      vector(512) NOT NULL
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_items_vec "
                    "ON items USING hnsw (vec vector_cosine_ops)"
                )
            else:
                self.dim = row[0]

    def set_meta(self, key, value):
        with self.conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            cur.execute(
                "INSERT INTO meta(key,value) VALUES(%s,%s) "
                "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                (key, str(value)),
            )

    def get_meta(self, key, default=None):
        with self.conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            cur.execute("SELECT value FROM meta WHERE key=%s", (key,))
            row = cur.fetchone()
            return row[0] if row else default

    def stored_dim(self):
        return 512

    def item_of_file(self, path):
        target = str(Path(path).resolve()).lower()
        with self.conn.cursor() as cur:
            cur.execute("SELECT item_id, file FROM items")
            for item_id, f in cur.fetchall():
                if str(Path(f).resolve()).lower() == target:
                    return item_id
        return None

    def upsert_batch(self, rows):
        if not rows:
            return
        import numpy as np

        with self.conn.cursor() as cur:
            for r in rows:
                vec = np.asarray(r["vec"], dtype="float32")
                cur.execute(
                    "INSERT INTO items(image_id,item_id,file,vec) VALUES(%s,%s,%s,%s) "
                    "ON CONFLICT (image_id) DO UPDATE SET "
                    "item_id=EXCLUDED.item_id, file=EXCLUDED.file, vec=EXCLUDED.vec",
                    (r["image_id"], r["item_id"], r["file"], vec),
                )

    def search(self, vec, k=5):
        import numpy as np

        q = np.asarray(vec, dtype="float32")
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT image_id, item_id, file, 1 - (vec <=> %s) AS score "
                "FROM items ORDER BY vec <=> %s LIMIT %s",
                (q, q, k),
            )
            return [
                {"image_id": r[0], "item_id": r[1], "file": r[2], "score": float(r[3])}
                for r in cur.fetchall()
            ]

    def count(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM items")
            return cur.fetchone()[0]

    def reset(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM items")
            cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            cur.execute("DELETE FROM meta")
