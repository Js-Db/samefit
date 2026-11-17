-- SameFit 第 6 关：把向量从 SQLite 搬到 PostgreSQL
-- 在云库（Supabase / Neon）的 SQL Editor 里，或者本地 psql 里整段执行。
-- 注意：这里维度写死 512，对应 CLIP ViT-B/32。用 hist 后端（96 维）别用这张表。

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS items;

CREATE TABLE items (
    image_id TEXT PRIMARY KEY,
    item_id  TEXT NOT NULL,
    file     TEXT NOT NULL,
    vec      vector(512) NOT NULL
);

CREATE INDEX idx_items_item ON items (item_id);

-- HNSW 索引：pgvector 0.5+ 才有。数据量小于几千条时建不建都很快，
-- 但简历上"建过 HNSW 索引"这句话，得真建了才敢写。
CREATE INDEX idx_items_vec
    ON items USING hnsw (vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 验证：
--   SELECT count(*) FROM items;
--   \d items
