# -*- coding: utf-8 -*-
"""第 8 关的迁移工具：把 SQLite 里已经算好的向量搬到 PostgreSQL。

为什么单独写一个迁移脚本，而不是重新 ingest？
    因为 ingest 要重新跑一遍 CLIP 编码（83 张大概 11 秒，几千张就要几十分钟），
    而向量的数值早就定下来了，搬家只需要读写，不需要再算。
    这个思路在真实项目里也成立：**模型升级才重算，换存储只搬迁**。

用法：
    set SAMEFIT_PG_DSN=postgresql://用户:密码@主机:5432/库名
    python migrate_to_pg.py --dry-run     # 先看看要搬多少
    python migrate_to_pg.py               # 真搬

搬完验证：
    set SAMEFIT_STORE=pg
    python evaluate.py --k 5
    python search.py --image data\\images\\dress_005\\01.jpg
"""
import argparse
import sys

import config
from store import PgStore, SqliteStore

META_KEYS = ("embed_backend", "embed_dim", "embed_name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=None, help="PG 连接串，不给就读 config.PG_DSN")
    ap.add_argument("--dry-run", action="store_true", help="只检查不写入")
    args = ap.parse_args()

    config.banner("迁移 SQLite -> PostgreSQL")

    src = SqliteStore(config.DB_PATH)
    ids, items, files, mat = src.load_all()
    if not ids:
        raise SystemExit(
            f"[错误] {config.DB_PATH} 里没有向量。\n"
            "先跑 python ingest.py 把图片编码入库，再来迁移。"
        )

    dim = mat.shape[1]
    meta = {k: src.get_meta(k) for k in META_KEYS}
    backend = meta.get("embed_backend") or "未知"

    print(f"源库：{len(ids)} 条向量，{dim} 维，{len(set(items))} 个商品，后端 {backend}")

    if backend != "clip" or dim != 512:
        raise SystemExit(
            f"\n[错误] PG 表结构是按 CLIP 的 512 维建的，而你要搬的是 "
            f"{dim} 维（{backend}）。\n\n"
            "       先切回真模型重建一遍 SQLite：\n"
            "         python ingest.py --reset\n"
            "       成功之后再跑本脚本。\n"
        )

    if args.dry_run:
        print("\n[dry-run] 检查通过，没有写入任何东西。去掉 --dry-run 就会真搬。")
        return 0

    print("\n连接 PostgreSQL ...")
    pg = PgStore(args.dsn or config.PG_DSN)

    print("清空目标表 ...")
    pg.reset()

    print("写入向量 ...")
    rows = [
        {"image_id": ids[i], "item_id": items[i], "file": files[i], "vec": mat[i]}
        for i in range(len(ids))
    ]
    pg.upsert_batch(rows)

    for k, v in meta.items():
        if v:
            pg.set_meta(k, v)

    n = pg.count()
    print(f"目标库：{n} 条")

    if n != len(ids):
        raise SystemExit(
            f"[错误] 条数对不上：源 {len(ids)}，目标 {n}。"
            "检查一下是不是被主键冲突挡掉了一部分。"
        )

    print("\n迁移完成。下面两条命令验证：")
    print("  set SAMEFIT_STORE=pg")
    print("  python evaluate.py --k 5")
    print("\n（同一批向量，评估数字应该和 SQLite 版完全一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
