# -*- coding: utf-8 -*-
"""把 data/images/ 下的图片全部编码入库。

目录约定（非常重要，评估脚本靠它算准确率）：

    data/images/
      item_001/          <- 一个文件夹 = 一个商品
        a.jpg
        b.jpg            <- 同一商品的不同角度 / 不同模特，放在一起
      item_002/
        a.jpg
      ...

如果图片是平铺的（没有子文件夹），也能入库，但每个文件会被当成独立商品，
第 5 关的准确率评估就没法算了。建议还是按文件夹分组。

用法：
    python ingest.py              增量入库（已存在的跳过）
    python ingest.py --reset      清库重来
    python ingest.py --limit 20   只处理前 20 张，先试水
"""
import argparse
import sys
import time
from pathlib import Path

import config
from embedder import build_embedder
from store import get_store


def scan_images(root=None):
    """返回 [(image_id, item_id, 绝对路径)]。"""
    root = Path(root or config.IMAGE_DIR)
    if not root.exists():
        raise SystemExit(f"[错误] 图片目录不存在：{root}\n请先在这里放图片。")

    found = []
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if subdirs:
        for d in subdirs:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in config.IMAGE_EXTS:
                    found.append((f"{d.name}__{f.stem}", d.name, f))
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() in config.IMAGE_EXTS:
            found.append((f"{f.stem}", f.stem, f))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="清空后重新入库")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张")
    parser.add_argument("--backend", default=None, help="clip 或 hist")
    args = parser.parse_args()

    config.banner("SameFit 入库")

    files = scan_images()
    if not files:
        raise SystemExit(
            f"[错误] {config.IMAGE_DIR} 里没有找到图片。\n"
            "支持格式：" + ", ".join(sorted(config.IMAGE_EXTS))
        )
    if args.limit:
        files = files[: args.limit]

    items = {it for _, it, _ in files}
    print(f"图片 {len(files)} 张，商品 {len(items)} 组")
    print(f"特征后端：{args.backend or config.EMBED_BACKEND}   "
          f"存储后端：{config.STORE_BACKEND}")

    store = get_store()

    # 先看库里存的是几维向量。维度和这次用的后端不一致就必须重建，
    # 否则两种向量混在一张表里，检索会算出一堆没意义的数。
    backend = args.backend or config.EMBED_BACKEND
    want_dim = 96 if backend == "hist" else 512
    stored_dim = store.stored_dim()
    if stored_dim is not None and stored_dim != want_dim and not args.reset:
        stored_backend = store.get_meta("embed_backend", "未知")
        raise SystemExit(
            f"\n[错误] 库里已有 {stored_dim} 维向量（当时用的 {stored_backend} 后端），\n"
            f"       这次要写的是 {want_dim} 维（{backend} 后端），两者不能混在一张表里。\n\n"
            f"       重新入库：python ingest.py --backend {backend} --reset\n"
        )

    if args.reset:
        print("已清空旧数据")
        store.reset()

    embedder = build_embedder(args.backend)

    rows = []
    t0 = time.time()
    failed = []
    for i, (image_id, item_id, path) in enumerate(files, 1):
        try:
            vec = embedder.embed_image(path)
        except Exception as exc:
            failed.append((path.name, repr(exc)))
            print(f"  [{i}/{len(files)}] 跳过 {path.name}：{exc}")
            continue
        rows.append({"image_id": image_id, "item_id": item_id,
                     "file": str(path), "vec": vec})
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] 已编码 {len(rows)} 张，"
                  f"耗时 {time.time() - t0:.1f}s")

    if not rows:
        raise SystemExit("[错误] 一张都没成功，检查图片能否正常打开。")

    store.upsert_batch(rows)
    store.set_meta("embed_backend", getattr(embedder, "backend_name", backend))
    store.set_meta("embed_dim", getattr(embedder, "dim", want_dim))
    store.set_meta("embed_name", getattr(embedder, "name", backend))
    cost = time.time() - t0

    print(f"\n入库完成：{len(rows)} 张，总耗时 {cost:.1f}s，"
          f"平均 {cost / len(rows) * 1000:.0f} ms/张")
    print(f"库里现在一共 {store.count()} 条向量")
    if failed:
        print(f"失败 {len(failed)} 张：")
        for name, err in failed[:5]:
            print(f"  - {name}: {err}")

    print("\n下一步：python search.py --image <某张图的路径>")


if __name__ == "__main__":
    sys.exit(main())
