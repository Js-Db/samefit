# -*- coding: utf-8 -*-
"""以图搜图 / 以文搜图，并生成一张可以直接发出去的结果图。

用法：
    python search.py --image data/images/item_001/a.jpg
    python search.py --image a.jpg --k 8 --card
    python search.py --text "red floral summer dress"     # 需要 clip 后端
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import config
from embedder import build_embedder
from store import get_store

CARD_BG = (255, 255, 255)
CARD_LINE = (210, 210, 210)
CARD_TEXT = (44, 44, 42)
CARD_MUTED = (110, 108, 102)
CARD_HIT = (15, 110, 86)
CARD_MISS = (163, 45, 45)


def _font(size, bold=False):
    candidates = (
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    )
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _thumb(path, size):
    img = Image.open(path).convert("RGB")
    return ImageOps.fit(img, (size, size), Image.LANCZOS)


def make_result_card(query_path, results, out_path, k=5, model_label="",
                     same_item=None):
    """左边是查询图，右边排一行 Top-K 结果，下面标分数。

    same_item 传查询图的商品编号时，同款结果用绿色标、异款用灰色，
    一眼就能看出系统是"认对了"还是"认错了"。
    """
    pad, qsize, rsize, gap = 24, 320, 130, 12
    head = 58
    width = pad + qsize + 40 + k * (rsize + gap) + pad - gap
    height = head + max(qsize + 22, rsize + 62) + pad

    canvas = Image.new("RGB", (width, height), CARD_BG)
    d = ImageDraw.Draw(canvas)
    f_title = _font(22, bold=True)
    f_small = _font(15)
    f_tiny = _font(13)
    f_score = _font(17, bold=True)

    d.text((pad, 18), "SameFit 同款检索", font=f_title, fill=CARD_TEXT)
    if model_label:
        d.text((pad + 210, 25), model_label, font=f_tiny, fill=CARD_MUTED)
    if same_item:
        d.text((width - pad - 108, 25), "绿=同款  灰=异款",
               font=f_tiny, fill=CARD_MUTED)

    if query_path:
        q = _thumb(query_path, qsize)
        canvas.paste(q, (pad, head))
        d.rectangle([pad, head, pad + qsize - 1, head + qsize - 1],
                    outline=CARD_LINE, width=1)
        d.text((pad, head + qsize + 4), "query image", font=f_small, fill=CARD_MUTED)
    else:
        d.rectangle([pad, head, pad + qsize, head + qsize], outline=CARD_LINE)
        d.text((pad + 12, head + qsize // 2), "text query", font=f_small, fill=CARD_MUTED)

    x0 = pad + qsize + 40
    d.line([(x0 - 20, head), (x0 - 20, head + qsize)], fill=CARD_LINE, width=1)

    # 结果块在查询图高度内垂直居中，免得下边空一大块
    ry = head + max(0, (qsize - (rsize + 42)) // 2)

    for i, r in enumerate(results[:k]):
        x = x0 + i * (rsize + gap)
        try:
            canvas.paste(_thumb(r["file"], rsize), (x, ry))
            canvas.rectangle([x, ry, x + rsize - 1, ry + rsize - 1],
                             outline=CARD_LINE, width=1)
        except Exception:
            d.rectangle([x, ry, x + rsize, ry + rsize], outline=CARD_LINE)
        color = CARD_HIT if (same_item and r["item_id"] == same_item) \
            else (CARD_MUTED if same_item else (CARD_HIT if i == 0 else CARD_MUTED))
        d.text((x, ry + rsize + 6), f"{r['score']:.3f}", font=f_score, fill=color)
        label = r["item_id"]
        if len(label) > 16:
            label = label[:15] + "."
        d.text((x, ry + rsize + 28), label, font=f_tiny, fill=CARD_MUTED)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="用这张图去搜")
    ap.add_argument("--text", help="用一句话去搜（需 clip 后端）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--backend", default=None, help="clip 或 hist")
    ap.add_argument("--card", action="store_true", help="同时生成结果图")
    ap.add_argument("--keep-self", action="store_true",
                    help="结果里保留查询图本身（默认排除，因为返回自己没意义）")
    ap.add_argument("--exclude-self", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.image and not args.text:
        raise SystemExit("要么给 --image，要么给 --text。例子：\n"
                         "  python search.py --image data/images/item_001/a.jpg")

    store = get_store()
    total = store.count()
    if total == 0:
        raise SystemExit("[错误] 库里没有向量，先跑 python ingest.py")
    print(f"库内向量 {total} 条")

    # 优先跟库里入库时用的后端保持一致，免得她忘了加 --backend 就报维度错
    stored_backend = store.get_meta("embed_backend")
    backend = args.backend or stored_backend or config.EMBED_BACKEND
    want_dim = 96 if backend == "hist" else 512
    stored_dim = store.stored_dim()
    if stored_dim is not None and stored_dim != want_dim:
        raise SystemExit(
            f"\n[错误] 库里的向量是 {stored_dim} 维"
            f"（入库时用的 {stored_backend or '未知'}），\n"
            f"       而 {backend} 后端是 {want_dim} 维，两者对不上。\n\n"
            f"       办法一，跟入库保持一致：加 --backend {stored_backend}\n"
            f"       办法二，重新入库：python ingest.py --backend {backend} --reset\n"
        )

    embedder = build_embedder(backend)

    t0 = time.time()
    if args.image:
        qpath = Path(args.image)
        if not qpath.exists():
            raise SystemExit(f"[错误] 找不到图片：{qpath}")
        vec = embedder.embed_image(qpath)
    else:
        try:
            vec = embedder.embed_text(args.text)
        except NotImplementedError as exc:
            raise SystemExit(f"[错误] {exc}")
        qpath = None
    t_embed = time.time() - t0

    t1 = time.time()
    # 默认多取一个，因为要把查询图自己剔掉（除非 --keep-self）
    results = store.search(vec, k=args.k + 1)
    if not args.keep_self and qpath is not None:
        q_abs = str(Path(qpath).resolve()).lower()
        results = [r for r in results
                   if str(Path(r["file"]).resolve()).lower() != q_abs]
    results = results[: args.k]
    t_search = time.time() - t1

    print(f"\n编码 {t_embed * 1000:.0f} ms，检索 {t_search * 1000:.1f} ms\n")
    if not results:
        print("没有结果。")
        return

    print(f"{'排名':<4}{'相似度':<10}{'商品':<14}{'文件'}")
    print("-" * 72)
    for i, r in enumerate(results, 1):
        name = Path(r["file"]).name
        print(f"{i:<6}{r['score']:<10.4f}{r['item_id']:<14}{name}")

    if args.card or args.image:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = config.OUTPUT_DIR / f"result_{ts}.png"
        query_item = store.item_of_file(qpath) if qpath else None
        p = make_result_card(qpath, results, out, k=args.k,
                             model_label=getattr(embedder, "name", ""),
                             same_item=query_item)
        print(f"\n结果图已保存：{p}")
        if query_item:
            hit = sum(1 for r in results if r["item_id"] == query_item)
            print(f"查询图属于 {query_item}，Top{len(results)} 里有 {hit} 个同款")


if __name__ == "__main__":
    sys.exit(main())
