# -*- coding: utf-8 -*-
"""算出这个检索系统到底准不准，阈值该定在哪。

两种评估口径，别混着看：

【口径一】检索命中率（Recall@K / Top1）
    拿同商品的其他图当标准答案，看能不能搜回来。衡量的是"检索本身行不行"。

【口径二】候选集内阈值扫描
    只看 Top-K 那一屏，在不同阈值下算 precision / recall / F1。
    这才是业务口径——运营看到的就是一屏候选，阈值决定"这一屏里哪些标红"。

为什么不做"全库两两配对"的阈值扫描：
    83 张图会产生 3403 个配对，其中同款只有 154 个，正负比 1:21。
    在这种极端不平衡下，precision 会被几千个"不同商品但长得像"的负样本拖死，
    算出来的 F1 毫无业务含义。所以只把全局分布拿来做对比参考。

用法：
    python evaluate.py
    python evaluate.py --k 5
"""
import argparse
import csv
import sys
from itertools import combinations

import numpy as np

import config
from store import get_store


def evaluate_recall(store, ids, items, mat, k):
    """留一法：用第 i 张图的向量去搜，答案是同 item 的其他图。"""
    hits, hit1, bases = [], [], []
    for i, image_id in enumerate(ids):
        res = store.search(mat[i], k=k + 1)
        res = [r for r in res if r["image_id"] != image_id][:k]
        hit = any(r["item_id"] == items[i] for r in res)
        top1 = bool(res) and res[0]["item_id"] == items[i]
        hits.append(hit)
        hit1.append(top1)
        bases.append(res[0]["score"] if res else 0.0)
    return {
        "n": len(hits),
        "recall": float(np.mean(hits)) if hits else 0.0,
        "top1": float(np.mean(hit1)) if hit1 else 0.0,
        "top1_score_mean": float(np.mean(bases)) if bases else 0.0,
    }


def collect_candidates(store, ids, items, mat, k):
    """收集 Top-K 候选，形如 (query_item, cand_item, score)。"""
    cands = []
    for i, image_id in enumerate(ids):
        res = store.search(mat[i], k=k + 1)
        res = [r for r in res if r["image_id"] != image_id][:k]
        for r in res:
            cands.append((items[i], r["item_id"], r["score"]))
    return cands


def scan_candidates(cands):
    """在候选集上扫阈值，算 precision / recall / F1。"""
    rows = []
    pos_total = sum(1 for q, c, _ in cands if q == c)
    for t in np.round(np.arange(0.60, 0.991, 0.02), 2):
        tp = sum(1 for q, c, s in cands if s >= t and q == c)
        fp = sum(1 for q, c, s in cands if s >= t and q != c)
        fn = sum(1 for q, c, s in cands if s < t and q == c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / pos_total if pos_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": float(t), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "命中": tp, "误报": fp, "漏报": fn,
        })
    return rows, pos_total


def global_distribution(items, mat):
    """全库两两相似度，只看分布重叠，不做阈值判定。"""
    same, diff = [], []
    for i, j in combinations(range(len(items)), 2):
        s = float(mat[i] @ mat[j])
        (same if items[i] == items[j] else diff).append(s)
    return same, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    config.banner("SameFit 效果评估")

    store = get_store()
    ids, items, files, mat = store.load_all()
    if len(ids) < 2:
        raise SystemExit("[错误] 至少要有 2 张图才能评估。")

    from collections import Counter
    counter = Counter(items)
    multi = [it for it, c in counter.items() if c >= 2]
    backend = store.get_meta("embed_backend", "未知")
    print(f"向量 {len(ids)} 条，商品 {len(counter)} 组，多图商品 {len(multi)} 组")
    print(f"特征后端：{backend}   维度：{mat.shape[1]}")
    if not multi:
        raise SystemExit(
            "[错误] 没有哪个商品有 2 张以上图片，算不了命中率。\n"
            "把同一商品的不同角度图片放进同一个文件夹，重新 ingest。"
        )

    # ---------- 口径一 ----------
    r = evaluate_recall(store, ids, items, mat, args.k)
    print(f"\n【口径一】检索命中  （样本 {r['n']} 张）")
    print(f"  Recall@{args.k}  = {r['recall']:.1%}")
    print(f"  Top1 命中率  = {r['top1']:.1%}")
    print(f"  Top1 平均相似度 = {r['top1_score_mean']:.3f}")

    # ---------- 分布对比 ----------
    same, diff = global_distribution(items, mat)
    print(f"\n【分布对比】同款 {len(same)} 对，异款 {len(diff)} 对")
    if same:
        print(f"  同款相似度  均值 {np.mean(same):.3f}  "
              f"[{np.min(same):.3f}, {np.max(same):.3f}]")
    if diff:
        print(f"  异款相似度  均值 {np.mean(diff):.3f}  "
              f"最高 {np.max(diff):.3f}")
    if same and diff:
        overlap_hi = float(np.percentile(diff, 99))
        overlap_lo = float(np.percentile(same, 1))
        print(f"  异款 99 分位 {overlap_hi:.3f} / 同款 1 分位 {overlap_lo:.3f}"
              f"  -> {'区间重叠，单靠阈值无法完全分开' if overlap_hi > overlap_lo else '分得开'}")

    # ---------- 口径二 ----------
    cands = collect_candidates(store, ids, items, mat, args.k)
    rows, pos_total = scan_candidates(cands)
    best = max(rows, key=lambda r: r["f1"])

    print(f"\n【口径二】候选集阈值扫描  （{args.k} 个候选/次，正样本 {pos_total} 个）")
    print(f"{'阈值':<8}{'准确率':<10}{'召回率':<10}{'F1':<8}{'命中':<6}{'误报':<6}{'漏报'}")
    print("-" * 62)
    for x in rows:
        mark = "  <- 最优" if x is best else ""
        print(f"{x['threshold']:<8.2f}{x['precision']:<10.3f}"
              f"{x['recall']:<10.3f}{x['f1']:<8.3f}{x['命中']:<6}"
              f"{x['误报']:<6}{x['漏报']}{mark}")

    # 业务上更关心高精度档：宁可漏，不可误伤
    hi = next((x for x in rows if x["precision"] >= 0.90), None)
    if hi:
        print(f"\n高精度档（准确率>={hi['precision']:.0%}）：阈值 {hi['threshold']:.2f}，"
              f"召回 {hi['recall']:.0%}")
    print(f"F1 最优：阈值 {best['threshold']:.2f}，"
          f"准确率 {best['precision']:.0%}，召回 {best['recall']:.0%}")

    # ---------- 落盘 ----------
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "threshold_scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pair_path = out / "pair_similarity.csv"
    with open(pair_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "item_a", "item_b",
                    "similarity", "is_same_item"])
        for i, j in combinations(range(len(ids)), 2):
            w.writerow([ids[i], ids[j], items[i], items[j],
                        round(float(mat[i] @ mat[j]), 4),
                        int(items[i] == items[j])])

    md = out / "eval_report.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# SameFit 检索效果评估\n\n")
        f.write(f"- 向量总数：{len(ids)}\n")
        f.write(f"- 商品组数：{len(counter)}\n")
        f.write(f"- 多图商品：{len(multi)} 组\n")
        f.write(f"- 特征后端：{backend}（{mat.shape[1]} 维）\n\n")

        f.write("## 一、检索命中\n\n")
        f.write(f"| 指标 | 数值 | 含义 |\n|---|---|---|\n")
        f.write(f"| Recall@{args.k} | **{r['recall']:.1%}** | "
                f"Top{args.k} 里出现同款的比例 |\n")
        f.write(f"| Top1 命中率 | **{r['top1']:.1%}** | 第一名就是同款的比例 |\n")
        f.write(f"| Top1 平均相似度 | {r['top1_score_mean']:.3f} | |\n\n")

        f.write("## 二、同款 / 异款相似度分布\n\n")
        f.write("| | 对数 | 均值 | 范围 |\n|---|---|---|---|\n")
        if same:
            f.write(f"| 同款 | {len(same)} | {np.mean(same):.3f} | "
                    f"{np.min(same):.3f} – {np.max(same):.3f} |\n")
        if diff:
            f.write(f"| 异款 | {len(diff)} | {np.mean(diff):.3f} | "
                    f"– {np.max(diff):.3f} |\n")
        f.write("\n同款和异款的相似度区间存在重叠，说明**单一阈值无法完全区分**，"
                "高分区仍需要人工复核。这也是实际业务里必须保留人工确认环节的原因。\n\n")

        f.write(f"## 三、候选集阈值扫描\n\n")
        f.write(f"只统计 Top-{args.k} 候选（业务口径），正样本 {pos_total} 个。\n\n")
        f.write("| 阈值 | 准确率 | 召回率 | F1 | 命中 | 误报 | 漏报 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for x in rows:
            f.write(f"| {x['threshold']:.2f} | {x['precision']:.3f} | "
                    f"{x['recall']:.3f} | {x['f1']:.3f} | {x['命中']} | "
                    f"{x['误报']} | {x['漏报']} |\n")
        f.write(f"\n- F1 最优阈值：**{best['threshold']:.2f}**"
                f"（准确率 {best['precision']:.0%}，召回 {best['recall']:.0%}）\n")
        if hi:
            f.write(f"- 高精度档阈值：**{hi['threshold']:.2f}**"
                    f"（准确率 {hi['precision']:.0%}，召回 {hi['recall']:.0%}）\n")
        f.write("\n## 四、结论\n\n")
        f.write("检索能力本身（Recall@K）主要取决于向量质量，阈值决定的是"
                "「多像才算撞款」的判定尺度。实际部署时建议双阈值：\n\n")
        f.write("- 高于高精度阈值：直接判为撞款，进人工复核队列\n")
        f.write("- 低于低精度阈值：直接放行\n")
        f.write("- 中间的灰区：人工判定，并把结果回流成新样本\n")

    print(f"\n已写出：\n  {md}\n  {csv_path}\n  {pair_path}")


if __name__ == "__main__":
    sys.exit(main())
