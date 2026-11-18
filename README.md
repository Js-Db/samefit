# SameFit

跨境女装的同款图像检索：给选品和上架环节做撞款检查。

![检索示例](output/result_dress_005__01.png)

*左为查询图，右侧 Top-5 结果；绿色分数为同款，灰色为异款。*

---

## 背景

跨境女装的选品与上架有一个高频动作：确认新品是否与在库商品重复，或是否构成侵权风险。人工做法是把主图丢进平台搜索、或凭经验比对——单品要花几分钟，而且换角度、换模特、换背景之后，很难判断是不是同一件衣服。

SameFit 把这件事做成机器可召回：新品图进来，在历史商品库里返回最相似的 5 个并给出相似度分数，由人做最终判定。

## 方案

| 环节 | 做法 |
|---|---|
| 编码 | CLIP ViT-B/32，图片映射到 512 维向量空间 |
| 归一化 | L2 归一化后，点积等于余弦相似度 |
| 存储 | SQLite（本地零依赖）／ PostgreSQL + pgvector（HNSW 索引） |
| 召回 | Top-K 余弦相似度 |
| 判定 | 双阈值：≥ 0.94 直接判撞款，0.84 – 0.94 进人工复核，< 0.84 放行 |

**为什么用 CLIP 而不是图像哈希**：pHash 一类方法对裁剪、调色、换背景很敏感，同一件衣服换个模特图就匹配不上。CLIP 在语义空间比较，认的是「这件衣服长什么样」，对角度和背景更鲁棒；代价是向量维度高、需要索引支撑召回速度。

## 效果

在 83 张跨境女装商品图（20 个 SPU）上评估：

| 指标 | 数值 |
|---|---|
| Recall@5 | **95.2%** |
| Top1 命中率 | **90.4%** |
| 同款相似度均值 | 0.854 |
| 异款相似度均值 | 0.692（最高 0.932） |
| F1 最优阈值 | 0.84（准确率 68%，召回 86%） |
| 高精度阈值 | 0.94（准确率 100%，召回 28%） |

同款与异款的相似度区间存在重叠（同款最低 0.542，异款最高 0.932），说明单一阈值无法把两类完全分开，高分区仍有人工确认的必要——这是双阈值设计的依据。

**评估口径**：阈值只在 Top-K 候选集上统计。若改用全库两两配对，83 张图会产生 3403 个配对，其中同款仅 157 个（正负比约 1:21），precision 会被大量负样本稀释，与「运营只看 Top-5 那一屏」的实际场景不符。

## 快速开始

### 环境

Python 3.10+。首次运行会下载约 600MB 的 CLIP 权重（默认走 hf-mirror 镜像）。

```bash
pip install -r requirements.txt
```

### 准备数据

按商品分文件夹，同一商品的多个角度放在同一个文件夹里——评估脚本依赖这个结构做留一法验证。

```
data/images/
  dress_001/
    01.jpg
    02.jpg
  dress_002/
    01.jpg
```

### 入库与检索

```bash
python ingest.py                       # 编码入库
python search.py --image data/images/dress_005/01.jpg --k 5
```

`search.py` 默认排除查询图自身，并在 `output/` 下生成结果图。CLIP 的图文共享同一向量空间，因此也支持文字检索（英文效果优于中文）：

```bash
python search.py --text "red floral summer dress"
```

### 评估

```bash
python evaluate.py --k 5
```

输出召回命中、同款/异款相似度分布、候选集阈值扫描，并写出 `output/eval_report.md` 与两份 CSV。

### 切换到 PostgreSQL

```bash
set SAMEFIT_PG_DSN=postgresql://user:password@host:5432/dbname
python migrate_to_pg.py --dry-run      # 检查
python migrate_to_pg.py                # 迁移已有向量，不重新编码
set SAMEFIT_STORE=pg
python evaluate.py --k 5
```

已入库的向量会直接搬迁到 `vector(512)` 列，不重跑编码——模型升级才需要重算，换存储只做搬迁。

### 自检

```bash
python selftest.py
```

自动生成 9 张测试图，跑完 ingest → search → evaluate 全链路，用于验证环境可用，不需要真实图片和模型权重。

## 项目结构

```
samefit/
  config.py            配置（存储后端、特征后端、路径）
  embedder.py          图片 → 向量（CLIP / 直方图降级版）
  store.py             向量仓库（SQLite / PostgreSQL 双实现）
  ingest.py            批量入库
  search.py            检索 + 结果图生成
  evaluate.py          命中率与阈值扫描
  migrate_to_pg.py     SQLite → PostgreSQL 迁移
  selftest.py          全链路自检
  sql/schema_pg.sql    PostgreSQL 建表语句
  data/images/         图片目录（不入版本库）
  output/              结果图与评估报告
```

## 技术栈

Python · CLIP (transformers/PyTorch) · PostgreSQL + pgvector (HNSW) · SQLite · Pillow · pytest

## 已知限制

- 测试集规模偏小（83 张 / 20 个 SPU），指标只作为链路正确性的基线，不代表生产环境表现。
  真实场景需要上千 SKU、并控制品类分布——女装的难点在于同色系不同款极多，样本不足会高估准确率。
- SQLite 后端采用全量载入内存做矩阵乘法，适合万级以内；更大规模需依赖 pgvector 的 HNSW 索引。
- 检索结果依赖 CLIP 的通用视觉表征，未针对服装品类做微调；版型相近但颜色不同的款会被判为高相似，
  这是当前准确率上限的主要来源。
