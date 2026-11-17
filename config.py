# -*- coding: utf-8 -*-
"""SameFit 全局配置。要改的东西全在这一个文件里，别的文件不用动。"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# 三个路径都支持环境变量覆盖，方便你另开一套实验数据而不动正式库
IMAGE_DIR = Path(os.getenv("SAMEFIT_IMAGES", str(BASE_DIR / "data" / "images")))
DB_PATH = Path(os.getenv("SAMEFIT_DB", str(BASE_DIR / "data" / "samefit.db")))
OUTPUT_DIR = Path(os.getenv("SAMEFIT_OUTPUT", str(BASE_DIR / "output")))

# 向量存储后端：
#   sqlite —— 第 1~5 关用这个，零配置，装上 Python 就能跑
#   pg     —— 第 6 关升级用，需要 PostgreSQL + pgvector
STORE_BACKEND = os.getenv("SAMEFIT_STORE", "sqlite")

# 只有 STORE_BACKEND=pg 时才会用到。改成你自己的云库或本地库连接串。
PG_DSN = os.getenv(
    "SAMEFIT_PG_DSN",
    "postgresql://postgres:postgres@localhost:5432/samefit",
)

# 特征提取后端：
#   hist —— 颜色直方图，不需要装 torch，5 分钟就能跑通全流程，但效果一般
#   clip —— CLIP ViT-B/32，真实可用效果，需要下载约 600MB 权重
EMBED_BACKEND = os.getenv("SAMEFIT_EMBED", "clip")

# CLIP 模型（中文/国内网络建议保持默认，配合 HF 镜像）
CLIP_MODEL = os.getenv("SAMEFIT_CLIP_MODEL", "openai/clip-vit-base-patch32")

# HuggingFace 国内镜像，避免下载卡死。已经设过就不覆盖。
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def banner(title):
    """统一的分隔标题，让终端输出好看一点。"""
    line = "=" * 56
    print(f"\n{line}\n  {title}\n{line}")
