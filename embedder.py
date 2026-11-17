# -*- coding: utf-8 -*-
"""把一张图片变成一串数字（向量）。两个实现，效果天差地别。

ClipEmbedder  真效果，靠 CLIP 模型。需要 torch + transformers。
HistEmbedder  降级版，只算颜色分布。不装 torch 也能跑，用来先跑通流程。
"""
import os

import numpy as np
from PIL import Image

import config

# 必须在 import transformers 之前设置，否则镜像不生效
os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)


def build_embedder(backend=None):
    backend = backend or config.EMBED_BACKEND
    if backend == "hist":
        return HistEmbedder()
    return ClipEmbedder()


class ClipEmbedder:
    """CLIP ViT-B/32：图片和文字都能映射到同一个 512 维空间。

    归一化之后，两个向量的点积就等于余弦相似度，取值范围 -1 ~ 1。
    """

    dim = 512
    name = "CLIP ViT-B/32 · 512d · cosine"
    backend_name = "clip"

    def __init__(self, model_name=None):
        model_name = model_name or config.CLIP_MODEL
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise SystemExit(
                "\n[错误] 没装 torch / transformers。两条路选一条：\n"
                "  A. 先用降级版跑通流程（不用下载任何东西，1 分钟）：\n"
                "       python ingest.py --backend hist\n"
                "       python search.py --image 你的图.jpg --backend hist\n"
                "  B. 装真模型（下载约 2-4GB，网络好的话 10 分钟）：\n"
                "       python -m pip install torch transformers "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                f"\n原始报错：{exc}"
            )
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[clip] 加载模型 {model_name}  device={self.device}")
        print("[clip] 首次运行要下载约 600MB 权重，走 hf-mirror 镜像，耐心等 1~5 分钟")
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as exc:
            raise SystemExit(
                "\n[错误] 模型下载失败。国内网络请确认镜像已生效，手动试一次：\n"
                "  set HF_ENDPOINT=https://hf-mirror.com\n"
                "  python -c \"from transformers import CLIPModel; "
                "CLIPModel.from_pretrained('openai/clip-vit-base-patch32')\"\n"
                f"原始报错：{exc}"
            )
        print("[clip] 模型就绪")

    @staticmethod
    def _to_tensor(feat):
        # 不同 transformers 版本返回类型不一样，这里统一成 tensor
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        elif isinstance(feat, (tuple, list)):
            feat = feat[0]
        return feat

    def _normalize(self, feat):
        return feat / feat.norm(p=2, dim=-1, keepdim=True)

    def embed_image(self, path):
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feat = self._to_tensor(self.model.get_image_features(**inputs))
        return self._normalize(feat)[0].float().cpu().numpy()

    def embed_text(self, text):
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with self.torch.no_grad():
            feat = self._to_tensor(self.model.get_text_features(**inputs))
        return self._normalize(feat)[0].float().cpu().numpy()


class HistEmbedder:
    """颜色直方图。只为让流程先跑起来，别拿它当最终效果。

    RGB 三通道各 32 个 bin，拼成 96 维。好处是零依赖、毫秒级；
    坏处是它只认颜色，换个背景、换个角度就抓瞎——正好反过来证明 CLIP 的价值。
    """

    dim = 96
    name = "RGB histogram · 96d · cosine"
    backend_name = "hist"

    def embed_image(self, path):
        img = Image.open(path).convert("RGB").resize((128, 128))
        arr = np.asarray(img, dtype=np.float32)
        parts = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 256))
            parts.append(hist.astype("float32"))
        vec = np.concatenate(parts)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def embed_text(self, text):
        raise NotImplementedError(
            "hist 后端不支持文字检索。要文字搜图请切到 clip 后端。"
        )
