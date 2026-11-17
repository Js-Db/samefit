# -*- coding: utf-8 -*-
"""冒烟测试：不用真实图片、不用 torch，30 秒验证整条链路是通的。

它会自己在临时目录里生成 9 张纯色块图（3 个商品各 3 张），
然后依次跑 ingest -> search -> evaluate，检查每个环节有没有报错。

你拿到项目后第一件事就跑这个。看到全部 PASS，说明环境没问题，
后面只需要把你自己的图片丢进 data/images/ 就行。

用法：python selftest.py
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
GROUPS = {
    "item_red": (205, 45, 50),
    "item_blue": (45, 85, 205),
    "item_green": (50, 170, 85),
}
PER_GROUP = 3


def make_image(path, rgb, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (256, 320), rgb)
    d = ImageDraw.Draw(img)
    for _ in range(8):
        x0, y0 = rnd.randint(0, 200), rnd.randint(0, 260)
        d.ellipse(
            [x0, y0, x0 + rnd.randint(20, 50), y0 + rnd.randint(20, 50)],
            fill=(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)),
        )
    img.save(path, "JPEG", quality=88)


def run(script, args, env, label):
    print(f"\n>>> {label}")
    proc = subprocess.run(
        [sys.executable, script, *args],
        cwd=str(HERE), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.strip().splitlines()[-14:]:
        print("    " + line)
    return proc.returncode, out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="samefit_selftest_"))
    img_dir = tmp / "images"
    try:
        for gid, rgb in GROUPS.items():
            d = img_dir / gid
            d.mkdir(parents=True, exist_ok=True)
            for i in range(PER_GROUP):
                make_image(d / f"img_{i}.jpg", rgb, seed=hash(gid) % 1000 + i)
        total = sum(1 for _ in img_dir.rglob("*.jpg"))
        print(f"已生成测试图片 {total} 张，位于 {img_dir}")

        env = os.environ.copy()
        env.update({
            "SAMEFIT_EMBED": "hist",
            "SAMEFIT_STORE": "sqlite",
            "SAMEFIT_IMAGES": str(img_dir),
            "SAMEFIT_DB": str(tmp / "test.db"),
            "SAMEFIT_OUTPUT": str(tmp / "output"),
            "PYTHONIOENCODING": "utf-8",
        })

        checks = []

        rc, out = run("ingest.py", [], env, "1/3 入库 ingest.py")
        checks.append(("ingest 入库成功", rc == 0 and "入库完成" in out))

        q = img_dir / "item_red" / "img_0.jpg"
        rc, out = run("search.py", ["--image", str(q), "--k", "3", "--exclude-self"],
                      env, "2/3 检索 search.py")
        checks.append(("search 检索成功", rc == 0 and "结果图已保存" in out))

        rc, out = run("evaluate.py", ["--k", "3"], env, "3/3 评估 evaluate.py")
        checks.append(("evaluate 产出 Recall", rc == 0 and "Recall@" in out))
        checks.append(("evaluate 产出 F1 最优阈值", rc == 0 and "F1 最优" in out))

        cd = tmp / "output"
        cards = list(cd.glob("result_*.png")) if cd.exists() else []
        report = cd / "eval_report.md"
        checks.append(("结果图已生成", len(cards) >= 1))
        checks.append(("评估报告已生成", report.exists()))

        print("\n" + "=" * 56)
        print("  自检结果")
        print("=" * 56)
        ok = True
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            ok = ok and passed
        print("=" * 56)
        print("  全部通过，环境没问题，可以放你自己的图片了。"
              if ok else "  有失败项，把上面带报错的输出发我。")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
