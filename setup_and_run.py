"""在新电脑上一键部署并启动。

做四件事：建虚拟环境 → 装依赖 → 定位数据集 → 自检并起界面。
可重复运行，已完成的步骤会跳过。

用法:
    python setup_and_run.py                      # 交互式，问数据集路径
    python setup_and_run.py --data D:\\daic       # 直接指定数据集文件夹
    python setup_and_run.py --no-ui              # 只装环境不起界面
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
STREAMLIT = VENV / ("Scripts/streamlit.exe" if os.name == "nt" else "bin/streamlit")
LOCAL_CFG = HERE / "local_paths.json"


def step(n: int, msg: str) -> None:
    print(f"\n[{n}/4] {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, **kw)


def ensure_venv() -> bool:
    step(1, "检查 Python 虚拟环境")
    if PY.exists():
        print("  已存在，跳过")
        return True
    v = sys.version_info
    print(f"  当前 Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        print("  ! 需要 Python 3.10 或更高版本，请先升级 Python")
        return False
    print("  创建 .venv ...")
    r = run([sys.executable, "-m", "venv", str(VENV)])
    if r.returncode != 0 or not PY.exists():
        print("  ! 创建失败。请手动执行: python -m venv .venv")
        return False
    print("  完成")
    return True


def ensure_deps() -> bool:
    step(2, "安装依赖")
    req = HERE / "requirements.txt"
    if not req.exists():
        print("  ! 找不到 requirements.txt")
        return False
    probe = run([str(PY), "-c", "import pandas, sklearn, streamlit, httpx"],
                capture_output=True)
    if probe.returncode == 0:
        print("  依赖已齐备，跳过")
        return True
    print("  安装中（首次约需 2-5 分钟）...")
    run([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    r = run([str(PY), "-m", "pip", "install", "-q", "-r", str(req)])
    if r.returncode != 0:
        print("  ! 安装失败。若在国内网络较慢，可换镜像源重试:")
        print("    .venv\\Scripts\\python.exe -m pip install -r requirements.txt "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False
    print("  完成")
    return True


def find_dataset(cli_path: str | None) -> Path | None:
    """定位放着 *_P.zip 的文件夹。"""
    step(3, "定位数据集")

    def count_zip(p: Path) -> int:
        try:
            return len(list(p.glob("*_P.zip")))
        except OSError:
            return 0

    if cli_path:
        p = Path(cli_path).expanduser()
        if p.is_dir():
            n = count_zip(p)
            print(f"  指定路径: {p}  找到 {n} 个 *_P.zip")
            if n:
                return p
            print("  ! 该文件夹里没有 *_P.zip")
        else:
            print(f"  ! 路径不存在: {p}")

    # 自动猜几个常见位置
    guesses = [
        Path.home() / "Desktop" / "daic数据集",
        Path.home() / "Desktop" / "daic",
        HERE / "data" / "zips",
    ]
    for d in "DEFG":
        guesses += [Path(f"{d}:/daic数据集"), Path(f"{d}:/daic"),
                    Path(f"{d}:/DAIC-WOZ"), Path(f"{d}:/dataset/daic")]
    for g in guesses:
        if g.is_dir() and count_zip(g):
            print(f"  自动发现: {g}  ({count_zip(g)} 个包)")
            return g

    print("  未自动找到。请输入放着 300_P.zip 这类文件的文件夹完整路径")
    print("  （直接回车可跳过，稍后在界面里设置）")
    try:
        raw = input("  路径> ").strip().strip('"')
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    p = Path(raw)
    if p.is_dir() and count_zip(p):
        print(f"  找到 {count_zip(p)} 个包")
        return p
    print(f"  ! 无效或无 *_P.zip: {p}")
    return None


def save_path(zip_dir: Path | None) -> None:
    if zip_dir is None:
        return
    cfg = {}
    if LOCAL_CFG.exists():
        try:
            cfg = json.loads(LOCAL_CFG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
    cfg["zip_dir"] = str(zip_dir)
    LOCAL_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"  路径已记入 local_paths.json")


def selfcheck() -> None:
    step(4, "自检")
    code = (
        "import config as C\n"
        "from eval_utils import load_labels\n"
        "d = load_labels()\n"
        "print(f'  标签: {len(d)} 人')\n"
        "print(f'  数据集目录: {C.RAW_ZIP_DIR}')\n"
        "z = len(list(C.RAW_ZIP_DIR.glob('*_P.zip'))) "
        "if C.RAW_ZIP_DIR.exists() else 0\n"
        "print(f'  待抽取压缩包: {z} 个')\n"
        "t = len(list(C.TRANSCRIPT_DIR.glob('*')))\n"
        "print(f'  已抽取转写: {t} 个')\n"
    )
    # 子进程按 UTF-8 输出，父进程须以同一编码解码；Windows 默认 GBK 会解码失败
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = run([str(PY), "-c", code], capture_output=True, cwd=HERE, env=env)
    out = (r.stdout or b"").decode("utf-8", errors="replace").rstrip()
    err = (r.stderr or b"").decode("utf-8", errors="replace").strip()
    print(out or "  (自检无输出)")
    if r.returncode != 0:
        print("  ! 自检报错：")
        print("   ", err[:400])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="放着 *_P.zip 的文件夹")
    ap.add_argument("--no-ui", action="store_true", help="只装环境，不启动界面")
    args = ap.parse_args()

    print("=" * 58)
    print("  DAIC 抑郁风险预测项目 · 环境部署")
    print("=" * 58)

    if not ensure_venv() or not ensure_deps():
        sys.exit(1)
    save_path(find_dataset(args.data))
    selfcheck()

    print("\n" + "=" * 58)
    print("  部署完成")
    print("=" * 58)
    print("\n常用命令（在本文件夹下执行）:")
    exe = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    print(f"  {exe} extract_from_zips.py --delete-zip   # 抽取特征文件")
    print(f"  {exe} build_features.py                   # 生成特征宽表")
    print(f"  {exe} run_experiment.py --lanes tt        # 跑实验")
    print(f"  {exe} make_download_list.py               # 查缺哪些包")
    print(f"  {exe} -m pytest tests/ -q                 # 自检（建议先跑）")
    print("\nv2 链路（分类已达标的那条，需先在界面里填 API key）:")
    print(f"  {exe} run_text_skill.py --skill text_observation   --jobs 8")
    print(f"  {exe} run_text_skill.py --skill d1_direct_scoring  --jobs 8")
    print(f"  {exe} calibrate_and_fuse.py --config E0,E1,E2,E3 \\")
    print("      --obs-file a2_text_observations__schema_v2_1.csv")

    if args.no_ui:
        return
    print("\n启动界面 http://localhost:8501  （Ctrl+C 停止）")
    if not STREAMLIT.exists():
        print("  ! 找不到 streamlit，请检查依赖安装")
        return
    try:
        run([str(STREAMLIT), "run", "app_ui.py",
             "--server.address", "0.0.0.0", "--server.port", "8501"], cwd=HERE)
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
