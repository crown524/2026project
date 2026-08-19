"""把项目打包成可拷到 U 盘的便携包。

只带代码、提示词、标签、配置模板，不带 .venv（541MB，目标机重建）
和 data/extracted（2.5GB，目标机自己有原始包）。

用法:
    python make_portable.py                     # 输出到桌面
    python make_portable.py --out D:\\           # 直接输出到 U 盘
    python make_portable.py --with-features      # 附带已算好的特征表
    python make_portable.py --with-key           # 附带 API key（默认剔除）
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 必带：代码与资产
CODE_FILES = [
    "config.py", "eval_utils.py", "llm_providers.py",
    "extract_from_zips.py", "build_features.py", "make_download_list.py",
    "run_a1_baseline.py", "run_experiment.py", "run_text_skill.py",
    # v2 校准与残差融合：分类达标那条链路靠它，漏了目标机就跑不了 E0-E3
    "calibrate_and_fuse.py",
    "app_ui.py", "requirements.txt",
    # 目标机的入口，必须随包同行，否则对方无法一键部署
    "setup_and_run.py", "setup_and_run.bat", "make_portable.py",
    # 数据机上的导出工具：把 zip 变成可携带的特征包
    # bat 文件名保持纯 ASCII：cmd.exe 按 GBK 解析，中文文件名在部分机器上会失败
    "export_tool.py", "export_gui.bat", "compare_configs.py",
]
# tests 必带：目标机跑实验前先 pytest 一遍，能立刻发现环境或标签换版问题
CODE_DIRS = ["features", "skills", "tests"]  # 特征抽取器 + Skill 提示词 + 测试
DATA_DIRS = ["data/labels"]                 # 标签 CSV，5KB，必带
# 不带：.venv / data/extracted / outputs/skill_cache / __pycache__


def collect(with_features: bool, with_key: bool) -> list[tuple[Path, str]]:
    """返回 [(源路径, 包内相对路径)]。"""
    items: list[tuple[Path, str]] = []

    for name in CODE_FILES:
        p = HERE / name
        if p.exists():
            items.append((p, name))
        else:
            print(f"  ! 缺少 {name}")

    for d in CODE_DIRS:
        for p in (HERE / d).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                items.append((p, str(p.relative_to(HERE)).replace("\\", "/")))

    for d in DATA_DIRS:
        for p in (HERE / d).rglob("*"):
            if p.is_file():
                items.append((p, str(p.relative_to(HERE)).replace("\\", "/")))

    # 流程图文档（对协作者有用）
    for p in HERE.glob("*.md"):
        items.append((p, p.name))

    if with_features:
        fdir = HERE / "outputs" / "features"
        for p in fdir.glob("*.csv"):
            items.append((p, f"outputs/features/{p.name}"))

    # Skill 模型配置：默认抹掉 key 只留结构
    cfg = HERE / "data" / "skill_models.json"
    if cfg.exists():
        if with_key:
            items.append((cfg, "data/skill_models.json"))
        else:
            import json
            d = json.loads(cfg.read_text(encoding="utf-8"))
            for v in d.values():
                if isinstance(v, dict) and v.get("api_key"):
                    v["api_key"] = ""
            tmp = HERE / "_skill_models_template.json"
            tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            items.append((tmp, "data/skill_models.json"))

    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Desktop"),
                    help="输出目录（可直接给 U 盘盘符）")
    ap.add_argument("--with-features", action="store_true",
                    help="附带已算好的特征表 CSV")
    ap.add_argument("--with-key", action="store_true",
                    help="附带 API key（默认抹掉，避免泄露）")
    ap.add_argument("--folder", action="store_true",
                    help="输出文件夹而非 zip（U 盘直接拷贝更方便）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("清点要打包的内容...")
    items = collect(args.with_features, args.with_key)
    total = sum(p.stat().st_size for p, _ in items)
    print(f"  共 {len(items)} 个文件, {total / 1024:.1f} KB")

    if args.folder:
        dest = out_dir / "daic_project_portable"
        if dest.exists():
            shutil.rmtree(dest)
        for src, rel in items:
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
        target = dest
    else:
        target = out_dir / "daic_project_portable.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for src, rel in items:
                z.write(src, rel)

    tmp = HERE / "_skill_models_template.json"
    if tmp.exists():
        tmp.unlink()

    print(f"\n完成: {target}")
    print(f"大小: {target.stat().st_size / 1024:.1f} KB"
          if target.is_file() else "")
    print("\n下一步（在目标电脑上）:")
    print("  1. 把这个包拷到目标电脑任意位置并解压")
    print("  2. 双击 setup_and_run.bat（或运行 python setup_and_run.py）")
    print("  3. 按提示填写原始数据集所在文件夹路径")
    if not args.with_key:
        print("\n注意: API key 已抹除，目标机需在界面里重新填写。")


if __name__ == "__main__":
    main()
