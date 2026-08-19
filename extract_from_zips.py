"""从 DAIC-WOZ 的 *_P.zip 或已解压的 *_P 文件夹中按需抽取文件，跳过 HOG 等巨大且不用的内容。

为什么不全量解压：单个包中 CLNF_hog 占约 73%（301 包为 441MB/604MB），
189 人全解压将达数百 GB。按需抽取后约 58MB/人，总计约 11GB。

支持两种数据源：
1. *_P.zip 压缩包（从 zip 中直接提取）
2. *_P 已解压文件夹（直接复制需要的文件）

用法：
    python extract_from_zips.py                # 抽取全部找到的 zip 或文件夹
    python extract_from_zips.py --only 300 301 # 只抽指定 participant
    python extract_from_zips.py --transcript-only  # 只抽转写（A1 组用，约 14KB/人）
    python extract_from_zips.py --delete-zip    # 抽取成功后删除原 zip（省空间，不影响文件夹）
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile

import config as C

ZIP_PATTERN = re.compile(r"^(\d+)_P\.zip$", re.IGNORECASE)
DIR_PATTERN = re.compile(r"^(\d+)_P$", re.IGNORECASE)


def target_dir_for(filename: str):
    """按文件类型决定落盘目录。返回 None 表示不需要该文件。"""
    low = filename.lower()
    if any(k.lower() in low for k in C.EXCLUDED_KEYWORDS):
        return None
    if filename.endswith("_TRANSCRIPT.csv"):
        return C.TRANSCRIPT_DIR
    if filename.endswith("_COVAREP.csv"):
        return C.COVAREP_DIR
    if filename.endswith("_FORMANT.csv"):
        return C.FORMANT_DIR
    if "_CLNF_" in filename:
        return C.CLNF_DIR
    return None


def discover_sources(only=None):
    """发现 zip 文件或已解压的文件夹。返回 {pid: (path, is_zip)}"""
    found = {}
    for p in C.RAW_ZIP_DIR.iterdir():
        # 检查 zip 文件
        m = ZIP_PATTERN.match(p.name)
        if m and p.is_file():
            pid = int(m.group(1))
            if only and pid not in only:
                continue
            found[pid] = (p, True)
            continue
        # 检查已解压的文件夹
        m = DIR_PATTERN.match(p.name)
        if m and p.is_dir():
            pid = int(m.group(1))
            if only and pid not in only:
                continue
            found[pid] = (p, False)
    return dict(sorted(found.items()))


def extract_from_zip(zip_path, transcript_only=False):
    """从 zip 文件中抽取。返回 (抽取文件数, 抽取字节数, 跳过文件数)。"""
    n_files = n_bytes = n_skip = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.split("/")[-1]
            if not name:
                continue
            if transcript_only and not name.endswith("_TRANSCRIPT.csv"):
                n_skip += 1
                continue
            dest_dir = target_dir_for(name)
            if dest_dir is None:
                n_skip += 1
                continue
            dest = dest_dir / name
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue  # 已抽取且大小一致，跳过（幂等）
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            n_files += 1
            n_bytes += info.file_size
    return n_files, n_bytes, n_skip


def extract_from_dir(dir_path, transcript_only=False):
    """从已解压的文件夹中复制文件。返回 (抽取文件数, 抽取字节数, 跳过文件数)。"""
    n_files = n_bytes = n_skip = 0
    for src_file in dir_path.iterdir():
        if not src_file.is_file():
            continue
        name = src_file.name
        if transcript_only and not name.endswith("_TRANSCRIPT.csv"):
            n_skip += 1
            continue
        dest_dir = target_dir_for(name)
        if dest_dir is None:
            n_skip += 1
            continue
        dest = dest_dir / name
        src_size = src_file.stat().st_size
        if dest.exists() and dest.stat().st_size == src_size:
            continue  # 已抽取且大小一致，跳过（幂等）
        shutil.copy2(src_file, dest)
        n_files += 1
        n_bytes += src_size
    return n_files, n_bytes, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="只处理指定 participant ID")
    ap.add_argument("--transcript-only", action="store_true",
                    help="只抽转写文件（A1 组实验用，体积极小）")
    ap.add_argument("--delete-zip", action="store_true",
                    help="抽取成功后删除原 zip 以释放空间（不影响文件夹）")
    args = ap.parse_args()

    sources = discover_sources(set(args.only) if args.only else None)
    if not sources:
        print(f"未在 {C.RAW_ZIP_DIR} 找到任何 *_P.zip 或 *_P 文件夹。", file=sys.stderr)
        print("请确认 config.RAW_ZIP_DIR 指向正确目录。", file=sys.stderr)
        return 1

    n_zips = sum(1 for _, is_zip in sources.values() if is_zip)
    n_dirs = len(sources) - n_zips
    print(f"发现 {len(sources)} 个数据源: {n_zips} 个 zip, {n_dirs} 个文件夹")
    print(f"participant IDs: {list(sources.keys())}")
    print(f"模式: {'仅转写' if args.transcript_only else '按需抽取(排除 HOG/视频/3D)'}\n")

    total_bytes = 0
    for pid, (source_path, is_zip) in sources.items():
        try:
            if is_zip:
                nf, nb, ns = extract_from_zip(source_path, args.transcript_only)
                source_type = "zip"
            else:
                nf, nb, ns = extract_from_dir(source_path, args.transcript_only)
                source_type = "文件夹"

            total_bytes += nb
            print(f"  [{pid}] ({source_type}) 抽取 {nf} 个文件 / {nb/1e6:.1f} MB，跳过 {ns} 个")

            if args.delete_zip and is_zip and nf > 0:
                source_path.unlink()
                print(f"        已删除原 zip {source_path.name}")
        except zipfile.BadZipFile:
            print(f"  [{pid}] 错误：zip 损坏或下载不完整，需重新下载", file=sys.stderr)
        except Exception as e:
            print(f"  [{pid}] 错误：{e}", file=sys.stderr)

    print(f"\n完成。累计抽取 {total_bytes/1e9:.2f} GB 到 {C.EXTRACTED_DIR}")
    print(f"转写文件数: {len(list(C.TRANSCRIPT_DIR.glob('*_TRANSCRIPT.csv')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
