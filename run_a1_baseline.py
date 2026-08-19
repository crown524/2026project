"""数据自检 + A1 转写时序特征抽取入口。

职责收窄（训练评测统一走 run_experiment.py，避免逻辑重复）：
  1. 数据完整性自检（转写数量、标签文件、split 归属、test 纪律提示）
  2. 抽取 A1 转写时序特征并落盘
  3. 指引下一步命令

用法：
    python run_a1_baseline.py              # 自检 + 抽特征
    python run_a1_baseline.py --check-only # 只自检
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import config as C
import eval_utils as E
from features import transcript_timing as tt


def check_data_readiness() -> dict:
    n = len(list(C.TRANSCRIPT_DIR.glob("*_TRANSCRIPT.csv")))
    s = {"n_transcripts": n,
         "train_split_exists": C.TRAIN_SPLIT.exists(),
         "dev_split_exists": C.DEV_SPLIT.exists(),
         "test_split_exists": C.TEST_SPLIT.exists()}
    s["has_labels"] = s["train_split_exists"] and s["dev_split_exists"]
    s["can_train"] = s["has_labels"] and n >= 30
    return s


def print_readiness(status: dict) -> None:
    print("=" * 62)
    print("数据完整性自检")
    print("=" * 62)
    print(f"  转写文件数           : {status['n_transcripts']}  (完整集应为 189)")
    print(f"  train_split 标签文件 : {'✓' if status['train_split_exists'] else '✗ 缺失'}")
    print(f"  dev_split 标签文件   : {'✓' if status['dev_split_exists'] else '✗ 缺失'}")
    print(f"  test_split 划分文件  : {'✓' if status['test_split_exists'] else '✗ 缺失'}")

    if status["train_split_exists"]:
        ids = E.split_membership()
        local = sorted(int(p.name.split("_")[0])
                       for p in C.TRANSCRIPT_DIR.glob("*_TRANSCRIPT.csv"))
        n_tr = sum(1 for i in local if i in ids["train"])
        n_dv = sum(1 for i in local if i in ids["dev"])
        n_te = sum(1 for i in local if i in ids["test"])
        print(f"\n  官方划分规模         : train={len(ids['train'])}, "
              f"dev={len(ids['dev'])}, test={len(ids['test'])}")
        print(f"  本地样本 split 归属  : train={n_tr}, dev={n_dv}, test={n_te}")
        if n_te and not (n_tr or n_dv):
            print("  ⚠ 本地样本全部属于 test 集！")
            print("    test 样本只能用于流水线调试与最终评估，")
            print("    禁止用于提示词开发或任何调参（方案 §7.3）。")
            missing_tr = sorted(ids["train"])[:8]
            print(f"    请优先下载 train 集包: {missing_tr} ...")
        elif status["has_labels"]:
            have = set(local)
            missing_tr = sorted(ids["train"] - have)
            missing_dv = sorted(ids["dev"] - have)
            if missing_tr or missing_dv:
                print(f"  尚缺 train 包 {len(missing_tr)} 个、"
                      f"dev 包 {len(missing_dv)} 个")
    print()

    if not status["has_labels"]:
        print("  ⚠ 缺少标签文件（train/dev split csv），无法训练评测。")
    elif not status["can_train"]:
        print(f"  ⚠ 仅 {status['n_transcripts']} 个样本，可验证流水线，"
              "不足以产出有意义指标。")
    else:
        print("  ✓ 数据就绪。")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    status = check_data_readiness()
    print_readiness(status)
    if args.check_only:
        return 0
    if status["n_transcripts"] == 0:
        print("无转写文件。先运行: python extract_from_zips.py --transcript-only")
        return 1

    print("=" * 62)
    print("抽取 A1 转写时序特征")
    print("=" * 62)
    feats = tt.extract_all()
    if feats.empty:
        return 1
    out = C.FEATURE_DIR / "a1_transcript_timing.csv"
    feats.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  {len(feats)} 个会话 × {feats.shape[1]} 列 → {out}")

    key = ["participant_id", "response_latency_mean", "response_latency_p90",
           "participant_turns", "total_words", "speak_time_ratio",
           "silence_ratio", "quality_sufficient"]
    print("\n  关键特征预览：")
    print(feats[key].to_string(index=False))

    print("\n下一步：")
    print("  三路全抽   : python build_features.py")
    print("  A1 实验    : python run_experiment.py --lanes tt")
    print("  文本 Skill : python run_text_skill.py --skill text_observation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
