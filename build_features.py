"""统一特征构建：三路确定性特征一次抽取，落盘为独立 CSV + 合并宽表。

用法：
    python build_features.py

产出（outputs/features/）：
    a1_transcript_timing.csv   转写时序
    a3_acoustic.csv            声学（COVAREP + FORMANT）
    a5_visual.csv              视觉（CLNF AU/gaze/pose）
    all_features.csv           三路合并宽表（列加前缀 tt_/ac_/vi_）

文本 Skill（A2/D 组）需调用 LLM API，不在本脚本内，单独实现。
"""
from __future__ import annotations

import sys

import pandas as pd

import config as C
from features import acoustic, transcript_timing, visual


def main() -> int:
    print("=" * 62)
    print("三路确定性特征抽取")
    print("=" * 62)

    print("\n[1/3] 转写时序 …")
    tt = transcript_timing.extract_all()
    print(f"  {len(tt)} 个会话")

    print("[2/3] 声学（COVAREP 较大，约数秒/人）…")
    ac = acoustic.extract_all()
    print(f"  {len(ac)} 个会话")

    print("[3/3] 视觉 …")
    vi = visual.extract_all()
    print(f"  {len(vi)} 个会话")

    if tt.empty and ac.empty and vi.empty:
        print("\n无任何特征可抽取。请先运行 extract_from_zips.py")
        return 1

    for df, name in ((tt, "a1_transcript_timing"),
                     (ac, "a3_acoustic"), (vi, "a5_visual")):
        if not df.empty:
            p = C.FEATURE_DIR / f"{name}.csv"
            df.to_csv(p, index=False, encoding="utf-8-sig")
            print(f"  已保存 {p.name}: {df.shape[0]} 行 × {df.shape[1]} 列")

    # 合并宽表：模态前缀防列名冲突
    merged = None
    for df, prefix in ((tt, "tt"), (ac, "ac"), (vi, "vi")):
        if df.empty:
            continue
        d = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns
                               if c != "participant_id"})
        merged = d if merged is None else merged.merge(
            d, on="participant_id", how="outer")
    if merged is not None:
        p = C.FEATURE_DIR / "all_features.csv"
        merged.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n合并宽表: {merged.shape[0]} 行 × {merged.shape[1]} 列 → {p}")

        qcols = [c for c in merged.columns if c.endswith("quality_sufficient")
                 or c.endswith("valid_ratio") or c.endswith("voiced_ratio")]
        if qcols:
            print("\n质量概览（融合层降权依据）：")
            print(merged[["participant_id"] + qcols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
