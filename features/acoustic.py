"""A3 组：COVAREP + FORMANT 声学特征会话级聚合。

实测坑位（已处理）：
  - COVAREP.csv 无表头，74 列，@100Hz
  - 第 2 列（index 1）为 VUV 发声标志；未发声帧的 F0 及声门特征均为 0/无效，
    必须按 VUV==1 过滤后再统计，否则均值被零污染
  - FORMANT.csv 无表头，5 列（F1-F5），@100Hz

特征选取原则：只取可解释、文献中与抑郁相关性有依据的维度
（F0 韵律、声门源特征、能量、发声占比），不整表拍平——
74 维 × 6 统计量 = 444 维对 n=107 的训练集是灾难。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# COVAREP 官方列序（前 11 列 + MCEP 起点）。完整表见官方文档 PDF。
COVAREP_COLS = {
    "f0": 0,        # 基频
    "vuv": 1,       # 发声标志 1/0
    "naq": 2,       # Normalized Amplitude Quotient（声门源）
    "qoq": 3,       # Quasi-Open Quotient
    "h1h2": 4,      # H1-H2 谱倾斜
    "psp": 5,       # Parabolic Spectral Parameter
    "mdq": 6,       # Maxima Dispersion Quotient（气声度）
    "peakslope": 7, # 峰值斜率（紧张度）
    "rd": 8,        # Rd 声门形状参数
    "mcep0": 11,    # 第 0 阶倒谱 ≈ 能量
}


def _stats(arr: np.ndarray, prefix: str) -> dict:
    """稳健统计。空数组返回 NaN（由下游 imputer 处理）。"""
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{k}": np.nan
                for k in ("mean", "std", "median", "iqr", "p10", "p90")}
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)) if arr.size > 1 else 0.0,
        f"{prefix}_median": float(q50),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p90": float(q90),
    }


def _voiced_segments(vuv: np.ndarray) -> np.ndarray:
    """连续发声段长度（帧数）。用于韵律节奏特征。"""
    if vuv.size == 0:
        return np.asarray([])
    changes = np.diff(np.concatenate([[0], vuv, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return (ends - starts).astype(float)


def extract(participant_id: int) -> dict:
    """抽取单人的会话级声学特征。"""
    cov_path = C.COVAREP_DIR / f"{participant_id}_COVAREP.csv"
    fmt_path = C.FORMANT_DIR / f"{participant_id}_FORMANT.csv"

    feats: dict = {"participant_id": participant_id}

    if cov_path.exists():
        cov = pd.read_csv(cov_path, header=None).to_numpy(dtype=float)
        vuv = (cov[:, COVAREP_COLS["vuv"]] > 0.5).astype(int)
        voiced = cov[vuv == 1]

        feats["voiced_ratio"] = float(vuv.mean()) if vuv.size else np.nan
        feats["n_frames_total"] = int(len(cov))
        feats["n_frames_voiced"] = int(vuv.sum())

        # 发声段节奏：段长与段数（语流连贯性的声学侧代理）
        segs = _voiced_segments(vuv)
        feats.update(_stats(segs / C.COVAREP_HZ, "voiced_seg_dur"))
        feats["voiced_seg_per_min"] = (
            float(len(segs)) / (len(cov) / C.COVAREP_HZ / 60.0)
            if len(cov) else np.nan)

        # 仅在发声帧上统计韵律与声门特征
        for name in ("f0", "naq", "qoq", "h1h2", "psp", "mdq",
                     "peakslope", "rd", "mcep0"):
            col = COVAREP_COLS[name]
            arr = voiced[:, col] if voiced.size else np.asarray([])
            feats.update(_stats(arr, name))
    else:
        feats["covarep_missing"] = 1

    if fmt_path.exists():
        fmt = pd.read_csv(fmt_path, header=None).to_numpy(dtype=float)
        # 共振峰仅在发声时有意义；无 VUV 对齐时用非零行近似
        nonzero = fmt[(fmt[:, 0] > 0)]
        for i, name in enumerate(("f1", "f2", "f3")):
            arr = nonzero[:, i] if nonzero.size else np.asarray([])
            feats.update(_stats(arr, f"formant_{name}"))
    else:
        feats["formant_missing"] = 1

    # 质量门控：发声帧过少则整路降权（供融合层使用）
    feats["quality_sufficient"] = int(
        feats.get("n_frames_voiced", 0) >= 30 * C.COVAREP_HZ)  # ≥30 秒发声
    return feats


def extract_all() -> pd.DataFrame:
    rows = []
    for p in sorted(C.COVAREP_DIR.glob("*_COVAREP.csv")):
        pid = int(p.name.split("_")[0])
        try:
            rows.append(extract(pid))
        except Exception as e:
            print(f"  [{pid}] 声学抽取失败: {e}")
    return (pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
            if rows else pd.DataFrame())
