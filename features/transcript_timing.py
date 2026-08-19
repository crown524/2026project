"""A1 组：转写时序特征抽取（确定性，零模型，零 API 成本）。

理论依据：精神运动迟滞（psychomotor retardation）是抑郁的核心症状之一，
响应延迟是其最直接的行为代理。本模块仅用转写自带的 speaker 与时间戳，
不调用任何模型，不受转写文字错误影响，完全可解释。

实测坑位（已处理）：
  - 转写文件名为 .csv 但实际 TAB 分隔
  - 部分行 value 可能为空（未转写片段）
  - speaker 字段可能含多余空白
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def load_transcript(path) -> pd.DataFrame:
    """读取转写文件，处理 TAB 分隔与空白。"""
    df = pd.read_csv(path, sep=C.TRANSCRIPT_SEP)
    df.columns = [c.strip() for c in df.columns]
    if C.COL_SPEAKER in df.columns:
        df[C.COL_SPEAKER] = df[C.COL_SPEAKER].astype(str).str.strip()
    for col in (C.COL_START, C.COL_STOP):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[C.COL_START, C.COL_STOP])
    df[C.COL_VALUE] = df[C.COL_VALUE].fillna("").astype(str)
    return df.sort_values(C.COL_START).reset_index(drop=True)


def compute_response_latencies(df: pd.DataFrame) -> np.ndarray:
    """计算 participant 对 interviewer 的响应延迟。

    定义：participant 某个 turn 的 start_time 减去其紧前一个
    interviewer turn 的 stop_time。仅统计紧跟在 interviewer 之后的
    participant turn（连续 participant turn 之间不算响应延迟）。

    负值（participant 抢话/重叠）保留为 0，因为重叠本身不代表迟滞。
    """
    latencies = []
    prev_speaker = None
    prev_stop = None
    for _, row in df.iterrows():
        spk = row[C.COL_SPEAKER]
        if spk == C.SPEAKER_PARTICIPANT and prev_speaker == C.SPEAKER_INTERVIEWER:
            lat = row[C.COL_START] - prev_stop
            latencies.append(max(0.0, float(lat)))
        prev_speaker, prev_stop = spk, row[C.COL_STOP]
    return np.asarray(latencies, dtype=float)


def _safe_stats(arr: np.ndarray, prefix: str) -> dict:
    """对可能为空的数组做稳健统计，空数组返回 NaN 而非报错。"""
    if arr.size == 0:
        return {f"{prefix}_{k}": np.nan
                for k in ("mean", "median", "std", "p75", "p90", "max")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)) if arr.size > 1 else 0.0,
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def extract(path, participant_id: int) -> dict:
    """抽取单个会话的全部时序特征。"""
    df = load_transcript(path)
    part = df[df[C.COL_SPEAKER] == C.SPEAKER_PARTICIPANT]
    intv = df[df[C.COL_SPEAKER] == C.SPEAKER_INTERVIEWER]

    session_start = float(df[C.COL_START].min()) if len(df) else 0.0
    session_end = float(df[C.COL_STOP].max()) if len(df) else 0.0
    session_dur = max(1e-6, session_end - session_start)

    part_durs = (part[C.COL_STOP] - part[C.COL_START]).to_numpy(dtype=float)
    part_speak_time = float(part_durs.sum()) if part_durs.size else 0.0
    word_counts = part[C.COL_VALUE].str.split().str.len().fillna(0).to_numpy(dtype=float)
    total_words = float(word_counts.sum())

    # 会话内静默：相邻 turn 之间的间隙（不分说话人）
    if len(df) > 1:
        gaps = df[C.COL_START].to_numpy()[1:] - df[C.COL_STOP].to_numpy()[:-1]
        gaps = np.clip(gaps, 0.0, None)
    else:
        gaps = np.asarray([], dtype=float)

    latencies = compute_response_latencies(df)

    feats: dict = {"participant_id": participant_id}
    feats.update(_safe_stats(latencies, "response_latency"))
    feats.update(_safe_stats(part_durs, "turn_duration"))
    feats.update(_safe_stats(word_counts, "turn_words"))
    feats.update(_safe_stats(gaps, "silence_gap"))
    feats.update({
        "n_response_latencies": int(latencies.size),
        "participant_turns": int(len(part)),
        "interviewer_turns": int(len(intv)),
        "total_words": total_words,
        "session_duration_s": session_dur,
        "speak_time_ratio": part_speak_time / session_dur,
        "words_per_second": total_words / max(1e-6, part_speak_time),
        "silence_total_s": float(gaps.sum()) if gaps.size else 0.0,
        "silence_ratio": (float(gaps.sum()) / session_dur) if gaps.size else 0.0,
        # 质量门控用：turn 数过少则统计量不稳定（见方案 §6.2）
        "quality_sufficient": int(len(part) >= 20 and latencies.size >= 10),
    })
    return feats


def extract_all(transcript_dir=None) -> pd.DataFrame:
    """批量抽取目录下全部转写文件。"""
    transcript_dir = transcript_dir or C.TRANSCRIPT_DIR
    rows = []
    for p in sorted(transcript_dir.glob("*_TRANSCRIPT.csv")):
        try:
            pid = int(p.name.split("_")[0])
        except ValueError:
            print(f"  跳过无法解析 ID 的文件: {p.name}")
            continue
        try:
            rows.append(extract(p, pid))
        except Exception as e:
            print(f"  [{pid}] 抽取失败: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


FEATURE_COLUMNS = [
    "response_latency_mean", "response_latency_median", "response_latency_std",
    "response_latency_p75", "response_latency_p90", "response_latency_max",
    "turn_duration_mean", "turn_duration_median", "turn_duration_std",
    "turn_words_mean", "turn_words_median", "turn_words_std",
    "silence_gap_mean", "silence_gap_median", "silence_gap_p90",
    "participant_turns", "total_words", "session_duration_s",
    "speak_time_ratio", "words_per_second", "silence_ratio",
]
