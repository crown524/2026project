"""A5 组：CLNF 面部特征（AU / gaze / pose）会话级聚合。

实测坑位（已处理）：
  - 三个文件均为 "逗号+空格" 分隔，带表头，列名含前导空格，须 strip
  - 均含 confidence 与 success 列；跟踪失败帧必须过滤
    （过滤条件：success==1 且 confidence >= 阈值）
  - @30fps

特征依据：
  - AU04(皱眉)、AU12(嘴角上扬)、AU15(嘴角下垂) 与抑郁的表情减弱
    （flat affect）在文献中有较扎实基础
  - 注视回避：gaze 向量偏离正前方的角度与离散度
  - pose：低头角度（Rx）与头部运动量（静止度）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

AU_INTENSITY = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU09_r",
                "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r",
                "AU25_r", "AU26_r"]
AU_PRESENCE = ["AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c"]


def _read_clnf(path) -> pd.DataFrame:
    """读取 CLNF 文件并按跟踪质量过滤。"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    n_total = len(df)
    ok = (df["success"] == C.CLNF_SUCCESS_VALUE) & \
         (df["confidence"] >= C.CLNF_MIN_CONFIDENCE)
    df = df[ok].reset_index(drop=True)
    df.attrs["valid_ratio"] = (len(df) / n_total) if n_total else 0.0
    return df


def _stats(arr: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    return {f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr)) if arr.size > 1 else 0.0}


def extract(participant_id: int) -> dict:
    feats: dict = {"participant_id": participant_id}
    valid_ratios = []

    # ---- AU：表情强度与激活率 ----
    au_path = C.CLNF_DIR / f"{participant_id}_CLNF_AUs.txt"
    if au_path.exists():
        au = _read_clnf(au_path)
        valid_ratios.append(au.attrs["valid_ratio"])
        for col in AU_INTENSITY:
            if col in au.columns:
                feats.update(_stats(au[col].to_numpy(), col.lower()))
        for col in AU_PRESENCE:
            if col in au.columns:
                feats[f"{col.lower()}_rate"] = float(au[col].mean()) if len(au) else np.nan
        # 总体表情活跃度：全部 AU 强度均值的均值（flat affect 的粗代理）
        inten = [c for c in AU_INTENSITY if c in au.columns]
        if inten and len(au):
            feats["au_overall_intensity"] = float(au[inten].to_numpy().mean())
            # 表情变化度：AU 强度的帧间变化量
            diffs = np.abs(np.diff(au[inten].to_numpy(), axis=0))
            feats["au_temporal_variation"] = float(diffs.mean()) if diffs.size else np.nan
    else:
        feats["au_missing"] = 1

    # ---- gaze：注视方向与离散度 ----
    gz_path = C.CLNF_DIR / f"{participant_id}_CLNF_gaze.txt"
    if gz_path.exists():
        gz = _read_clnf(gz_path)
        valid_ratios.append(gz.attrs["valid_ratio"])
        # 双眼视线向量 (x_0,y_0,z_0), (x_1,y_1,z_1)；正前方约为 (0,0,-1)
        need = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]
        if all(c in gz.columns for c in need) and len(gz):
            v = gz[need].to_numpy(dtype=float)
            mean_gaze = (v[:, :3] + v[:, 3:]) / 2.0
            norm = np.linalg.norm(mean_gaze, axis=1)
            norm[norm == 0] = 1.0
            unit = mean_gaze / norm[:, None]
            # 与正前方 (0,0,-1) 的夹角
            cos_fwd = np.clip(-unit[:, 2], -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_fwd))
            feats.update(_stats(angle, "gaze_off_forward_deg"))
            # 视线游移：帧间角度变化
            wander = np.abs(np.diff(angle))
            feats.update(_stats(wander, "gaze_wander"))
    else:
        feats["gaze_missing"] = 1

    # ---- pose：低头与静止度 ----
    ps_path = C.CLNF_DIR / f"{participant_id}_CLNF_pose.txt"
    if ps_path.exists():
        ps = _read_clnf(ps_path)
        valid_ratios.append(ps.attrs["valid_ratio"])
        cols = [c for c in ps.columns if c not in
                ("frame", "timestamp", "confidence", "success")]
        # 官方列序: Tx, Ty, Tz, Rx, Ry, Rz（平移 mm + 旋转 rad）
        if len(cols) >= 6 and len(ps):
            rx = ps[cols[3]].to_numpy(dtype=float)   # 俯仰角（低头为正/负视坐标系）
            feats.update(_stats(np.degrees(rx), "head_pitch_deg"))
            rot = ps[cols[3:6]].to_numpy(dtype=float)
            motion = np.linalg.norm(np.diff(rot, axis=0), axis=1)
            feats.update(_stats(np.degrees(motion), "head_motion"))
            feats["head_still_ratio"] = (
                float((motion < np.radians(0.2)).mean()) if motion.size else np.nan)
    else:
        feats["pose_missing"] = 1

    # 质量：有效跟踪帧占比（供融合层降权）
    feats["clnf_valid_ratio"] = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    feats["quality_sufficient"] = int(feats["clnf_valid_ratio"] >= 0.6)
    return feats


def extract_all() -> pd.DataFrame:
    pids = sorted({int(p.name.split("_")[0])
                   for p in C.CLNF_DIR.glob("*_CLNF_AUs.txt")})
    rows = []
    for pid in pids:
        try:
            rows.append(extract(pid))
        except Exception as e:
            print(f"  [{pid}] 视觉抽取失败: {e}")
    return (pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
            if rows else pd.DataFrame())
