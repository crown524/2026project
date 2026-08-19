"""D1 分数校准 + 结构化残差融合（方案 v2 的 P2/P5/P6）。

为什么需要这个脚本：D1 的 Spearman 最强（0.742）但 RMSE 最差（5.191），
说明 LLM 知道"谁更严重"却不知道"该给多少分"——这是尺度偏差，不是没信号。
把尺度校准和残差修正分开做，两个问题各自可诊断：

    base     = calibrator(d1_raw)              只学 2 个参数，train 折内拟合
    residual = residual_model(structured)      只学 base 没解释掉的部分
    final    = clip(base + residual, 0, 24)

关键纪律（方案 §4.3）：校准器、残差模型、分类阈值全部只在外层训练折内拟合。
用全体 142 人先拟合校准器再报告 LOSO 是泄漏，会把 RMSE 系统性压低。

用法:
    python calibrate_and_fuse.py --config E1          # 只校准
    python calibrate_and_fuse.py --config E3          # 校准 + 残差
    python calibrate_and_fuse.py --config all         # 跑完整矩阵
    python calibrate_and_fuse.py --config E3 --lanes obs,tt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import eval_utils as E

PHQ8_MIN, PHQ8_MAX = 0.0, 24.0

LANE_FILES = {
    "tt": "a1_transcript_timing.csv",
    "ac": "a3_acoustic.csv",
    "vi": "a5_visual.csv",
    "obs": "a2_text_observations.csv",
    "d1": "d1_direct_scores.csv",
}
META_COLS = {"participant_id", "data_sufficiency", "d1_confidence"}

# 配置矩阵（方案 §7.1）。父配置用于配对比较时的对照。
CONFIGS = {
    "E0": {"desc": "D1 raw（无训练对照）", "calib": None, "resid": None},
    "E1": {"desc": "D1 affine 校准", "calib": "affine", "resid": None,
           "parent": "E0"},
    "E1b": {"desc": "D1 isotonic 校准", "calib": "isotonic", "resid": None,
            "parent": "E0"},
    "E2": {"desc": "仅结构化观察（无 D1）", "calib": None, "resid": "ridge",
           "lanes": ["obs"], "standalone": True},
    "E3": {"desc": "affine D1 + obs 残差", "calib": "affine",
           "resid": "ridge", "lanes": ["obs"], "parent": "E1"},
    "E4": {"desc": "E3 + 转写时序", "calib": "affine", "resid": "ridge",
           "lanes": ["obs", "tt"], "parent": "E3"},
    "E5": {"desc": "E3 + 声学", "calib": "affine", "resid": "ridge",
           "lanes": ["obs", "ac"], "parent": "E3"},
    "E6": {"desc": "E3 + 视觉", "calib": "affine", "resid": "ridge",
           "lanes": ["obs", "vi"], "parent": "E3"},
    "E7": {"desc": "E3 + 声学 + 视觉", "calib": "affine", "resid": "ridge",
           "lanes": ["obs", "ac", "vi"], "parent": "E3"},
}


# ---------- 校准器 ----------

class AffineCalibrator:
    """base = clip(a + b * d1_raw, 0, 24)。只有两个参数，142 样本下最稳。"""

    name = "affine"

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        # polyfit 而非 LinearRegression：省一个 sklearn 依赖，结果等价
        self.b, self.a = np.polyfit(x, y, 1)
        return self

    def predict(self, x):
        return np.clip(self.a + self.b * np.asarray(x, float),
                       PHQ8_MIN, PHQ8_MAX)

    def params(self):
        return {"a": float(self.a), "b": float(self.b)}


class IsotonicCalibrator:
    """保序回归。不假设线性，但小样本下折内波动大，仅作对照（方案 §2.2）。"""

    name = "isotonic"

    def fit(self, x, y):
        from sklearn.isotonic import IsotonicRegression
        self.m = IsotonicRegression(out_of_bounds="clip",
                                    y_min=PHQ8_MIN, y_max=PHQ8_MAX)
        self.m.fit(np.asarray(x, float), np.asarray(y, float))
        return self

    def predict(self, x):
        return np.clip(self.m.predict(np.asarray(x, float)),
                       PHQ8_MIN, PHQ8_MAX)

    def params(self):
        return {"n_thresholds": int(len(self.m.X_thresholds_))}


class IdentityCalibrator:
    """E0/E2 用：不校准，直接透传或返回零基线。"""

    name = "identity"

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.clip(np.asarray(x, float), PHQ8_MIN, PHQ8_MAX)

    def params(self):
        return {}


def make_calibrator(kind: str | None):
    if kind == "affine":
        return AffineCalibrator()
    if kind == "isotonic":
        return IsotonicCalibrator()
    return IdentityCalibrator()


def make_residual_model(kind: str):
    """残差模型。Ridge 优先：正则化线性在 142 样本 × 50 特征下比 GBDT 稳。

    正则强度在折内自选，不写死。51 个结构化特征配 ~140 个训练样本，
    alpha=1.0 明显偏弱——残差目标 y-base 本身方差大，欠正则会把噪声
    学进去，表现为 RMSE 反而比不做残差修正更差。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    inner = (ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=20, cv=5,
                          max_iter=5000, random_state=C.RANDOM_SEED)
             if kind == "elasticnet"
             else RidgeCV(alphas=np.logspace(-1, 3.5, 30)))
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", inner),
    ])


# ---------- 数据装载 ----------

LANE_OVERRIDE: dict[str, str] = {}


def load_lane(lane: str) -> pd.DataFrame | None:
    # Skill 换版会另存侧表（见 run_text_skill.py 的 schema 隔离），
    # 用覆盖表指过去即可对比两版，不必改动这里的默认路径。
    path = C.FEATURE_DIR / LANE_OVERRIDE.get(lane, LANE_FILES[lane])
    if not path.exists():
        print(f"  ✗ {lane}: 缺少 {path.name}")
        return None
    if lane in LANE_OVERRIDE:
        print(f"  · {lane}: 使用 {path.name}")
    df = pd.read_csv(path)
    return df.rename(columns={c: f"{lane}__{c}" for c in df.columns
                              if c != "participant_id"})


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        base = c.split("__", 1)[-1]
        if c == "participant_id" or base in META_COLS or c.endswith("split"):
            continue
        if c in (C.COL_PHQ8_SCORE, C.COL_PHQ8_BINARY):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_frame(lanes: list[str], need_d1: bool) -> pd.DataFrame | None:
    """按 participant_id 内连接指定的路 + d1 + 标签。"""
    want = list(dict.fromkeys((["d1"] if need_d1 else []) + lanes))
    merged = None
    for lane in want:
        df = load_lane(lane)
        if df is None:
            return None
        merged = df if merged is None else merged.merge(
            df, on="participant_id", how="inner")

    labels = E.load_labels()
    if labels.empty:
        print("  缺少标签文件。")
        return None
    df = merged.merge(labels[[C.COL_PARTICIPANT_ID, C.COL_PHQ8_SCORE,
                              C.COL_PHQ8_BINARY, "split"]],
                      left_on="participant_id",
                      right_on=C.COL_PARTICIPANT_ID, how="inner")
    return df.drop(columns=[C.COL_PARTICIPANT_ID])


# ---------- 单折拟合 ----------

# 数据不足时残差修正的信任折扣（方案 §4.2）。证据稀薄的样本，结构化特征
# 大多是 0，残差模型会把它们当"无症状"推向低分；缩小修正幅度让预测退回
# 校准后的 base，而不是给出一个看似精确的数字。
QUALITY_WEIGHT = {"ok": 1.0, "thin": 0.6, "insufficient": 0.25}


def quality_weights(df: pd.DataFrame, idx=None) -> np.ndarray:
    col = next((c for c in df.columns
                if c.split("__", 1)[-1] == "data_sufficiency"), None)
    if col is None:
        return np.ones(len(df) if idx is None else len(idx))
    s = df[col] if idx is None else df[col].iloc[idx]
    return s.map(lambda v: QUALITY_WEIGHT.get(str(v), 1.0)).to_numpy(float)


def fit_shrinkage(kind: str, X_tr, target) -> float:
    """折内估计残差收缩系数（方案 §5.1 的 alpha）。

    残差模型在自己的训练数据上必然比在新样本上准，直接用 base+resid 等于
    默认它样本外同样可靠。这里用折内 CV 拿到样本外残差预测 r_oof，再解
    最小二乘 min_s ||target - s*r_oof||²，得到闭式解。夹到 [0,1]：只允许
    收缩，不允许放大一个噪声估计。
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    n = len(target)
    if n < 20:
        return 1.0
    r_oof = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    for i_tr, i_te in kf.split(X_tr):
        m = clone(make_residual_model(kind))
        m.fit(X_tr[i_tr], target[i_tr])
        r_oof[i_te] = m.predict(X_tr[i_te])
    denom = float(r_oof @ r_oof)
    if denom < 1e-9:
        return 0.0
    return float(np.clip((r_oof @ target) / denom, 0.0, 1.0))


def fit_fold(cfg: dict, d1_tr, X_tr, y_tr):
    """在一个训练折内拟合校准器与残差模型。返回可用于预测的闭包。

    严格只看 *_tr。这是整个脚本唯一允许接触真值的地方。
    """
    calib = make_calibrator(cfg.get("calib"))
    standalone = cfg.get("standalone", False)

    if standalone:
        base_tr = np.zeros(len(y_tr))          # E2：不用 D1，base 恒为 0
    else:
        calib.fit(d1_tr, y_tr)
        base_tr = calib.predict(d1_tr)

    resid_model, shrink = None, 1.0
    if cfg.get("resid") and X_tr is not None and X_tr.shape[1] > 0:
        target = y_tr - base_tr
        resid_model = make_residual_model(cfg["resid"])
        resid_model.fit(X_tr, target)
        shrink = fit_shrinkage(cfg["resid"], X_tr, target)

    def predict(d1_te, X_te, w_te=None):
        base = (np.zeros(len(d1_te)) if standalone
                else calib.predict(d1_te))
        resid = (resid_model.predict(X_te) if resid_model is not None
                 else np.zeros(len(base)))
        resid = resid * shrink
        if w_te is not None:
            resid = resid * w_te
        return np.clip(base + resid, PHQ8_MIN, PHQ8_MAX), base, resid

    predict.shrink = shrink
    return predict, calib


def fit_classifier(base_tr, X_tr, y_bin_tr):
    """独立分类头（方案 §P6）：直接拟合二分类标签，不复用回归分数卡阈值。

    回归目标是压低整个 0-24 区间的平方误差，而筛查只在乎 10 分附近的判别面，
    两者最优解不同。这里用 Logistic + Platt 概率校准，输入是校准后的 base、
    结构化特征和质量标记。Platt 的 sigmoid 只有两个参数，142 人下比 isotonic 稳。

    正类可能极少（LOSO 折内偶尔单类别），此时返回常数概率而不是崩掉。
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [base_tr.reshape(-1, 1)]
    if X_tr is not None and X_tr.shape[1] > 0:
        feats.append(X_tr)
    F_tr = np.hstack(feats)

    pos = int(np.sum(y_bin_tr == 1))
    neg = int(np.sum(y_bin_tr == 0))
    if pos < 5 or neg < 5:
        rate = float(np.mean(y_bin_tr)) if len(y_bin_tr) else 0.0
        return lambda b, X: np.full(len(b), rate)

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(C=0.1, max_iter=2000,
                                 class_weight="balanced",
                                 random_state=C.RANDOM_SEED))])
    # cv 折数受少数类样本数限制，否则 CalibratedClassifierCV 会抛错
    n_cv = int(min(5, pos, neg))
    if n_cv >= 3:
        model = CalibratedClassifierCV(pipe, method="sigmoid", cv=n_cv)
    else:
        model = pipe
    model.fit(F_tr, y_bin_tr)

    def proba(base_te, X_te):
        f = [np.asarray(base_te).reshape(-1, 1)]
        if X_te is not None and X_te.shape[1] > 0:
            f.append(X_te)
        return model.predict_proba(np.hstack(f))[:, 1]

    return proba


def pick_proba_threshold(y_bin_tr, p_tr) -> float:
    """在概率上选判定阈值。约束同 pick_threshold，候选网格改为 [0,1]。"""
    best, best_ba = None, -1.0
    for thr in np.arange(0.05, 0.96, 0.025):
        m = E.classification_metrics(y_bin_tr, p_tr, threshold=float(thr),
                                     with_ci=False)
        if m["recall"] is None or m["precision"] is None:
            continue
        if m["recall"] >= 0.60 and m["precision"] >= 0.70:
            if m["balanced_accuracy"] > best_ba:
                best, best_ba = float(thr), m["balanced_accuracy"]
    return best if best is not None else 0.5


def pick_threshold(y_bin_tr, score_tr) -> float:
    """在训练折内选分类阈值（方案 §6.2）。

    先约束 Recall>=0.60 与 Precision>=0.70，再在满足者中取 Balanced Accuracy
    最高。都不满足时退回量表阈值 10 并标记未达门槛——不偷偷放宽条件。
    """
    cands = np.arange(4.0, 18.05, 0.5)
    best, best_ba = None, -1.0
    for thr in cands:
        m = E.classification_metrics(y_bin_tr, score_tr, threshold=thr,
                                     with_ci=False)
        # 分母为 0 时指标是 None（如阈值过高导致无预测阳性），跳过该候选
        if m["recall"] is None or m["precision"] is None:
            continue
        if m["recall"] >= 0.60 and m["precision"] >= 0.70:
            if m["balanced_accuracy"] > best_ba:
                best, best_ba = float(thr), m["balanced_accuracy"]
    return best if best is not None else float(C.PHQ8_BINARY_THRESHOLD)


# ---------- 评测协议 ----------

def run_official(cfg: dict, df: pd.DataFrame, X_cols: list[str]) -> dict | None:
    """train 拟合 → dev 评测。与文献及旧 baseline 可比。"""
    tr = (df["split"] == "train").to_numpy()
    dv = (df["split"] == "dev").to_numpy()
    if tr.sum() < 20 or dv.sum() < 10:
        return None

    y = df[C.COL_PHQ8_SCORE].to_numpy(float)
    d1 = df["d1__d1_phq8_estimate"].to_numpy(float)
    X = df[X_cols].to_numpy(float) if X_cols else None

    w = quality_weights(df)
    predict, calib = fit_fold(cfg, d1[tr], X[tr] if X is not None else None,
                              y[tr])
    pred_tr, _, _ = predict(d1[tr], X[tr] if X is not None else None, w[tr])
    pred_dv, base_dv, resid_dv = predict(d1[dv],
                                        X[dv] if X is not None else None,
                                        w[dv])

    ids = df["participant_id"].astype(int).to_numpy()
    yb_tr = E.binary_labels(y[tr], "official", ids[tr])
    yb_dv = E.binary_labels(y[dv], "official", ids[dv])
    thr = pick_threshold(yb_tr, pred_tr)

    res = E.evaluate(y[dv], pred_dv, f"{cfg['_id']} · 官方 dev",
                     ids=ids[dv], label_policy="official")
    res.update(E.classification_metrics(yb_dv, pred_dv, threshold=thr))

    # 独立分类头（§P6）：与"回归分数卡阈值"并列报告，便于判断哪条更该进最终系统
    _, base_tr_only, _ = predict(d1[tr], X[tr] if X is not None else None, w[tr])
    clf = fit_classifier(base_tr_only, X[tr] if X is not None else None, yb_tr)
    p_tr = clf(base_tr_only, X[tr] if X is not None else None)
    p_dv = clf(base_dv, X[dv] if X is not None else None)
    p_thr = pick_proba_threshold(yb_tr, p_tr)
    res["clf"] = E.classification_metrics(yb_dv, p_dv, threshold=p_thr)
    res["clf"]["threshold_kind"] = "probability"
    res["clf_per_subject"] = {str(p): float(v) for p, v in zip(ids[dv], p_dv)}
    res["calibration_params"] = calib.params()
    res["per_subject"] = {str(p): float(v) for p, v in zip(ids[dv], pred_dv)}
    res["base_mean"] = float(np.mean(base_dv))
    res["residual_abs_mean"] = float(np.mean(np.abs(resid_dv)))
    return res


def run_loso(cfg: dict, df: pd.DataFrame, X_cols: list[str]) -> dict:
    """留一受试者交叉验证。每折内重新拟合校准器、残差模型和阈值。"""
    from sklearn.model_selection import LeaveOneOut

    y = df[C.COL_PHQ8_SCORE].to_numpy(float)
    d1 = df["d1__d1_phq8_estimate"].to_numpy(float)
    X = df[X_cols].to_numpy(float) if X_cols else None
    ids = df["participant_id"].astype(int).to_numpy()
    y_bin_all = E.binary_labels(y, "official", ids)

    preds = np.zeros(len(y))
    bases = np.zeros(len(y))
    resids = np.zeros(len(y))
    probas = np.zeros(len(y))
    thrs = []
    p_thrs = []
    fold_params = []
    shrinks = []

    w = quality_weights(df)
    for tr_i, te_i in LeaveOneOut().split(y):
        Xtr = X[tr_i] if X is not None else None
        Xte = X[te_i] if X is not None else None
        predict, calib = fit_fold(cfg, d1[tr_i], Xtr, y[tr_i])
        p, b, r = predict(d1[te_i], Xte, w[te_i])
        preds[te_i], bases[te_i], resids[te_i] = p, b, r
        pred_tr, base_tr, _ = predict(d1[tr_i], Xtr, w[tr_i])
        thrs.append(pick_threshold(y_bin_all[tr_i], pred_tr))
        fold_params.append(calib.params())
        shrinks.append(predict.shrink)

        # 分类头同样每折重拟合：概率校准与阈值都只看训练折
        clf = fit_classifier(base_tr, Xtr, y_bin_all[tr_i])
        probas[te_i] = clf(b, Xte)
        p_thrs.append(pick_proba_threshold(y_bin_all[tr_i], clf(base_tr, Xtr)))

    # 每折阈值不同，报告中位数：外层折的阈值不能互相污染，但汇总需要一个代表值
    thr = float(np.median(thrs))
    res = E.evaluate(y, preds, f"{cfg['_id']} · LOSO train+dev",
                     ids=ids, label_policy="official")
    res.update(E.classification_metrics(y_bin_all, preds, threshold=thr))
    res["threshold_folds"] = {"median": thr, "min": float(np.min(thrs)),
                              "max": float(np.max(thrs))}
    res["per_subject"] = {str(p): float(v) for p, v in zip(ids, preds)}
    res["base_mean"] = float(np.mean(bases))
    res["residual_abs_mean"] = float(np.mean(np.abs(resids)))

    p_thr = float(np.median(p_thrs))
    res["clf"] = E.classification_metrics(y_bin_all, probas, threshold=p_thr)
    res["clf"]["threshold_kind"] = "probability"
    res["clf"]["threshold_folds"] = {"median": p_thr,
                                     "min": float(np.min(p_thrs)),
                                     "max": float(np.max(p_thrs))}
    res["clf_per_subject"] = {str(p): float(v) for p, v in zip(ids, probas)}
    if any(s != 1.0 for s in shrinks):
        res["shrinkage_folds"] = {"mean": float(np.mean(shrinks)),
                                  "std": float(np.std(shrinks))}
    if fold_params and fold_params[0]:
        keys = fold_params[0].keys()
        res["calibration_params_folds"] = {
            k: {"mean": float(np.mean([f[k] for f in fold_params])),
                "std": float(np.std([f[k] for f in fold_params]))}
            for k in keys}
    return res


def paired_diff(a: dict, b: dict, n_boot: int | None = None) -> dict | None:
    """同一批受试者上 B-A 的差值 bootstrap CI（方案 §4.3 第 4 条）。"""
    from scipy.stats import spearmanr

    if not (a.get("per_subject") and b.get("per_subject")):
        return None
    labels = E.load_labels()
    truth = dict(zip(labels[C.COL_PARTICIPANT_ID].astype(str),
                     labels[C.COL_PHQ8_SCORE].astype(float)))
    ids = sorted(set(a["per_subject"]) & set(b["per_subject"]) & set(truth))
    if len(ids) < 10:
        return None

    y = np.array([truth[i] for i in ids])
    pa = np.array([a["per_subject"][i] for i in ids])
    pb = np.array([b["per_subject"][i] for i in ids])
    rng = np.random.default_rng(C.RANDOM_SEED)
    d_rho, d_rmse = [], []
    for _ in range(n_boot or C.N_BOOTSTRAP):
        idx = rng.integers(0, len(ids), len(ids))
        if len(np.unique(y[idx])) < 2:
            continue
        sa, sb = spearmanr(y[idx], pa[idx])[0], spearmanr(y[idx], pb[idx])[0]
        if not (np.isnan(sa) or np.isnan(sb)):
            d_rho.append(sb - sa)
        d_rmse.append(np.sqrt(np.mean((y[idx] - pb[idx]) ** 2))
                      - np.sqrt(np.mean((y[idx] - pa[idx]) ** 2)))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))
    return {
        "n_common": len(ids),
        "spearman_delta": float(spearmanr(y, pb)[0] - spearmanr(y, pa)[0]),
        "spearman_delta_ci95": ci(d_rho),
        "rmse_delta": rmse(pb) - rmse(pa),
        "rmse_delta_ci95": ci(d_rmse),
    }


# ---------- 主流程 ----------

def run_config(cid: str, lanes_override: list[str] | None,
               resid_override: str | None) -> dict | None:
    cfg = dict(CONFIGS[cid])
    cfg["_id"] = cid
    lanes = lanes_override if lanes_override is not None else cfg.get("lanes", [])
    if resid_override and cfg.get("resid"):
        cfg["resid"] = resid_override

    print("=" * 62)
    print(f"{cid}: {cfg['desc']}"
          + (f"  lanes={lanes}" if lanes else "")
          + (f"  resid={cfg['resid']}" if cfg.get("resid") else ""))
    print("=" * 62)

    df = build_frame(lanes, need_d1=True)
    if df is None:
        return None

    X_cols = feature_cols(df.drop(columns=["d1__d1_phq8_estimate"])) \
        if cfg.get("resid") else []
    # d1 的估分是 base 的输入，不能同时作为残差模型的特征：残差模型会重新
    # 学一遍 d1→y 的映射，把校准器的作用抵消掉，等于绕过分层结构
    X_cols = [c for c in X_cols if not c.startswith("d1__")]

    n_tr = int((df["split"] == "train").sum())
    n_dv = int((df["split"] == "dev").sum())
    print(f"  样本: {len(df)} (train={n_tr}, dev={n_dv})"
          f"  残差特征: {len(X_cols)}")

    results = []
    r_off = run_official(cfg, df, X_cols)
    if r_off:
        results.append(r_off)
        E.print_results(r_off)
    r_loso = run_loso(cfg, df, X_cols)
    results.append(r_loso)
    E.print_results(r_loso)

    if "calibration_params_folds" in r_loso:
        p = r_loso["calibration_params_folds"]
        desc = "  ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}"
                         for k, v in p.items())
        print(f"  折内校准参数: {desc}")
    if cfg.get("resid"):
        print(f"  残差修正幅度(平均绝对值): {r_loso['residual_abs_mean']:.3f}")
        if "shrinkage_folds" in r_loso:
            s = r_loso["shrinkage_folds"]
            print(f"  残差收缩系数: {s['mean']:.3f}±{s['std']:.3f}"
                  "  (1=全信，0=完全退回校准 base)")

    if "clf" in r_loso:
        num = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
        print("  独立分类头(logistic+Platt) vs 回归分数卡阈值:")
        for name, m in (("分类头", r_loso["clf"]), ("卡阈值", r_loso)):
            print(f"    {name}  Acc={num(m.get('accuracy'))}"
                  f"  BalAcc={num(m.get('balanced_accuracy'))}"
                  f"  P={num(m.get('precision'))}  R={num(m.get('recall'))}"
                  f"  Brier={num(m.get('brier'))}")

    return {"config_id": cid, "desc": cfg["desc"], "lanes": lanes,
            "calibration": cfg.get("calib"), "residual_model": cfg.get("resid"),
            "n_samples": int(len(df)), "n_residual_features": len(X_cols),
            "parent": cfg.get("parent"), "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="E3",
                    help=f"配置编号或 all，可选 {list(CONFIGS)}")
    ap.add_argument("--lanes", help="覆盖该配置的残差特征路，逗号分隔")
    ap.add_argument("--obs-file",
                    help="改用指定的观察表文件名（Skill 换版后的侧表）")
    ap.add_argument("--residual-model", choices=["ridge", "elasticnet"])
    args = ap.parse_args()

    if args.config == "all":
        order = ["E0", "E1", "E1b", "E2", "E3", "E4", "E5", "E6", "E7"]
    else:
        order = [x.strip() for x in args.config.split(",") if x.strip()]
    bad = [x for x in order if x not in CONFIGS]
    if bad:
        print(f"未知配置: {bad}，可选 {list(CONFIGS)}")
        return 1

    lanes_override = ([x.strip() for x in args.lanes.split(",") if x.strip()]
                      if args.lanes else None)
    if lanes_override:
        unknown = [x for x in lanes_override if x not in LANE_FILES]
        if unknown:
            print(f"未知特征路: {unknown}")
            return 1

    if args.obs_file:
        if not (C.FEATURE_DIR / args.obs_file).exists():
            print(f"找不到观察表: {args.obs_file}")
            return 1
        LANE_OVERRIDE["obs"] = args.obs_file

    done = {}
    for cid in order:
        out = run_config(cid, lanes_override, args.residual_model)
        if out:
            done[cid] = out
        print()

    # 配对比较：每个配置对其父配置，用 LOSO 行（评测集相同才可配对）
    comparisons = []
    for cid, out in done.items():
        parent = out.get("parent")
        if not parent or parent not in done:
            continue
        a = next((r for r in done[parent]["results"] if "LOSO" in r["label"]), None)
        b = next((r for r in out["results"] if "LOSO" in r["label"]), None)
        d = paired_diff(a, b) if a and b else None
        if not d:
            continue
        d.update({"baseline": parent, "candidate": cid})
        comparisons.append(d)

    if comparisons:
        print("=" * 62)
        print("配对比较（LOSO，差值 95% CI，同一批受试者）")
        print("=" * 62)
        for d in comparisons:
            print(f"\n  {d['candidate']} vs {d['baseline']}"
                  f"  (n={d['n_common']})")
            for name, key, better_lower in (
                    ("Spearman ρ", "spearman", False),
                    ("RMSE", "rmse", True)):
                lo, hi = d[f"{key}_delta_ci95"]
                delta = d[f"{key}_delta"]
                verdict = ("差异不显著" if lo <= 0 <= hi else
                           (f"{d['candidate']} 显著更好"
                            if (delta < 0) == better_lower
                            else f"{d['baseline']} 显著更好"))
                print(f"    {name:<12} Δ={delta:+.3f}  "
                      f"CI [{lo:+.3f}, {hi:+.3f}]  → {verdict}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = C.RESULT_DIR / f"calib_{'_'.join(order)}_{stamp}.json"
    out_path.write_text(json.dumps({
        "configs": done, "paired_comparisons": comparisons,
        "label_policy": "official", "seed": C.RANDOM_SEED,
        "n_bootstrap": C.N_BOOTSTRAP, "timestamp": stamp,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



