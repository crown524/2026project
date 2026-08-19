"""Stacking 集成实验：Ridge + SVR + GBDT/LGBM → meta-Ridge。

训练协议：
  - 超参数搜索：train-only LOSO（n=107），按 Spearman ρ 选优，dev 全程隔离。
  - Stacking OOF：在 train（n=107）上 LOSO 生成三路基模型预测，训练 meta-Ridge。
  - Dev 评估：基模型在整个 train 集上全量训练，预测 dev，经 meta-Ridge 得到最终值。
  - LOSO train+dev 评估：固定 meta 权重（来自 train OOF），基模型在 140/142 上训练，
    预测留出的一个人，meta 应用得分。此为可泛化性估计（近似，已在结果说明中标注）。

基模型：
  Ridge (StandardScaler + SimpleImputer)
  SVR   (StandardScaler + SimpleImputer, rbf 核)
  LGBM  (SimpleImputer；若未安装则退回 sklearn GradientBoostingRegressor)

用法：
    python run_stacking.py                  # tt+obs，完整搜索
    python run_stacking.py --lanes obs      # obs 单路
    python run_stacking.py --skip-search    # 跳过搜索，使用内置默认值（验证流水线用）
    python run_stacking.py --meta-alpha 0.1 # 调整 meta-Ridge 正则系数
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    _LGBM = False

import config as C
import eval_utils as E

# ──────────────────────────────────────────────────────────────
# 特征加载（镜像 run_experiment.py）
# ──────────────────────────────────────────────────────────────

LANE_FILES = {
    "tt":  ("a1_transcript_timing.csv",  "tt"),
    "obs": ("a2_text_observations.csv",  "obs"),
    "ac":  ("a3_acoustic.csv",           "ac"),
    "vi":  ("a5_visual.csv",             "vi"),
}
META_COLS = {"participant_id", "data_sufficiency", "d1_confidence"}


def _load_lanes(lanes: list[str]) -> pd.DataFrame | None:
    merged = None
    for lane in lanes:
        fname, prefix = LANE_FILES[lane]
        path = C.FEATURE_DIR / fname
        if not path.exists():
            print(f"  ✗ {lane}: 缺少 {fname}（先运行对应抽取脚本）")
            return None
        df = pd.read_csv(path)
        df = df.rename(columns={c: f"{prefix}__{c}" for c in df.columns
                                 if c != "participant_id"})
        merged = df if merged is None else merged.merge(df, on="participant_id", how="inner")
    return merged


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != "participant_id"
            and c.split("__", 1)[-1] not in META_COLS
            and not c.endswith("split")
            and pd.api.types.is_numeric_dtype(df[c])]


# ──────────────────────────────────────────────────────────────
# 基模型工厂
# ──────────────────────────────────────────────────────────────

def _ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("m",      Ridge(alpha=alpha, random_state=C.RANDOM_SEED)),
    ])


def _svr_pipeline(C: float = 1.0, epsilon: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("m",      SVR(kernel="rbf", C=C, epsilon=epsilon)),
    ])


def _gbdt_pipeline(n_estimators: int = 100,
                   max_depth: int = 3,
                   lr: float = 0.05) -> Pipeline:
    if _LGBM:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("m", lgb.LGBMRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=lr, min_child_samples=5,
                random_state=C.RANDOM_SEED, verbose=-1)),
        ])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("m", GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, min_samples_leaf=5, max_features="sqrt",
            random_state=C.RANDOM_SEED)),
    ])


# ──────────────────────────────────────────────────────────────
# 搜索空间（train-only LOSO 优化，dev 全程隔离）
# ──────────────────────────────────────────────────────────────

# Ridge：alpha 是唯一超参；SVR：C 和 epsilon 组合；GBDT：参数从 tune_gbdt.py 已知最优
_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
_SVR_GRID = [
    {"C": 0.5,  "epsilon": 0.5},
    {"C": 1.0,  "epsilon": 0.5},
    {"C": 1.0,  "epsilon": 1.0},
    {"C": 5.0,  "epsilon": 0.5},
    {"C": 10.0, "epsilon": 1.0},
    {"C": 10.0, "epsilon": 2.0},
]
# GBDT 超参直接采用 tune_gbdt.py 的 tt+obs 最优结果（折中组合）
_GBDT_DEFAULT = {"n_estimators": 100, "max_depth": 3, "lr": 0.05}

# meta-Ridge 正则候选（在 train OOF 3×1 矩阵上搜索）
_META_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]


def _loso_spearman(X: np.ndarray, y: np.ndarray,
                   make_fn) -> float:
    """在给定数据上做 LOO，返回 Spearman ρ 点估计。"""
    preds = np.zeros(len(y))
    for tr_i, te_i in LeaveOneOut().split(X):
        preds[te_i] = make_fn().fit(X[tr_i], y[tr_i]).predict(X[te_i])
    rho, _ = spearmanr(y, preds)
    return float(rho)


def search_base_params(X_tr: np.ndarray, y_tr: np.ndarray,
                       verbose: bool = True) -> dict:
    """在 train-only LOSO 上搜索三路基模型超参，返回最优配置字典。"""
    best: dict = {}

    # Ridge
    best_rho, best_alpha = -999.0, _RIDGE_ALPHAS[0]
    for a in _RIDGE_ALPHAS:
        rho = _loso_spearman(X_tr, y_tr, lambda a=a: _ridge_pipeline(a))
        if verbose:
            print(f"    Ridge alpha={a:<8} LOSO ρ={rho:.4f}")
        if rho > best_rho:
            best_rho, best_alpha = rho, a
    best["ridge_alpha"] = best_alpha
    if verbose:
        print(f"  → Ridge 最优 alpha={best_alpha}  ρ={best_rho:.4f}")

    # SVR
    best_rho, best_svr = -999.0, _SVR_GRID[0]
    for kw in _SVR_GRID:
        rho = _loso_spearman(X_tr, y_tr,
                             lambda kw=kw: _svr_pipeline(**kw))
        if verbose:
            print(f"    SVR C={kw['C']:<5} eps={kw['epsilon']:<4} LOSO ρ={rho:.4f}")
        if rho > best_rho:
            best_rho, best_svr = rho, kw
    best["svr_C"]       = best_svr["C"]
    best["svr_epsilon"] = best_svr["epsilon"]
    if verbose:
        print(f"  → SVR 最优 C={best_svr['C']} eps={best_svr['epsilon']}  ρ={best_rho:.4f}")

    # GBDT（直接使用已知最优，不再重复 tune_gbdt 的工作）
    best.update(_GBDT_DEFAULT)
    if verbose:
        gbdt_rho = _loso_spearman(
            X_tr, y_tr,
            lambda: _gbdt_pipeline(**_GBDT_DEFAULT))
        print(f"  → GBDT 固定参数  ρ={gbdt_rho:.4f}  "
              f"({'LightGBM' if _LGBM else 'sklearn GBDT'})")

    return best


# ──────────────────────────────────────────────────────────────
# Stacking：OOF 生成 + meta 训练
# ──────────────────────────────────────────────────────────────

def make_base_models(params: dict) -> list[Pipeline]:
    return [
        _ridge_pipeline(params["ridge_alpha"]),
        _svr_pipeline(params["svr_C"], params["svr_epsilon"]),
        _gbdt_pipeline(params["n_estimators"], params["max_depth"], params["lr"]),
    ]


def build_oof_matrix(X: np.ndarray, y: np.ndarray,
                     params: dict) -> np.ndarray:
    """在给定数据上 LOO，返回 shape=(n, 3) 的 OOF 预测矩阵。"""
    n = len(y)
    oof = np.zeros((n, 3))
    for tr_i, te_i in LeaveOneOut().split(X):
        for k, model in enumerate(make_base_models(params)):
            oof[te_i, k] = model.fit(X[tr_i], y[tr_i]).predict(X[te_i])
    return oof


def fit_meta(oof: np.ndarray, y: np.ndarray,
             alpha: float = 1.0) -> Pipeline:
    meta = Pipeline([
        ("scale", StandardScaler()),
        ("m",     Ridge(alpha=alpha, random_state=C.RANDOM_SEED)),
    ])
    return meta.fit(oof, y)


def search_meta_alpha(oof: np.ndarray, y: np.ndarray,
                      verbose: bool = True) -> float:
    """在同一份 OOF 矩阵上做 LOO 搜索 meta-Ridge alpha。

    注意：这里 OOF 矩阵本身已经是 LOO 预测，所以只要 meta 不做 LOO
    就会用到自身标签，但样本量只有 107 而特征只有 3，
    直接在 OOF 上用 LOO 估计 meta alpha 是合理且成本极低的。
    """
    best_rho, best_a = -999.0, _META_ALPHAS[0]
    for a in _META_ALPHAS:
        preds = np.zeros(len(y))
        for tr_i, te_i in LeaveOneOut().split(oof):
            m = fit_meta(oof[tr_i], y[tr_i], a)
            preds[te_i] = m.predict(oof[te_i])
        rho, _ = spearmanr(y, preds)
        if verbose:
            print(f"    meta-Ridge alpha={a:<6} OOF ρ={rho:.4f}")
        if rho > best_rho:
            best_rho, best_a = rho, a
    if verbose:
        print(f"  → meta-Ridge 最优 alpha={best_a}  ρ={best_rho:.4f}")
    return best_a


# ──────────────────────────────────────────────────────────────
# 官方 dev 评估
# ──────────────────────────────────────────────────────────────

def evaluate_dev(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_dv: np.ndarray, y_dv: np.ndarray,
                 params: dict, meta_alpha: float,
                 pid_dv: list) -> dict:
    """在整个 train 上全量训练基模型，预测 dev，经 meta-Ridge 得最终分数。

    meta-Ridge 是在 train 的 OOF 矩阵上拟合的，不接触 dev 标签。
    """
    # 1. 生成 train OOF 矩阵，训练 meta
    oof = build_oof_matrix(X_tr, y_tr, params)
    meta = fit_meta(oof, y_tr, meta_alpha)

    # 2. 基模型全量训练 → dev 预测矩阵
    base_dv = np.column_stack([
        m.fit(X_tr, y_tr).predict(X_dv)
        for m in make_base_models(params)
    ])

    # 3. meta 推断
    yhat_dv = meta.predict(base_dv)

    r = E.evaluate(y_dv, yhat_dv, "Stacking · 官方 dev")
    r["per_subject"] = {str(p): float(v) for p, v in zip(pid_dv, yhat_dv)}
    return r


# ──────────────────────────────────────────────────────────────
# LOSO train+dev 评估
# ──────────────────────────────────────────────────────────────

def evaluate_loso(X_td: np.ndarray, y_td: np.ndarray,
                  params: dict, meta_alpha: float,
                  pid_td: list) -> dict:
    """近似 LOSO：每折保留 1 人，142-1 人中用 LOSO 生成 OOF 并训练 meta，
    再用全量 141 人的基模型预测留出人。

    此处 meta_alpha 已在 train-only 阶段固定，不在每折内重新搜索，
    属于轻微数据泄漏（meta 超参来自 train 集），可接受；结果说明中已标注。
    """
    preds = np.zeros(len(y_td))
    n = len(y_td)
    for outer_test_idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[outer_test_idx] = False
        X_inner, y_inner = X_td[mask], y_td[mask]

        # 内层 OOF（141 人 LOO）→ 训练 meta
        oof_inner = build_oof_matrix(X_inner, y_inner, params)
        meta = fit_meta(oof_inner, y_inner, meta_alpha)

        # 基模型全量训练 → 预测留出 1 人
        base_pred = np.column_stack([
            m.fit(X_inner, y_inner).predict(X_td[[outer_test_idx]])
            for m in make_base_models(params)
        ])
        preds[outer_test_idx] = meta.predict(base_pred)[0]

    r = E.evaluate(y_td, preds, "Stacking · LOSO train+dev (meta α 来自 train，轻微泄漏)")
    r["per_subject"] = {str(p): float(v) for p, v in zip(pid_td, preds)}
    return r


def evaluate_test(X_td: np.ndarray, y_td: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray,
                  params: dict, meta_alpha: float,
                  pid_te: list) -> dict:
    """train+dev 合并训练基模型和 meta，预测 test 集。"""
    oof_td = build_oof_matrix(X_td, y_td, params)
    meta = fit_meta(oof_td, y_td, meta_alpha)

    base_te = np.column_stack([
        m.fit(X_td, y_td).predict(X_te)
        for m in make_base_models(params)
    ])
    yhat_te = meta.predict(base_te)

    r = E.evaluate(y_te, yhat_te, "Stacking · 官方 test")
    r["per_subject"] = {str(p): float(v) for p, v in zip(pid_te, yhat_te)}
    return r


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

# 默认超参（跳过搜索时使用；已更新为本次 train-only LOSO 搜索结果）
_DEFAULT_PARAMS = {
    "ridge_alpha": 100.0,
    "svr_C": 1.0,
    "svr_epsilon": 0.5,
    **_GBDT_DEFAULT,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stacking 集成：Ridge + SVR + GBDT/LGBM → meta-Ridge")
    ap.add_argument("--lanes", default="tt,obs",
                    help="逗号分隔特征路，默认 tt,obs")
    ap.add_argument("--skip-search", action="store_true",
                    help="跳过超参搜索，使用内置默认值（流水线验证用）")
    ap.add_argument("--meta-alpha", type=float, default=None,
                    help="强制指定 meta-Ridge alpha（跳过 meta alpha 搜索）")
    ap.add_argument("--skip-loso", action="store_true",
                    help="跳过 LOSO train+dev 评估（LOSO 耗时 ~20 min，调试时跳过）")
    ap.add_argument("--test", action="store_true",
                    help="最终评估模式：train+dev 合并训练，在 test 集上评测。")
    args = ap.parse_args()

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    bad = [x for x in lanes if x not in LANE_FILES]
    if bad:
        print(f"未知特征路: {bad}，可选 {list(LANE_FILES)}")
        return 1

    backend = "LightGBM" if _LGBM else "sklearn GBDT"
    print("=" * 62)
    print(f"Stacking 实验: lanes={lanes}  GBDT后端={backend}")
    print(f"  meta-Ridge  |  base: Ridge + SVR + {backend}")
    print("=" * 62)

    # 加载特征
    feat_df = _load_lanes(lanes)
    if feat_df is None:
        return 1

    # 合并标签
    labels = E.load_labels(include_test=args.test)
    if labels.empty:
        print("  缺少标签文件。")
        return 1
    df = feat_df.merge(
        labels[[C.COL_PARTICIPANT_ID, C.COL_PHQ8_SCORE, C.COL_PHQ8_BINARY, "split"]],
        left_on="participant_id", right_on=C.COL_PARTICIPANT_ID, how="inner"
    ).drop(columns=[C.COL_PARTICIPANT_ID])

    X_cols = _feat_cols(df.drop(columns=[C.COL_PHQ8_SCORE, C.COL_PHQ8_BINARY]))
    X = df[X_cols].to_numpy(float)
    y = df[C.COL_PHQ8_SCORE].to_numpy(float)
    pid = df["participant_id"].astype(int).tolist()

    tr = (df["split"] == "train").to_numpy()
    dv = (df["split"] == "dev").to_numpy()
    td = tr | dv

    n_tr, n_dv, n_td = tr.sum(), dv.sum(), td.sum()
    print(f"  特征维度: {len(X_cols)}  "
          f"train={n_tr}, dev={n_dv}, train+dev={n_td}")

    if n_tr < 20 or n_dv < 5:
        print(f"  ⚠ 样本不足，中止。")
        return 1

    X_tr, y_tr = X[tr], y[tr]
    X_dv, y_dv = X[dv], y[dv]
    X_td, y_td = X[td], y[td]
    pid_dv = [p for p, m in zip(pid, dv) if m]
    pid_td = [p for p, m in zip(pid, td) if m]
    if args.test:
        te = (df["split"] == "test").to_numpy()
        X_te, y_te = X[te], y[te]
        pid_te = [p for p, m in zip(pid, te) if m]

    # ── 超参搜索（train-only LOSO）──
    if args.skip_search:
        params = _DEFAULT_PARAMS.copy()
        print("  [跳过搜索] 使用默认参数:", params)
    else:
        print("\n── 基模型超参搜索（train-only LOSO, n=107）──")
        params = search_base_params(X_tr, y_tr, verbose=True)

    # ── meta alpha 搜索（train OOF）──
    print("\n── OOF 矩阵生成中（train LOO, 107 次）──")
    oof_tr = build_oof_matrix(X_tr, y_tr, params)

    if args.meta_alpha is not None:
        meta_alpha = args.meta_alpha
        print(f"  [强制指定] meta-Ridge alpha={meta_alpha}")
    else:
        print("\n── meta-Ridge alpha 搜索（train OOF LOO）──")
        meta_alpha = search_meta_alpha(oof_tr, y_tr, verbose=True)

    # OOF 基模型相关性（诊断信息）
    rho_ridge, _ = spearmanr(y_tr, oof_tr[:, 0])
    rho_svr,   _ = spearmanr(y_tr, oof_tr[:, 1])
    rho_gbdt,  _ = spearmanr(y_tr, oof_tr[:, 2])
    print(f"\n  OOF 基模型单独 ρ:  Ridge={rho_ridge:.4f}  "
          f"SVR={rho_svr:.4f}  {backend}={rho_gbdt:.4f}")

    all_results = []

    if args.test:
        # ── test 最终评估：train+dev 合并训练，在 test 集上评测 ──
        print("\n── 最终 test 评估（train+dev 合并训练，n_train=142，n_test=47）──")
        r_test = evaluate_test(X_td, y_td, X_te, y_te, params, meta_alpha, pid_te)
        E.print_results(r_test)
        all_results.append(r_test)
    else:
        # ── 官方 dev 评估 ──
        print("\n── 官方 dev 评估（n=35）──")
        r_dev = evaluate_dev(X_tr, y_tr, X_dv, y_dv, params, meta_alpha, pid_dv)
        E.print_results(r_dev)
        all_results.append(r_dev)

        # ── LOSO train+dev 评估 ──
        if not args.skip_loso:
            print(f"\n── LOSO train+dev 评估（n={n_td}，每折重建 OOF+meta，耗时较长）──")
            r_loso = evaluate_loso(X_td, y_td, params, meta_alpha, pid_td)
            E.print_results(r_loso)
            all_results.append(r_loso)
        else:
            print("\n  [跳过 LOSO]")

    # ── 落盘 ──
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = C.RESULT_DIR / f"stacking_{'_'.join(lanes)}_{stamp}.json"
    out.write_text(json.dumps({
        "lanes": lanes,
        "backend": backend,
        "params": params,
        "meta_alpha": meta_alpha,
        "n_samples": int(len(df)),
        "results": all_results,
        "timestamp": stamp,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
