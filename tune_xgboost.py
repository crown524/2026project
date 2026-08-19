"""XGBoost 超参数搜索脚本（LOSO on train only）

搜索范围：n_estimators, min_child_weight, colsample_bytree
固定参数：max_depth=3, learning_rate=0.05, subsample=0.8
评估协议：仅在 train 集（107 人）内做留一交叉验证，dev 集完全隔离。

用法：
    python tune_xgboost.py --lanes obs
    python tune_xgboost.py --lanes tt,obs
    python tune_xgboost.py --lanes tt,obs --top 5
"""
from __future__ import annotations

import argparse
import itertools
import sys
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline

import config as C
import eval_utils as E

LANE_FILES = {
    "tt":  ("a1_transcript_timing.csv",  "tt"),
    "ac":  ("a3_acoustic.csv",           "ac"),
    "vi":  ("a5_visual.csv",             "vi"),
    "obs": ("a2_text_observations.csv",  "obs"),
}

# 搜索空间（36 组，与 tune_gbdt.py 规模一致）
GRID: dict[str, list[Any]] = {
    "n_estimators":     [100, 200, 300],
    "min_child_weight": [1, 3, 5, 10],
    "colsample_bytree": [0.5, 0.7, 1.0],
}

# 固定参数（与调优 GBDT 对齐，便于直接对比）
FIXED = dict(
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=C.RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
)


def load_lane(lane: str) -> pd.DataFrame | None:
    fname, prefix = LANE_FILES[lane]
    path = C.FEATURE_DIR / fname
    if not path.exists():
        print(f"  ✗ {lane}: 缺少 {fname}")
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={c: f"{prefix}__{c}" for c in df.columns
                             if c != "participant_id"})
    return df


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    meta = {"participant_id", "data_sufficiency", "d1_confidence"}
    return [c for c in df.columns
            if c != "participant_id"
            and c.split("__", 1)[-1] not in meta
            and not c.endswith("split")
            and pd.api.types.is_numeric_dtype(df[c])]


def make_model(params: dict) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("m", xgb.XGBRegressor(**FIXED, **params)),
    ])


def loso_score(X: np.ndarray, y: np.ndarray, params: dict) -> dict:
    preds = np.zeros(len(y))
    for tr_i, te_i in LeaveOneOut().split(X):
        preds[te_i] = make_model(params).fit(X[tr_i], y[tr_i]).predict(X[te_i])
    return E.evaluate(y, preds, label="")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", required=True,
                    help="逗号分隔特征路，例如 tt,obs")
    ap.add_argument("--top", type=int, default=5,
                    help="打印最优前 N 个参数组合（默认 5）")
    args = ap.parse_args()

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    bad = [x for x in lanes if x not in LANE_FILES]
    if bad:
        print(f"未知特征路: {bad}"); return 1

    # 合并特征
    merged = None
    for lane in lanes:
        df = load_lane(lane)
        if df is None: return 1
        merged = df if merged is None else merged.merge(
            df, on="participant_id", how="inner")

    # 合并标签，只保留 train
    labels = E.load_labels()
    if labels.empty:
        print("缺少标签文件。"); return 1
    df = merged.merge(
        labels[[C.COL_PARTICIPANT_ID, C.COL_PHQ8_SCORE, "split"]],
        left_on="participant_id", right_on=C.COL_PARTICIPANT_ID, how="inner",
    ).drop(columns=[C.COL_PARTICIPANT_ID])

    train_df = df[df["split"] == "train"].copy()
    n_tr = len(train_df)
    print(f"lanes={lanes}  XGBoost  train 样本: {n_tr}")
    print(f"固定参数: max_depth=3, lr=0.05, subsample=0.8")
    if n_tr < 20:
        print("样本过少，无法搜索。"); return 1

    X_cols = numeric_feature_cols(train_df.drop(columns=[C.COL_PHQ8_SCORE]))
    X = train_df[X_cols].to_numpy(float)
    y = train_df[C.COL_PHQ8_SCORE].to_numpy(float)
    print(f"特征维度: {len(X_cols)}\n")

    # 网格搜索
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"共 {len(combos)} 个参数组合，每个做 LOSO（n={n_tr}）...\n")

    rows = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        r = loso_score(X, y, params)
        rows.append({
            "spearman": r.get("spearman", float("nan")),
            "rmse":     r.get("rmse",     float("nan")),
            "acc":      r.get("accuracy", float("nan")),
            **params,
        })
        print(f"  [{i:2d}/{len(combos)}] "
              f"n_est={params['n_estimators']:3d}  "
              f"min_cw={params['min_child_weight']:2d}  "
              f"colsamp={params['colsample_bytree']:.1f}  "
              f"=> Spearman={r.get('spearman', float('nan')):.3f}  "
              f"RMSE={r.get('rmse', float('nan')):.3f}  "
              f"Acc={r.get('accuracy', float('nan')):.3f}")

    result_df = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    print(f"\n{'='*62}")
    print(f"Top {args.top} 参数组合（按 LOSO Spearman 排序）")
    print(f"{'='*62}")
    for _, row in result_df.head(args.top).iterrows():
        print(f"  Spearman={row['spearman']:.3f}  RMSE={row['rmse']:.3f}  "
              f"Acc={row['acc']:.3f}  |  "
              f"n_estimators={int(row['n_estimators'])}  "
              f"min_child_weight={int(row['min_child_weight'])}  "
              f"colsample_bytree={row['colsample_bytree']:.1f}")

    best = result_df.iloc[0]
    print(f"\n最优参数（写入 run_experiment.py make_model xgb 分支）:")
    print(f"  n_estimators={int(best['n_estimators'])}")
    print(f"  min_child_weight={int(best['min_child_weight'])}")
    print(f"  colsample_bytree={best['colsample_bytree']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
