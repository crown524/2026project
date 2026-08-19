"""统一消融实验脚本：任选特征路组合，一个命令跑一组实验。

覆盖 v3 方案 §7.1 消融矩阵中除 A4(wav2vec)、E1(跨库) 外的全部组：

    A1  --lanes tt              仅转写时序
    A2  --lanes obs             仅文本 Skill 观察特征
    A3  --lanes ac              仅声学
    A5  --lanes vi              仅视觉
    B1  --lanes tt,obs          时序+文本
    B2  --lanes obs,ac          文本+声学
    B3  --lanes obs,ac,vi       三模态
    B4  --lanes tt,obs,ac,vi    全配置
    D1  --lanes d1              LLM 直接打分（不训模型，直接评它的输出）
    D2  --lanes tt,obs,ac,vi    等价 B4（观察特征 + 小模型 = 分层架构）
    B0  自动附带（预测训练集均值的 dummy 基线）

评测协议（两个都跑）：
    官方划分: train 上训练 → dev 上评测（与文献可比）
    LOSO    : train+dev 合并做留一交叉验证（小样本更稳）

用法：
    python run_experiment.py --lanes tt
    python run_experiment.py --lanes tt,ac,vi --model both
    python run_experiment.py --lanes d1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import eval_utils as E

LANE_FILES = {
    "tt": ("a1_transcript_timing.csv", "tt"),
    "ac": ("a3_acoustic.csv", "ac"),
    "vi": ("a5_visual.csv", "vi"),
    "obs": ("a2_text_observations.csv", "obs"),
    "d1": ("d1_direct_scores.csv", "d1"),
}
META_COLS = {"participant_id", "data_sufficiency", "d1_confidence"}


def load_lane(lane: str) -> pd.DataFrame | None:
    fname, prefix = LANE_FILES[lane]
    path = C.FEATURE_DIR / fname
    if not path.exists():
        print(f"  ✗ {lane}: 缺少 {fname}（先运行对应抽取脚本）")
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={c: f"{prefix}__{c}" for c in df.columns
                            if c not in ("participant_id",)})
    return df


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        base = c.split("__", 1)[-1]
        if c == "participant_id" or base in META_COLS or c.endswith("split"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_model(kind: str):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "gbdt":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("m", GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, max_features="sqrt",
                random_state=C.RANDOM_SEED))])
    if kind == "xgb":
        import xgboost as xgb
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("m", xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_child_weight=3, colsample_bytree=1.0,
                reg_alpha=0.0, reg_lambda=1.0,
                random_state=C.RANDOM_SEED, n_jobs=-1, verbosity=0))])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", Ridge(alpha=1.0, random_state=C.RANDOM_SEED))])


def run_d1_direct(df: pd.DataFrame) -> list[dict]:
    """D1 特殊路径：LLM 的输出即预测，不训练任何模型。"""
    col = "d1__d1_phq8_estimate"
    if col not in df.columns:
        print("  d1 特征文件中缺少 phq8_estimate 列")
        return []
    sub = df.dropna(subset=[col, C.COL_PHQ8_SCORE])

    def per_subject(rows: pd.DataFrame) -> dict[str, float]:
        return {str(int(p)): float(v) for p, v in
                zip(rows["participant_id"], rows[col])}

    results = []
    for split_name in ("train", "dev"):
        m = sub["split"] == split_name
        if m.sum() >= 5:
            r = E.evaluate(sub.loc[m, C.COL_PHQ8_SCORE], sub.loc[m, col],
                           f"D1 LLM直接打分 · 官方 {split_name}")
            # D1 不训练，逐人预测就是 LLM 原始输出；没有它 compare_configs.py
            # 无法把 D1 与 D2 配对比较，而那正是分层架构主张的成立依据
            r["per_subject"] = per_subject(sub.loc[m])
            results.append(r)
    if len(sub) >= 10:
        r = E.evaluate(sub[C.COL_PHQ8_SCORE], sub[col],
                       "D1 LLM直接打分 · train+dev 全体")
        r["per_subject"] = per_subject(sub)
        results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", required=True,
                    help="逗号分隔: tt,ac,vi,obs,d1")
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "ridge", "xgb", "both"])
    ap.add_argument("--test", action="store_true",
                    help="最终评估模式：train+dev 合并训练，在 test 集上评测。"
                         "请遵守「test 只用一次」纪律。")
    args = ap.parse_args()

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    bad = [x for x in lanes if x not in LANE_FILES]
    if bad:
        print(f"未知特征路: {bad}，可选 {list(LANE_FILES)}")
        return 1

    print("=" * 62)
    print(f"实验: lanes={lanes}  model={args.model}"
          + ("  [TEST 最终评估]" if args.test else ""))
    print("=" * 62)

    # 合并特征
    merged = None
    for lane in lanes:
        df = load_lane(lane)
        if df is None:
            return 1
        merged = df if merged is None else merged.merge(
            df, on="participant_id", how="inner")
    print(f"  特征合并: {merged.shape[0]} 会话 × {merged.shape[1]-1} 列")

    # 合并标签
    labels = E.load_labels(include_test=args.test)
    if labels.empty:
        print("  缺少标签文件。")
        return 1
    df = merged.merge(labels[[C.COL_PARTICIPANT_ID, C.COL_PHQ8_SCORE,
                              C.COL_PHQ8_BINARY, "split"]],
                      left_on="participant_id",
                      right_on=C.COL_PARTICIPANT_ID, how="inner")
    df = df.drop(columns=[C.COL_PARTICIPANT_ID])
    n_tr = int((df["split"] == "train").sum())
    n_dv = int((df["split"] == "dev").sum())
    n_te = int((df["split"] == "test").sum())
    print(f"  带标签样本: {len(df)}  (train={n_tr}, dev={n_dv}, test={n_te})")

    all_results = []

    if lanes == ["d1"]:
        all_results += run_d1_direct(df)
        for r in all_results:
            E.print_results(r)
    else:
        if len(df) < 30:
            print(f"\n  ⚠ 带标签样本仅 {len(df)}，不足以训练。"
                  "流水线检查通过，补数据后重跑。")
            return 0

        X_cols = numeric_feature_cols(df.drop(columns=[C.COL_PHQ8_SCORE,
                                                       C.COL_PHQ8_BINARY]))
        X = df[X_cols].to_numpy(float)
        y = df[C.COL_PHQ8_SCORE].to_numpy(float)
        print(f"  特征维度: {len(X_cols)}")

        kinds = ["gbdt", "ridge"] if args.model == "both" else [args.model]

        from sklearn.dummy import DummyRegressor
        from sklearn.model_selection import LeaveOneOut

        tr  = (df["split"] == "train").to_numpy()
        dv  = (df["split"] == "dev").to_numpy()
        te  = (df["split"] == "test").to_numpy()
        # 最终评估：train+dev 合并作为训练集，test 作为评测集
        tr_full = (tr | dv)

        if args.test:
            # B0 基线（train+dev 均值 → test）
            dummy = DummyRegressor(strategy="mean").fit(X[tr_full], y[tr_full])
            r = E.evaluate(y[te], dummy.predict(X[te]), "B0 均值基线 · 官方 test")
            all_results.append(r); E.print_results(r)

            pid = df["participant_id"].astype(int).tolist()
            for kind in kinds:
                tag = f"{'+'.join(lanes)}({kind})"
                m = make_model(kind).fit(X[tr_full], y[tr_full])
                yhat_te = m.predict(X[te])
                r = E.evaluate(y[te], yhat_te, f"{tag} · 官方 test")
                r["per_subject"] = {str(p): float(v) for p, v in zip(
                    np.asarray(pid)[te], yhat_te)}
                all_results.append(r); E.print_results(r)
                if kind == "gbdt":
                    imp = sorted(zip(X_cols, m.named_steps["m"].feature_importances_),
                                 key=lambda t: -t[1])[:12]
                    print("  特征重要性 Top 12:")
                    for name, v in imp:
                        print(f"    {v:6.3f}  {name}")
        else:
            # 原有 dev 评测逻辑
            if tr.sum() >= 20 and dv.sum() >= 10:
                dummy = DummyRegressor(strategy="mean").fit(X[tr], y[tr])
                r = E.evaluate(y[dv], dummy.predict(X[dv]), "B0 均值基线 · 官方 dev")
                all_results.append(r); E.print_results(r)

            b0 = np.zeros(len(y[tr | dv]))
            X_td, y_td = X[tr | dv], y[tr | dv]
            for tr_i, te_i in LeaveOneOut().split(X_td):
                b0[te_i] = y_td[tr_i].mean()
            r = E.evaluate(y_td, b0, "B0 均值基线 · LOSO train+dev")
            for k in ("spearman", "spearman_p", "spearman_ci95", "pearson",
                      "pearson_p", "average_precision", "average_precision_ci95"):
                r.pop(k, None)
            all_results.append(r); E.print_results(r)

            pid = df["participant_id"].astype(int).tolist()
            pid_td = df.loc[tr | dv, "participant_id"].astype(int).tolist()

            for kind in kinds:
                tag = f"{'+'.join(lanes)}({kind})"
                if tr.sum() >= 20 and dv.sum() >= 10:
                    m = make_model(kind).fit(X[tr], y[tr])
                    yhat_dv = m.predict(X[dv])
                    r = E.evaluate(y[dv], yhat_dv, f"{tag} · 官方 dev")
                    r["per_subject"] = {str(p): float(v) for p, v in zip(
                        np.asarray(pid)[dv], yhat_dv)}
                    all_results.append(r); E.print_results(r)
                    if kind == "gbdt":
                        imp = sorted(zip(X_cols, m.named_steps["m"].feature_importances_),
                                     key=lambda t: -t[1])[:12]
                        print("  特征重要性 Top 12:")
                        for name, v in imp:
                            print(f"    {v:6.3f}  {name}")

                preds = np.zeros(len(y_td))
                for tr_i, te_i in LeaveOneOut().split(X_td):
                    preds[te_i] = make_model(kind).fit(X_td[tr_i], y_td[tr_i]).predict(X_td[te_i])
                r = E.evaluate(y_td, preds, f"{tag} · LOSO train+dev")
                r["per_subject"] = {str(p): float(v) for p, v in zip(pid_td, preds)}
                all_results.append(r); E.print_results(r)

    # 落盘
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = C.RESULT_DIR / f"exp_{'_'.join(lanes)}_{stamp}.json"
    out.write_text(json.dumps({
        "lanes": lanes, "model": args.model,
        "n_samples": int(len(df)),
        "results": all_results, "timestamp": stamp,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
