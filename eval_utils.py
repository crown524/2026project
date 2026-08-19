"""共享评测工具：标签加载、split 归属、bootstrap CI、指标计算。

run_a1_baseline.py 与 run_experiment.py 共用，改动只需一处。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


# ---------- 标签 ----------

def load_labels(include_test: bool = False) -> pd.DataFrame:
    """加载官方划分与 PHQ-8 标签。

    默认只加载 train/dev（test 纪律：仅最终评估时显式传 include_test=True，
    方案 §7.3 要求 test 只使用一次并在报告中声明）。
    兼容 train/dev 的 PHQ8_* 与 full_test 的 PHQ_* 列名差异。
    """
    sources = [(C.TRAIN_SPLIT, "train"), (C.DEV_SPLIT, "dev")]
    if include_test:
        sources.append((C.TEST_SPLIT, "test"))

    frames = []
    for path, split_name in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("participant_id", "participantid"):
                rename[c] = C.COL_PARTICIPANT_ID
            elif cl in ("phq8_score", "phq_score", "phq8score"):
                rename[c] = C.COL_PHQ8_SCORE
            elif cl in ("phq8_binary", "phq_binary"):
                rename[c] = C.COL_PHQ8_BINARY
        df = df.rename(columns=rename)
        df["split"] = split_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    labels = pd.concat(frames, ignore_index=True)
    labels[C.COL_PARTICIPANT_ID] = labels[C.COL_PARTICIPANT_ID].astype(int)
    if C.COL_PHQ8_SCORE not in labels.columns:
        raise ValueError(f"标签文件缺少 PHQ-8 分数列: {list(labels.columns)}")
    if C.COL_PHQ8_BINARY not in labels.columns:
        labels[C.COL_PHQ8_BINARY] = (
            labels[C.COL_PHQ8_SCORE] >= C.PHQ8_BINARY_THRESHOLD).astype(int)
    return labels


def split_membership() -> dict:
    """三个 split 的 ID 集合（不读 test 标签）。"""
    ids = {}
    for path, name in ((C.TRAIN_SPLIT, "train"), (C.DEV_SPLIT, "dev"),
                       (C.TEST_SPLIT, "test")):
        if not path.exists():
            ids[name] = set()
            continue
        df = pd.read_csv(path)
        id_col = next((c for c in df.columns
                       if c.strip().lower() in ("participant_id", "participantid")),
                      None)
        ids[name] = set(df[id_col].astype(int)) if id_col else set()
    return ids


# ---------- 标签口径 ----------

def label_audit() -> pd.DataFrame:
    """逐人核对官方 PHQ8_Binary 与 PHQ8_Score>=10 是否一致。

    方案 §4.1 要求：两种口径不一致时不能把 Accuracy 合并成一个数字。
    已知 409 号 score=10 但官方标 0，而其余 8 个 score=10 的样本官方均标 1，
    因此判定为官方录入异常而非另一套阈值口径。
    """
    lab = load_labels()
    if lab.empty:
        return lab
    out = lab[[C.COL_PARTICIPANT_ID, C.COL_PHQ8_SCORE,
               C.COL_PHQ8_BINARY, "split"]].copy()
    out["derived_score_ge_10"] = (
        out[C.COL_PHQ8_SCORE] >= C.PHQ8_BINARY_THRESHOLD).astype(int)
    out["match"] = out[C.COL_PHQ8_BINARY] == out["derived_score_ge_10"]
    return out


def binary_labels(y_score, policy: str = "derived", ids=None):
    """把连续 PHQ-8 真值转成二分类标签。

    policy="official" 时按 participant_id 查官方字段，需要 ids；
    policy="derived" 时按阈值派生。两种口径的结果必须分开报告。
    """
    y_score = np.asarray(y_score, float)
    if policy == "derived" or ids is None:
        return (y_score >= C.PHQ8_BINARY_THRESHOLD).astype(int)

    lab = load_labels()
    official = dict(zip(lab[C.COL_PARTICIPANT_ID].astype(int),
                        lab[C.COL_PHQ8_BINARY].astype(int)))
    return np.array([official.get(int(i), int(s >= C.PHQ8_BINARY_THRESHOLD))
                     for i, s in zip(ids, y_score)], dtype=int)


# ---------- 指标 ----------

def bootstrap_ci(y_true, y_pred, metric_fn, n=None, seed=None):
    import warnings

    n = n or C.N_BOOTSTRAP
    rng = np.random.default_rng(seed or C.RANDOM_SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    # 重采样必然抽出常数子样本，scipy 每次都要警告一遍；这里的处理是
    # 跳过该次重采样（下面的 isfinite 判断），警告本身没有信息量
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                v = metric_fn(y_true[idx], y_pred[idx])
            except Exception:
                continue
            if v is not None and np.isfinite(v):
                vals.append(float(v))
    # 少于 3 个点时重采样只是在复制同一批值，区间宽度没有意义
    if len(vals) < 100 or len(y_true) < 3:
        return None
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def classification_metrics(y_bin, score, threshold=None,
                           with_ci: bool = True) -> dict:
    """由连续分数与阈值导出的全套分类指标（方案 §4.2）。

    average_precision 用连续分数，与阈值无关；其余指标随阈值变化。
    Accuracy 单独看会被类别偏置骗过（正类仅 30%，全判阴性也有 70%），
    所以 Recall/Specificity/Balanced Accuracy 必须同时报告。

    with_ci=False 跳过 AP 的 bootstrap。阈值搜索会在每折内调用本函数数十次，
    而选阈值只用点估计；带 CI 会让 LOSO 的重采样次数乘上折数×候选数。
    """
    from sklearn.metrics import average_precision_score, brier_score_loss

    thr = C.PHQ8_BINARY_THRESHOLD if threshold is None else threshold
    y_bin = np.asarray(y_bin, int)
    pred = (np.asarray(score, float) >= thr).astype(int)

    tp = int(((pred == 1) & (y_bin == 1)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())

    # 分母为 0 的率返回 None，不返回 nan：nan 在 json 里是非法字面量，
    # 且 nan 会静默传播到 balanced_accuracy，读者看不出是哪一项缺失
    recall = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    out = {
        "threshold": float(thr),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": (tp + tn) / len(y_bin) if len(y_bin) else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": recall,
        "specificity": spec,
        "balanced_accuracy": ((recall + spec) / 2
                              if recall is not None and spec is not None
                              else None),
        "positive_rate": float(y_bin.mean()),
        "n_positive": int((y_bin == 1).sum()),
    }
    # 键始终存在，缺失时为 None：调用方用 res["average_precision"] is None
    # 判断"无定义"，不必区分 KeyError 和真实缺失
    out["average_precision"] = None
    out["average_precision_ci95"] = None
    out["brier"] = None
    if len(np.unique(y_bin)) > 1:
        out["average_precision"] = float(average_precision_score(y_bin, score))
        if with_ci:
            out["average_precision_ci95"] = bootstrap_ci(
                y_bin, score, average_precision_score)
        # 分数映射到 [0,1] 才能算 Brier；PHQ-8 上界 24 是量表定义，不是数据极值
        p = np.clip(np.asarray(score, float) / 24.0, 0.0, 1.0)
        out["brier"] = float(brier_score_loss(y_bin, p))
    return out


def evaluate(y_true, y_pred, label="", ids=None, label_policy="derived",
             y_binary=None, threshold=None):
    """回归为主（PHQ-8 维度性），分类由阈值导出。

    二分类标签的三种来源，优先级从高到低：
      1. y_binary 显式传入（校准脚本按官方字段预取好，最可靠）
      2. ids + label_policy="official" 按 participant_id 查官方字段
      3. 按阈值从 y_true 派生
    两种口径的 Accuracy 不可混用（方案 §4.1）。
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    y_bin_in = None if y_binary is None else np.asarray(y_binary, int)

    # 缺失对必须成对丢弃：sklearn 的 mean_squared_error 遇 nan 会抛错，
    # 而 spearmanr 会静默返回 nan，两种失败方式都不该传给调用方
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.all():
        y_true, y_pred = y_true[ok], y_pred[ok]
        if y_bin_in is not None:
            y_bin_in = y_bin_in[ok]
        if ids is not None:
            ids = np.asarray(ids)[ok]

    res = {"label": label, "n": int(len(y_true)),
           "label_policy": "explicit" if y_bin_in is not None else label_policy,
           "spearman": None, "pearson": None, "spearman_ci95": None}
    if len(y_true) == 0:
        res["rmse"] = res["mae"] = None
        return res

    res["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    res["mae"] = float(mean_absolute_error(y_true, y_pred))
    res["rmse_ci95"] = bootstrap_ci(
        y_true, y_pred, lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))))
    res["mae_ci95"] = bootstrap_ci(
        y_true, y_pred, lambda a, b: float(np.mean(np.abs(a - b))))

    # 常数预测的秩相关无定义。留 None，不要填 0——0 会被读成"无相关"
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        rho, p_s = spearmanr(y_true, y_pred)
        r, p_p = pearsonr(y_true, y_pred)
        res["spearman"], res["spearman_p"] = float(rho), float(p_s)
        res["pearson"], res["pearson_p"] = float(r), float(p_p)
        res["spearman_ci95"] = bootstrap_ci(
            y_true, y_pred, lambda a, b: spearmanr(a, b)[0])

    y_bin = (y_bin_in if y_bin_in is not None
             else binary_labels(y_true, label_policy, ids))
    # 单类别时 AP/Brier 无定义，但 Accuracy 仍有意义，所以照样进
    res.update(classification_metrics(y_bin, y_pred, threshold))
    return res


def _n(v, fmt="{:.3f}", dash="  n/a") -> str:
    """None 打成 n/a 而不是崩在格式化上。缺失是常态，不是异常。"""
    return dash if v is None else fmt.format(v)


def _ci(v) -> str:
    return "[n/a]" if not v else f"[{v[0]:.3f}, {v[1]:.3f}]"


def print_results(res: dict) -> None:
    print(f"\n--- {res['label']} (n={res['n']}) ---")
    print(f"  RMSE : {_n(res.get('rmse'))} {_ci(res.get('rmse_ci95'))}"
          f"   MAE : {_n(res.get('mae'))}")
    if res.get("spearman") is not None:
        print(f"  Spearman : {res['spearman']:.3f}  "
              f"95%CI {_ci(res.get('spearman_ci95'))}  "
              f"p={res['spearman_p']:.4f}")
    if "accuracy" in res:
        print(f"  分类 @{res['threshold']:.0f}  "
              f"({res['label_policy']} 口径, 正类占比 {res['positive_rate']:.2f})")
        print(f"    Acc {_n(res['accuracy'])}   "
              f"BalAcc {_n(res['balanced_accuracy'])}   "
              f"Prec {_n(res['precision'])}   Rec {_n(res['recall'])}   "
              f"Spec {_n(res['specificity'])}")
        cm = res["confusion"]
        extra = ""
        if res.get("brier") is not None:
            extra = f"   Brier {res['brier']:.3f}"
        if res.get("average_precision") is not None:
            extra += (f"   AP {res['average_precision']:.3f} "
                      f"{_ci(res.get('average_precision_ci95'))}")
        print(f"    TP {cm['tp']} TN {cm['tn']} FP {cm['fp']} FN {cm['fn']}{extra}")
