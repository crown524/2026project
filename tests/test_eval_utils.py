"""eval_utils 的边界行为测试（方案 §P1 放行条件）。

这些用例针对的都是真出过错或极易出错的情形：单类别、常数预测、
PHQ-8=10 边界、缺失值、指标命名。跑：
    python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
import eval_utils as E


# ---------- 回归指标 ----------

def test_perfect_prediction():
    y = np.array([0.0, 5, 10, 15, 20])
    r = E.evaluate(y, y.copy(), "perfect")
    assert r["rmse"] == pytest.approx(0.0)
    assert r["mae"] == pytest.approx(0.0)
    assert r["spearman"] == pytest.approx(1.0)


def test_constant_prediction_has_no_correlation():
    """常数预测的相关系数无定义，必须是 None 而不是 nan 或 0。

    nan 会在 json.dumps 里变成非法的 NaN 字面量，0 会被误读成"无相关"。
    """
    y = np.array([0.0, 3, 7, 12, 18, 22])
    r = E.evaluate(y, np.full(len(y), 8.0), "constant")
    assert r["spearman"] is None
    assert r["pearson"] is None
    assert r["rmse"] > 0


def test_rmse_matches_manual():
    y = np.array([2.0, 4.0])
    p = np.array([4.0, 8.0])
    r = E.evaluate(y, p, "manual")
    assert r["rmse"] == pytest.approx(np.sqrt((4 + 16) / 2))
    assert r["mae"] == pytest.approx(3.0)


def test_nan_pairs_are_dropped():
    y = np.array([5.0, 10.0, np.nan, 15.0])
    p = np.array([6.0, np.nan, 12.0, 14.0])
    r = E.evaluate(y, p, "with-nan")
    assert r["n"] == 2


# ---------- 分类指标 ----------

def test_threshold_boundary_is_inclusive():
    """PHQ-8 == 10 必须算阳性。这是 DAIC 的官方口径，差一个样本就差 2.9%。"""
    y = np.array([9.0, 10.0, 11.0])
    r = E.evaluate(y, np.array([9.0, 10.0, 11.0]), "boundary",
                   y_binary=(y >= 10).astype(int))
    assert r["n_positive"] == 2


def test_all_negative_class():
    """单类别时 AP/PR-AUC 无定义，Accuracy 仍有定义。"""
    y = np.array([1.0, 3, 5, 7])
    r = E.evaluate(y, np.array([2.0, 2, 4, 6]), "all-neg",
                   y_binary=np.zeros(4, dtype=int))
    assert r["average_precision"] is None
    assert r["accuracy"] is not None
    assert r["recall"] is None          # 无阳性 → 召回无定义


def test_all_positive_class():
    y = np.array([12.0, 15, 18, 21])
    r = E.evaluate(y, np.array([13.0, 14, 19, 20]), "all-pos",
                   y_binary=np.ones(4, dtype=int))
    assert r["average_precision"] is None
    assert r["specificity"] is None    # 无阴性 → 特异度无定义


def test_classification_metrics_are_consistent():
    """手算一遍混淆矩阵，确认四个率的分母没搞反。"""
    y_bin = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    score = np.array([20.0, 15, 5, 12, 3, 2, 1, 0])   # 阈值 10：TP=2 FN=1 FP=1 TN=4
    r = E.evaluate(np.where(y_bin == 1, 15.0, 5.0), score, "confusion",
                   y_binary=y_bin, threshold=10.0)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(4 / 5)
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_brier_is_bounded():
    y_bin = np.array([1, 0, 1, 0])
    r = E.evaluate(np.array([15.0, 5, 15, 5]), np.array([20.0, 0, 12, 8]),
                   "brier", y_binary=y_bin)
    assert 0.0 <= r["brier"] <= 1.0


def test_average_precision_is_not_named_precision():
    """方案 §4.2：旧代码把 average_precision_score 叫 pr_auc，读者会当成 Precision。

    构造一个阈值分类差、但排序好的例子：三个阳性分数都在阈值以下，
    所以 Precision 无定义（没有预测阳性），而 AP 很高（排序几乎完美）。
    两者若同名，这种情况会被完全误读。
    """
    y_bin = np.array([1, 1, 1, 0, 0, 0])
    score = np.array([9.0, 8.0, 7.0, 3.0, 2.0, 1.0])   # 阈值 10：无预测阳性
    r = E.evaluate(np.where(y_bin == 1, 15.0, 5.0), score, "naming",
                   y_binary=y_bin, threshold=10.0)
    assert r["precision"] is None                  # 分母 tp+fp = 0
    assert r["average_precision"] == pytest.approx(1.0)   # 排序完美
    assert "pr_auc" not in r                       # 旧的误导性字段名已移除


# ---------- 置信区间 ----------

def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 24, 60)
    p = y + rng.normal(0, 3, 60)
    r = E.evaluate(y, p, "ci")
    lo, hi = r["spearman_ci95"]
    assert lo <= r["spearman"] <= hi
    assert lo < hi


def test_too_few_samples_returns_missing_not_crash():
    r = E.evaluate(np.array([5.0, 10.0]), np.array([6.0, 9.0]), "tiny")
    assert r["n"] == 2
    assert r["rmse"] is not None
    assert r["spearman_ci95"] is None      # 2 个点算不出 CI


# ---------- 标签口径 ----------

def test_load_labels_excludes_test_by_default():
    """test 47 人只在最终评测时读一次，默认必须读不到。"""
    df = E.load_labels()
    assert set(df["split"].unique()) <= {"train", "dev"}


def test_derived_label_disagreement_is_confined():
    """官方 PHQ8_Binary 与 PHQ8_Score>=10 只允许在已知样本上不一致。

    409 号是官方标签的录入错误（score=10 标 0，其余 8 个 score=10 都标 1）。
    如果这个测试开始失败，说明标签文件被换过，所有分类结果都要重算。
    """
    df = E.load_labels()
    derived = (df[C.COL_PHQ8_SCORE] >= 10).astype(int)
    mismatch = set(df.loc[derived != df[C.COL_PHQ8_BINARY],
                          C.COL_PARTICIPANT_ID].astype(int))
    assert mismatch == {409}, f"标签口径不一致的样本变了: {mismatch}"
