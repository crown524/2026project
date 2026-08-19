"""标签口径的回归测试。

方案 §4.1 要求主结果用官方 PHQ8_Binary，敏感性分析另报 PHQ8_Score>=10。
两者在 train+dev 142 人上只有 409 号不一致（PHQ-8=10 却标 0），
而其余 8 个 10 分的人官方都标 1。这些测试守住这个已核对的事实：
若数据换版导致不一致人数变化，说明标签源变了，必须重新审计后再跑实验。
"""
from __future__ import annotations

import pandas as pd
import pytest

import config as C
import eval_utils as E

KNOWN_MISMATCH = {409}


@pytest.fixture(scope="module")
def labels() -> pd.DataFrame:
    df = E.load_labels()
    if df.empty:
        pytest.skip("缺少标签文件")
    df = df.copy()
    df["pid"] = df[C.COL_PARTICIPANT_ID].astype(int)
    df["official"] = df[C.COL_PHQ8_BINARY].astype(int)
    df["derived"] = (df[C.COL_PHQ8_SCORE].astype(float) >= 10).astype(int)
    return df


def test_load_labels_excludes_test_split(labels):
    """test 47 人只在最终冻结后评一次，默认装载不能带进来。"""
    assert set(labels["split"]) <= {"train", "dev"}


def test_only_known_mismatch(labels):
    mismatch = set(labels.loc[labels.official != labels.derived, "pid"])
    assert mismatch == KNOWN_MISMATCH, (
        f"两种标签口径的不一致集合变了: {mismatch}，"
        "说明标签源或划分已换版，需重新生成 label_audit 后才能沿用旧结果"
    )


def test_boundary_score_mostly_positive(labels):
    """PHQ-8=10 是官方阈值，除 409 外都应标为阳性。"""
    at = labels[labels[C.COL_PHQ8_SCORE].astype(float) == 10]
    assert len(at) >= 5
    assert set(at.loc[at.official == 0, "pid"]) == KNOWN_MISMATCH


def test_positive_rate_close_between_policies(labels):
    """一个人的差异不该让正类比例明显漂移，否则不能做敏感性对比。"""
    assert abs(labels.official.mean() - labels.derived.mean()) < 0.02


def test_binary_is_binary(labels):
    assert set(labels.official) <= {0, 1}
    assert labels[C.COL_PHQ8_SCORE].between(0, 24).all()


def test_no_duplicate_participants(labels):
    """重复 ID 会在 merge 时静默放大样本，必须挡住。"""
    assert labels["pid"].is_unique
