import pandas as pd
import numpy as np
import pytest
import sys, types

# Stub out hddm so the module can be imported without it installed
hddm_stub = types.ModuleType("hddm")
hddm_stub.utils = types.SimpleNamespace(flip_errors=lambda df: df)
sys.modules.setdefault("hddm", hddm_stub)

from hddm.hddm_prep import (
    compute_acceptance_pair,
    compute_ov_value,
    compute_abs_value,
    compute_ov_money_pain,
    compute_abs_money_pain,
    compute_z_scores,
)


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "participant": ["s1", "s1", "s1", "s2", "s2"],
        "moneylevel":  [3,    5,    2,    4,    1],
        "painlevel":   [3,    2,    5,    4,    2],
    })


def test_acceptance_pair(base_df):
    result = compute_acceptance_pair(base_df)
    # money==pain → I; money>pain → M; money<pain → P
    assert list(result) == ["I", "M", "P", "I", "M"]


def test_ov_value(base_df):
    result = compute_ov_value(base_df)
    # sum<=5 → low_OV; sum>6 → high_OV; else mid_OV
    # sums: 6, 7, 7, 8, 3
    assert list(result) == ["mid_OV", "high_OV", "high_OV", "high_OV", "low_OV"]


def test_abs_value(base_df):
    result = compute_abs_value(base_df)
    # |diff|<2 → low_abs; |diff|>2 → high_abs; else mid_abs
    # diffs: 0, 3, 3, 0, 1
    assert list(result) == ["low_abs", "high_abs", "high_abs", "low_abs", "low_abs"]


def test_ov_money_pain(base_df):
    result = compute_ov_money_pain(base_df)
    # row 0: sum=6>5, money==pain → h_OV_h_pain
    # row 1: sum=7>5, money>pain  → h_OV_h_money
    # row 2: sum=7>5, pain>money  → h_OV_h_pain
    # row 3: sum=8>5, money==pain → h_OV_h_pain
    # row 4: sum=3<=5, pain>money → low_OV_h_pain
    assert list(result) == [
        "h_OV_h_pain", "h_OV_h_money", "h_OV_h_pain", "h_OV_h_pain", "low_OV_h_pain"
    ]


def test_abs_money_pain(base_df):
    result = compute_abs_money_pain(base_df)
    # diffs: 0, 3, 3, 0, 1
    # row 0: |0|<2, money==pain  → low_abs_h_pain (pain>=money branch)
    # row 1: |3|>2, money>pain   → high_abs_h_money
    # row 2: |3|>2, pain>money   → high_abs_h_pain
    # row 3: |0|<2, money==pain  → low_abs_h_pain
    # row 4: |1|<2, pain>money   → low_abs_h_pain
    assert list(result) == [
        "low_abs_h_pain", "high_abs_h_money", "high_abs_h_pain",
        "low_abs_h_pain", "low_abs_h_pain"
    ]


def test_compute_z_scores():
    df = pd.DataFrame({
        "subj_idx":   ["s1", "s1", "s1", "s2", "s2", "s2"],
        "painlevel":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "moneylevel": [2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
    })
    result = compute_z_scores(df)
    # within s1: pain mean=2, std=1 → z-scores [-1, 0, 1]
    np.testing.assert_allclose(result.loc[result.subj_idx == "s1", "pain_z"].values,
                               [-1.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(result.loc[result.subj_idx == "s2", "pain_z"].values,
                               [-1.0, 0.0, 1.0], atol=1e-10)
    # money z-scores for s1: mean=4, std=2 → [-1, 0, 1]
    np.testing.assert_allclose(result.loc[result.subj_idx == "s1", "money_z"].values,
                               [-1.0, 0.0, 1.0], atol=1e-10)


def test_z_scores_single_value_subject():
    """Subject with no variance gets NaN z-scores, not a crash."""
    df = pd.DataFrame({
        "subj_idx":   ["s1", "s1", "s2"],
        "painlevel":  [3.0, 3.0, 2.0],
        "moneylevel": [2.0, 2.0, 4.0],
    })
    result = compute_z_scores(df)
    assert pd.isna(result.loc[result.subj_idx == "s1", "pain_z"]).all()
