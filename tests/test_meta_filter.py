"""Tests for meta_filter.py's MetaModelFilter.

No real model file, joblib install, or network access is required: the
"unavailable" path is exercised with a nonexistent model path, and the
scoring path is exercised by swapping in a small fake model object after
construction (bypassing the real joblib.load entirely).
"""
from types import SimpleNamespace

import pytest

import meta_filter


def _signal(**overrides):
    base = dict(
        confidence=0.7, trend_change_pct=0.5, atr_pct=1.2, news_sentiment=0.1,
        batch_weight=0.5, online_weight=0.3, markov_weight=0.1, meanrev_weight=0.1,
        hmm_regime_prob=0.6, volatility_regime="normal", trend_regime="trending_up",
        hmm_regime="bull",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_filter_is_unavailable_when_model_file_is_missing(tmp_path):
    filt = meta_filter.MetaModelFilter(model_path=str(tmp_path / "does_not_exist.joblib"))
    assert filt.available is False


def test_score_returns_neutral_default_when_unavailable(tmp_path):
    filt = meta_filter.MetaModelFilter(model_path=str(tmp_path / "missing.joblib"))
    assert filt.score(_signal()) == 0.5


class _FakeModel:
    classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


def test_score_uses_models_probability_for_the_positive_class(tmp_path):
    filt = meta_filter.MetaModelFilter(model_path=str(tmp_path / "missing.joblib"))
    filt.available = True
    filt.model = _FakeModel()
    filt.feature_cols = [
        "entry_confidence", "entry_trend_change_pct", "entry_atr_pct",
        "entry_news_sentiment", "entry_batch_weight", "entry_online_weight",
        "entry_markov_weight", "entry_meanrev_weight", "entry_hmm_regime_prob",
        "vol_normal", "trend_trending_up", "hmm_bull",
    ]
    assert filt.score(_signal()) == pytest.approx(0.8)


def test_score_falls_back_to_neutral_when_positive_class_absent(tmp_path):
    class NoPositiveClassModel:
        classes_ = [0]

        def predict_proba(self, X):
            return [[1.0]]

    filt = meta_filter.MetaModelFilter(model_path=str(tmp_path / "missing.joblib"))
    filt.available = True
    filt.model = NoPositiveClassModel()
    filt.feature_cols = ["entry_confidence"]
    assert filt.score(_signal()) == 0.5


def test_score_swallows_model_exceptions_and_returns_neutral(tmp_path):
    class ExplodingModel:
        classes_ = [0, 1]

        def predict_proba(self, X):
            raise RuntimeError("boom")

    filt = meta_filter.MetaModelFilter(model_path=str(tmp_path / "missing.joblib"))
    filt.available = True
    filt.model = ExplodingModel()
    filt.feature_cols = ["entry_confidence"]
    assert filt.score(_signal()) == 0.5
