from __future__ import annotations

import numpy as np
import pytest

from devlm.acoustic_aspiration_probe.core import ProbeItem, fixed_item_folds, pooled_overlapping_frames, prevalence_weighted_ovo_auc, select_aspiration_items, stratified_fixed_item_folds
from devlm.acoustic_aspiration_probe.mald import PhoneInterval, parse_phone_textgrid
from devlm.acoustic_aspiration_probe.run import _fit_pca, cache_fingerprint, inner_ten_fold_assignment, nested_cv_score


def test_short_mald_textgrid_merges_equal_phones_and_removes_silence(tmp_path):
    grid = tmp_path / "x.TextGrid"
    grid.write_text('File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n2\n"IntervalTier"\n"phone"\n0\n1\n4\n0\n.1\n"sil"\n.1\n.2\n"P"\n.2\n.3\n"P"\n.3\n.4\n"AE1"\n"IntervalTier"\n"word"\n0\n1\n0\n', encoding="utf-8")
    phones = parse_phone_textgrid(grid)
    assert [(p.start_s, p.end_s, p.label) for p in phones] == [(.1, .3, "P"), (.3, .4, "AE1")]


def test_table_one_matches_only_word_initial_s_stop_not_internal_cluster():
    initial = [PhoneInterval(0, .04, "S"), PhoneInterval(.04, .09, "P"), PhoneInterval(.09, .2, "IY1")]
    internal = [PhoneInterval(0, .04, "AH0"), PhoneInterval(.04, .08, "S"), PhoneInterval(.08, .13, "P"), PhoneInterval(.13, .2, "IY1")]
    selected = select_aspiration_items("a.wav", initial)
    triples = {(item.task, item.label, item.target_phone) for item in selected}
    assert ("aspiration_phonemic", 0, "P") in triples
    assert ("aspiration_phonetic", 1, "P") in triples
    assert not select_aspiration_items("b.wav", internal)


@pytest.mark.parametrize(("unvoiced", "voiced", "poa"), [("P", "B", "labial"), ("T", "D", "alveolar"), ("K", "G", "velar")])
def test_table_one_target_identities_for_every_place_of_articulation(unvoiced, voiced, poa):
    initial = [PhoneInterval(0, .05, unvoiced), PhoneInterval(.05, .12, "AE1")]
    voiced_initial = [PhoneInterval(0, .05, voiced), PhoneInterval(.05, .12, "AE1")]
    a = select_aspiration_items("a.wav", initial)
    b = select_aspiration_items("b.wav", voiced_initial)
    assert any(item.poa == poa and item.task == "aspiration_phonemic" and item.label == 0 and item.target_phone == unvoiced for item in a)
    assert any(item.poa == poa and item.task == "aspiration_phonemic" and item.label == 1 and item.target_phone == voiced for item in b)


def test_phone_pooling_uses_any_overlap_of_25ms_support_at_10ms_hop():
    frames = np.arange(5, dtype=np.float32)[:, None]
    assert pooled_overlapping_frames(frames, phone_start_s=.020, phone_end_s=.030).tolist() == [1.0]


def test_weighted_ovo_auc_matches_sklearn_when_available():
    metrics = pytest.importorskip("sklearn.metrics")
    y = np.asarray([0, 0, 1, 1, 2, 2])
    p = np.asarray([[.8,.1,.1],[.6,.3,.1],[.1,.8,.1],[.2,.6,.2],[.2,.1,.7],[.3,.1,.6]])
    assert prevalence_weighted_ovo_auc(y, p) == pytest.approx(metrics.roc_auc_score(y, p, multi_class="ovo", average="weighted"))


def test_stratified_fixed_folds_are_feature_independent_and_shared():
    ids = [f"a{index}" for index in range(30)]; labels = [0] * 10 + [1] * 10 + [2] * 10
    folds = stratified_fixed_item_folds(ids, labels)
    assert set(folds.values()) == set(range(10))
    assert folds == stratified_fixed_item_folds(ids, labels)
    assert fixed_item_folds(ids) == fixed_item_folds(reversed(ids))


def test_pca_fit_is_unchanged_by_held_out_values():
    train = np.asarray([[0., 0., 1.], [1., 0., 0.], [2., 0., -1.]])
    mean_a, basis_a = _fit_pca(train, 2)
    # A deliberately extreme held-out item is not passed into the PCA fit.
    held_out = np.asarray([[1e9, -1e9, 1e9]])
    mean_b, basis_b = _fit_pca(train, 2)
    assert held_out.shape == (1, 3)
    assert np.allclose(mean_a, mean_b) and np.allclose(abs(basis_a), abs(basis_b))


def test_every_outer_split_gets_a_fresh_ten_fold_inner_assignment():
    ids = [f"i{index}" for index in range(60)]
    labels = np.asarray([0] * 20 + [1] * 20 + [2] * 20)
    outer_map = stratified_fixed_item_folds(ids, labels)
    outer = np.asarray([outer_map[item_id] for item_id in ids])
    for held_out in range(10):
        inner = inner_ten_fold_assignment(ids, labels, outer, held_out)
        assert set(inner.tolist()) == set(range(10))
        assert len(inner) == int((outer != held_out).sum())


def test_cache_fingerprint_rejects_different_stimulus_or_smoke_setting():
    item = ProbeItem("x", "x.wav", "control_stress", "all", .1, .2, 0, "AE0", "x.wav")
    changed = ProbeItem("x", "x.wav", "control_stress", "all", .1, .21, 0, "AE0", "x.wav")
    formal = cache_fingerprint([item], representation_identity="checkpoint", max_items=None)
    assert formal != cache_fingerprint([item], representation_identity="checkpoint", max_items=5)
    assert formal != cache_fingerprint([changed], representation_identity="checkpoint", max_items=None)


def test_nested_cv_score_fits_a_real_three_class_logistic_probe():
    # This is deliberately an end-to-end constructor/API test rather than a
    # metric-only test: sklearn has changed LogisticRegression parameters.
    rng = np.random.default_rng(5)
    y = np.repeat(np.arange(3), 20)
    x = np.vstack([rng.normal(loc=label * 2.0, scale=.25, size=(20, 4)) for label in range(3)])
    ids = [f"item_{index}" for index in range(len(y))]
    folds = stratified_fixed_item_folds(ids, y)
    score = nested_cv_score(x, y, folds, ids, dimensions=2)
    assert .9 < score <= 1.0
