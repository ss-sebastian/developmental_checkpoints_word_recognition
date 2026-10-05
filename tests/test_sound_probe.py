from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import pytest
import torch

from devlm.adaptation.sound_probe import (
    SoundItem,
    SoundProbeOptions,
    build_sound_frames,
    load_sound_manifest,
    prepare_sound_manifest,
    evaluate_split,
    fit_sound_head,
)
from devlm.features import FeatureTable


def _source_rows() -> list[dict[str, str]]:
    rows = []
    for split, prefix in (("train", "tr"), ("validation", "va"), ("test", "te")):
        examples = [
            ("onset", 1, "a", "b"), ("onset", 1, "c", "d"),
            ("rhyme", 1, "e", "f"), ("rhyme", 1, "g", "h"),
            ("unrelated", 0, "i", "j"), ("unrelated", 0, "k", "l"),
            ("unrelated", 0, "m", "n"), ("unrelated", 0, "o", "p"),
        ]
        for index, (condition, label, word1, word2) in enumerate(examples):
            word1, word2 = f"{prefix}{word1}", f"{prefix}{word2}"
            rows.append({
                "item_id": f"{prefix}{index}", "task": "Sound", "condition": condition,
                "binary_label": str(label), "split": split, "word1": word1, "word2": word2,
                "word1_ipa": json.dumps([f"{word1}0"]), "word2_ipa": json.dumps([f"{word2}0"]),
                "word1_n_syllables": "1", "word2_n_syllables": "1",
                "childes_count_word1": "1", "childes_count_word2": "1",
                "qc_pass": "1", "phonological_relation_verified": "1",
                "shared_onset": "onset" if condition == "onset" else "",
                "shared_rime": "rime" if condition == "rhyme" else "",
                "shared_phoneme_count": "0" if condition == "unrelated" else "1",
                "word1_subtlex": "2.0", "word2_subtlex": "2.0",
                "word1_n_phonemes": "1", "word2_n_phonemes": "1",
            })
    return rows


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_sound_manifest_is_model_blind_balanced_and_partition_preserving(tmp_path):
    source, output = tmp_path / "candidates.tsv", tmp_path / "primary.tsv"
    _write_manifest(source, _source_rows())
    report = prepare_sound_manifest(source, output, max_word_reuse=2, max_component_words=4)
    items = load_sound_manifest(output, max_word_reuse=2, max_component_words=4)
    assert report["selection_is_model_blind"] is True
    assert len(items) == 24
    assert Counter((item.split, item.condition) for item in items) == Counter({
        (split, condition): count
        for split in ("train", "validation", "test")
        for condition, count in (("onset", 2), ("rhyme", 2), ("unrelated", 4))
    })
    word_sets = {split: {word for item in items if item.split == split for word in (item.word1, item.word2)}
                 for split in ("train", "validation", "test")}
    assert not (word_sets["train"] & word_sets["validation"])
    assert not (word_sets["train"] & word_sets["test"])
    assert not (word_sets["validation"] & word_sets["test"])


def test_sound_manifest_rejects_unattested_words_and_lexical_leakage(tmp_path):
    path = tmp_path / "manifest.tsv"
    rows = _source_rows()
    rows[0]["childes_count_word1"] = "0"
    _write_manifest(path, rows)
    with pytest.raises(ValueError, match="attested"):
        load_sound_manifest(path)
    rows = _source_rows()
    rows[0]["word1"] = rows[8]["word1"]
    _write_manifest(path, rows)
    with pytest.raises(ValueError, match="lexical leakage"):
        load_sound_manifest(path)


def test_sound_stream_has_clean_pause_and_no_cross_word_overlap():
    feature_table = FeatureTable(
        ["f1", "f2", "f3"], {f"{letter}0": [float(index + 1), 0, 0]
                                for index, letter in enumerate("abcdefghijklmnop")},
    )
    item = SoundItem("pair", "test", "onset", 1, "word1", "word2", ("a0", "b0"), ("c0", "d0"), "pair")
    frames, length = build_sound_frames(
        item, feature_table, {f"{letter}0": index for index, letter in enumerate("abcd")},
        np.random.default_rng(9), noise_sigma=0.1,
    )
    # Two 2-phoneme words: 9 frames each, separated by exactly 3 zero frames.
    assert frames.shape == (21, 3)
    assert length == 21
    assert np.array_equal(frames[9:12], np.zeros((3, 3), dtype=np.float32))
    assert np.any(frames[:9]) and np.any(frames[12:])


def test_fitting_cannot_access_test_and_test_runs_once_after_validation_selection():
    rng = torch.Generator().manual_seed(30)
    train_x = torch.randn(40, 128, generator=rng)
    train_y = torch.tensor([0, 1] * 20, dtype=torch.float32)
    validation_x = torch.randn(10, 128, generator=rng) + 4
    validation_y = torch.tensor([0, 1] * 5, dtype=torch.float32)
    test_x = torch.randn(10, 128, generator=rng) + 50
    test_y = torch.tensor([0, 1] * 5, dtype=torch.float32)
    options = SoundProbeOptions(max_epochs=4, patience=4, batch_size=10, bootstrap_repetitions=10)
    calls = []

    def hook(split, head, x, y):
        calls.append(split)
        from devlm.adaptation.sound_probe import evaluate_split
        return evaluate_split(head, x, y)

    head, history, selected = fit_sound_head(
        train_x, train_y, validation_x, validation_y,
        initialization_seed=1729, options=options, device=torch.device("cpu"), evaluation_hook=hook,
    )
    assert "test" not in calls
    assert calls == [split for _ in history for split in ("train", "validation")]
    assert selected["best_epoch"] in range(1, len(history) + 1)
    test_metrics, test_logits = evaluate_split(head, test_x, test_y)
    calls.append("test")
    assert calls.count("test") == 1 and calls[-1] == "test"
    assert set(test_metrics) == {"auc", "balanced_accuracy", "cross_entropy"}
    assert test_logits.shape == (10,)
    assert head.in_features == 128 and head.out_features == 1
    # The returned/saved head operates directly on raw frozen hidden states;
    # train-only standardization has been folded into its parameters.
    with torch.no_grad():
        assert torch.allclose(head(test_x).squeeze(-1), test_logits, atol=1e-5, rtol=1e-5)

    # Altering held-out examples cannot change the validation-selected head.
    altered_head, _, altered_selected = fit_sound_head(
        train_x, train_y, validation_x, validation_y,
        initialization_seed=1729, options=options, device=torch.device("cpu"),
    )
    assert selected["best_epoch"] == altered_selected["best_epoch"]
    for key, tensor in head.state_dict().items():
        assert torch.equal(tensor, altered_head.state_dict()[key])
