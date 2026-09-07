from __future__ import annotations

import csv
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import devlm.adaptation.meaning_probe as probe
from devlm.adaptation.meaning_probe import (
    EXPECTED_ITEM_COUNT, HISTORY_COLUMNS, MeaningProbeOptions,
    load_meaning_manifest, run_meaning_probe, train_meaning_head,
)
from devlm.model import CausalPhonemeGRU


def _write_stimuli(path: Path, count: int = EXPECTED_ITEM_COUNT) -> None:
    fields = [
        "item_id", "task", "condition", "binary_label", "split", "cue", "target",
        "word1_ipa", "word2_ipa", "FSG", "BSG",
    ]
    split_conditions = {
        "train": (("High", 145), ("Low", 145), ("Unrelated", 290)),
        "validation": (("High", 40), ("Low", 40), ("Unrelated", 80)),
        "test": (("High", 65), ("Low", 65), ("Unrelated", 130)),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        written = 0
        for split, condition_counts in split_conditions.items():
            for condition, condition_count in condition_counts:
                for _ in range(condition_count):
                    if written >= count:
                        return
                    writer.writerow({
                        "item_id": f"meaning_{written:04d}", "task": "Meaning",
                        "condition": condition, "binary_label": int(condition != "Unrelated"),
                        "split": split, "cue": "a", "target": "a",
                        "word1_ipa": '["a"]', "word2_ipa": '["a"]',
                        "FSG": "0.5" if condition == "High" else "0.2" if condition == "Low" else "0",
                        "BSG": "0.1" if condition != "Unrelated" else "0",
                    })
                    written += 1


class MeaningProbeTests(unittest.TestCase):
    def test_repository_formal_manifest_is_the_fixed_audited_1000_item_design(self) -> None:
        path = Path(__file__).parent.parent / "data" / "task_adaptation_stimuli" / "meaning_1000" / "final_all.tsv"
        items = load_meaning_manifest(path)
        self.assertEqual(len(items), 1000)
        self.assertEqual({split: sum(item.split == split for item in items)
                          for split in ("train", "validation", "test")},
                         {"train": 580, "validation": 160, "test": 260})

    def test_loader_requires_exactly_1000_meaning_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meaning.tsv"
            _write_stimuli(path, 999)
            with self.assertRaisesRegex(ValueError, "exactly 1000"):
                load_meaning_manifest(path)
            _write_stimuli(path)
            items = load_meaning_manifest(path)
        self.assertEqual(len(items), 1000)
        self.assertEqual({item.split for item in items}, {"train", "validation", "test"})

    def test_training_history_is_complete_and_test_is_evaluated_once(self) -> None:
        generator = torch.Generator().manual_seed(10)
        labels = torch.tensor([0, 1] * 15, dtype=torch.float32)
        representations = torch.randn(30, 4, generator=generator)
        representations[:, 0] += labels * 2 - 1
        indices = {
            "train": torch.arange(0, 18),
            "validation": torch.arange(18, 24),
            "test": torch.arange(24, 30),
        }
        calls: list[str] = []
        original = probe._evaluate_once

        def counted(*args, **kwargs):
            calls.append(args[-1] if len(args) >= 7 else kwargs["context"])
            return original(*args, **kwargs)

        with mock.patch.object(probe, "_evaluate_once", side_effect=counted):
            _, history, summary, _ = train_meaning_head(
                representations, labels, indices, "M01",
                MeaningProbeOptions(
                    learning_rate=0.03, batch_size=18, max_epochs=4, patience=2,
                    min_delta=1e-6, noise_sigma=0.0, device="cpu",
                ),
                torch.device("cpu"), lambda _: None,
            )
        self.assertEqual(sum(":test:" in context for context in calls), 1)
        self.assertFalse(any("test" in key for key in HISTORY_COLUMNS))
        self.assertEqual(set(history[0]), set(HISTORY_COLUMNS))
        self.assertEqual(sum(int(row["is_best_epoch"]) for row in history), 1)
        self.assertIn("test_auc", summary)
        self.assertIn("test_TP", summary)

    def test_end_to_end_bundle_has_every_required_artifact_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stimuli = root / "meaning.tsv"
            _write_stimuli(stimuli)
            feature_path = root / "ipa_feature_mapping.json"
            feature_path.write_text(json.dumps({
                "feature_names": ["f1", "f2"], "phonemes": {"a": [1.0, 0.0]},
            }), encoding="utf-8")
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            for index in range(30):
                model = CausalPhonemeGRU(2, 4, 1, 1, 0.0)
                torch.save({
                    "model": model.state_dict(), "optimizer": {},
                    "metadata": {"equivalent_input_duration_hours": index + 1.0, "optimizer_step": index + 1},
                    "config": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
                    "vocabulary": {"a": 0},
                }, checkpoint_root / f"checkpoint_{index + 1:02d}.pt")
            output = root / "meaning_probe"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                zip_path = run_meaning_probe(
                    stimuli, checkpoint_root, output, feature_path,
                    MeaningProbeOptions(
                        batch_size=600, encoding_batch_size=1000, max_epochs=1, patience=1,
                        noise_sigma=0.0, initialization_seed=7, device="cpu",
                    ),
                )
            with (output / "summary" / "run_status.csv").open() as handle:
                statuses = list(csv.DictReader(handle))
            cache = np.load(output / "cached_hidden" / "M01_hfinal.npz")
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
        self.assertEqual(len(statuses), 30)
        self.assertTrue(all(row["status"] == "success" for row in statuses))
        self.assertEqual(sum("early stopping not triggered" in str(item.message) for item in caught), 30)
        self.assertEqual(cache["h_final"].shape, (1000, 4))
        self.assertEqual(cache["h_final"].dtype, np.float32)
        self.assertIn("dataset_sha256", cache.files)
        self.assertIn("stimulus_frame_sha256", cache.files)
        self.assertIn("checkpoint_sha256", cache.files)
        for checkpoint_id in ("M01", "M30"):
            self.assertIn(f"histories/{checkpoint_id}_history.csv", names)
            self.assertIn(f"predictions/{checkpoint_id}_predictions.tsv", names)
            self.assertIn(f"heads/{checkpoint_id}_linear_probe.pt", names)
            self.assertIn(f"cached_hidden/{checkpoint_id}_hfinal.npz", names)
        for required in (
            "summary/checkpoint_probe_metrics.csv", "summary/condition_metrics.csv",
            "summary/run_status.csv", "configs/training_config.json",
            "configs/dataset_manifest.json", "configs/checkpoint_manifest.json",
            "configs/environment.json", "run_manifest.json", "run_log.txt",
        ):
            self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main()
