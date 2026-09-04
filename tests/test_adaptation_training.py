from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from devlm.adaptation.train import (
    TrainingOptions, discover_checkpoints, load_construction_manifest, train_all,
)
from devlm.model import CausalPhonemeGRU


class AdaptationTrainingTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        feature_path = root / "ipa_feature_mapping.json"
        feature_path.write_text(json.dumps({
            "feature_names": ["f1", "f2"], "phonemes": {"a": [1.0, 0.0]},
        }), encoding="utf-8")
        stimulus_path = root / "adaptation_all_tasks.tsv"
        fields = [
            "item_id", "task", "condition", "binary_label", "split",
            "word1_ipa", "word2_ipa", "sentence_ipa", "source_record_id",
        ]
        split_sizes = {"train": 360, "validation": 100, "test": 160}
        with stimulus_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for task in ("Sound", "Meaning", "Plausibility", "Grammaticality"):
                index = 0
                for split, count in split_sizes.items():
                    for within_split in range(count):
                        index += 1
                        pair = task in {"Sound", "Meaning"}
                        writer.writerow({
                            "item_id": f"{task}_{index:04d}", "task": task,
                            "condition": "synthetic", "binary_label": within_split % 2,
                            "split": split,
                            "word1_ipa": '["a"]' if pair else "",
                            "word2_ipa": '["a"]' if pair else "",
                            "sentence_ipa": "" if pair else '["a"]',
                            "source_record_id": "synthetic-test-only",
                        })
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        for index in range(30):
            model = CausalPhonemeGRU(2, 4, 1, 1, 0.0)
            torch.save({
                "model": model.state_dict(), "optimizer": {},
                "metadata": {
                    "equivalent_input_duration_hours": float(index + 1),
                    "optimizer_step": index + 1,
                },
                "config": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
                "vocabulary": {"a": 0},
            }, checkpoint_dir / f"checkpoint_{index + 1:02d}.pt")
        return stimulus_path, checkpoint_dir, feature_path

    def test_manifest_loader_and_checkpoint_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stimulus, checkpoints, _ = self._fixture(Path(directory))
            items = load_construction_manifest(stimulus)
            ordered = discover_checkpoints(checkpoints)
        self.assertEqual(len(items), 4 * 620)
        self.assertEqual([row[0] for row in ordered], [f"M{i:02d}" for i in range(1, 31)])
        self.assertEqual([row[2] for row in ordered], list(map(float, range(1, 31))))

    def test_end_to_end_cpu_smoke_writes_every_head_and_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stimulus, checkpoints, features = self._fixture(root)
            output = root / "output"
            rows = train_all(
                stimulus, checkpoints, output, features,
                TrainingOptions(
                    batch_size=360, encoding_batch_size=620,
                    max_epochs=1, patience=1, noise_sigma=0.0,
                    initialization_seeds=(7,), device="cpu",
                ),
            )
            heads = list(output.rglob("*_linear_readout.pt"))
            with (output / "all_task_checkpoint_metrics.tsv").open() as handle:
                written = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 4 * 30)
        self.assertEqual(len(written), 4 * 30)
        self.assertEqual(len(heads), 4 * 30)


if __name__ == "__main__":
    unittest.main()
