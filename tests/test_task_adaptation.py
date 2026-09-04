from __future__ import annotations

import csv
import inspect
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

import devlm.adaptation as adaptation
from devlm.adaptation.config import DEFAULT_ADAPTATION_SIZES, DEFAULT_INITIALIZATION_SEEDS, load_adaptation_config
from devlm.adaptation.input import build_adaptation_batch
from devlm.adaptation.model import (
    CHECKPOINT_IDS,
    FrozenGRUEncoder,
    instantiate_checkpoint_readouts,
    make_binary_readout,
    readout_logits,
    task_loss,
)
from devlm.adaptation.outputs import AdaptationResult, seed_output_layout, seed_run_manifest
from devlm.adaptation.schema import TASK_NAMES, load_manifest
from devlm.adaptation.split import create_item_split, load_item_split, save_item_split
from devlm.features import FeatureTable
from devlm.model import CausalPhonemeGRU


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "task_adaptation.synthetic.tsv"


class TaskAdaptationInfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.features = FeatureTable.from_json(FIXTURES / "features.json")
        self.vocabulary = {phoneme: index for index, phoneme in enumerate(sorted(self.features.mapping))}
        self.items = load_manifest(MANIFEST)

    def make_checkpoint(self, directory: str, hidden_dim: int = 5) -> Path:
        torch.manual_seed(91)
        model = CausalPhonemeGRU(self.features.width, hidden_dim, 1, len(self.vocabulary), 0.0)
        path = Path(directory) / "synthetic_checkpoint.pt"
        torch.save({
            "config": {"hidden_size": hidden_dim, "num_layers": 1, "dropout": 0.0},
            "vocabulary": self.vocabulary,
            "model": model.state_dict(),
        }, path)
        return path

    def make_encoder(self, directory: str, hidden_dim: int = 5) -> FrozenGRUEncoder:
        return FrozenGRUEncoder.from_checkpoint(
            self.make_checkpoint(directory, hidden_dim), self.features.width,
            expected_hidden_dim=hidden_dim,
        )

    def test_01_all_language_model_parameters_are_frozen_and_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = self.make_encoder(directory)
            self.assertFalse(encoder.language_model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in encoder.language_model.parameters()))
            encoder.train()
            self.assertFalse(encoder.language_model.training)

    def test_02_backward_and_head_step_do_not_change_gru(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = self.make_encoder(directory)
            before = encoder.parameter_snapshot()
            batch = build_adaptation_batch(self.items[:2], self.features, self.vocabulary, seed=7)
            head = make_binary_readout(encoder.hidden_dim, seed=12)
            optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
            optimizer.zero_grad()
            task_loss()(readout_logits(encoder, head, batch.frames, batch.lengths), batch.labels).backward()
            optimizer.step()  # The single synthetic head-only step permitted by the specification.
            encoder.assert_unchanged(before)

    def test_03_only_linear_readout_parameters_receive_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = self.make_encoder(directory)
            batch = build_adaptation_batch(self.items[:2], self.features, self.vocabulary, seed=8)
            head = make_binary_readout(encoder.hidden_dim, seed=12)
            task_loss()(readout_logits(encoder, head, batch.frames, batch.lengths), batch.labels).backward()
            self.assertTrue(all(parameter.grad is None for parameter in encoder.language_model.parameters()))
            self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_04_readout_is_exactly_linear_hidden_dim_to_one(self):
        head = make_binary_readout(17, seed=3)
        self.assertIs(type(head), nn.Linear)
        self.assertEqual((head.in_features, head.out_features), (17, 1))
        self.assertEqual(set(dict(head.named_parameters())), {"weight", "bias"})

    def test_05_task_loss_is_bce_with_logits(self):
        self.assertIs(type(task_loss()), nn.BCEWithLogitsLoss)

    def test_06_age_never_enters_readout(self):
        self.assertNotIn("age", inspect.signature(readout_logits).parameters)
        self.assertNotIn("age", inspect.signature(build_adaptation_batch).parameters)

    def test_07_participant_identity_never_enters_readout(self):
        parameters = inspect.signature(readout_logits).parameters
        self.assertNotIn("participant", parameters)
        self.assertNotIn("participant_id", parameters)

    def test_08_rt_never_enters_readout(self):
        parameters = inspect.signature(readout_logits).parameters
        self.assertNotIn("rt", parameters)
        self.assertNotIn("reaction_time", parameters)

    def test_09_manifest_rejects_behavioral_columns(self):
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
            fieldnames = list(rows[0]) + ["participant_id", "age", "rt"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerow({**rows[0], "participant_id": "p1", "age": "5", "rt": "500"})
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_manifest(path)

    def test_10_item_partitions_are_grouped_and_disjoint(self):
        sound = [item for item in self.items if item.task_name == "Sound"]
        split = create_item_split(sound, 42, 0.5, 0.25, 0.25)
        partitions = {
            name: {item_id for item_id, assigned in split.assignments.items() if assigned == name}
            for name in ("train", "validation", "test")
        }
        self.assertTrue(partitions["train"].isdisjoint(partitions["validation"]))
        self.assertTrue(partitions["train"].isdisjoint(partitions["test"]))
        self.assertTrue(partitions["validation"].isdisjoint(partitions["test"]))
        self.assertEqual(set.union(*partitions.values()), {item.item_id for item in sound})

    def test_11_saved_split_is_reused_for_all_checkpoints(self):
        sound = [item for item in self.items if item.task_name == "Sound"]
        split = create_item_split(sound, 42, 0.5, 0.25, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_item_split(split, path)
            assignments = [load_item_split(path, sound).assignments for _ in CHECKPOINT_IDS]
        self.assertEqual(len(assignments), 30)
        self.assertTrue(all(value == split.assignments for value in assignments))

    def test_12_synthetic_word_pair_and_sentence_complete_pipeline(self):
        examples = [
            next(item for item in self.items if item.stimulus_kind == "word_pair"),
            next(item for item in self.items if item.stimulus_kind == "sentence"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            encoder = self.make_encoder(directory, hidden_dim=7)
            batch = build_adaptation_batch(examples, self.features, self.vocabulary, seed=5, noise_sigma=0.05)
            hidden = encoder(batch.frames, batch.lengths)
            logits = make_binary_readout(7, seed=6)(hidden)
        self.assertEqual(tuple(batch.frames.shape[:1]), (2,))
        self.assertEqual(tuple(hidden.shape), (2, 7))
        self.assertEqual(tuple(logits.shape), (2, 1))
        self.assertEqual(tuple(batch.labels.shape), (2,))

    def test_13_all_thirty_identically_initialized_heads_instantiate_without_training(self):
        heads = instantiate_checkpoint_readouts(9, seed=27)
        self.assertEqual(tuple(heads), CHECKPOINT_IDS)
        self.assertEqual(len(heads), 30)
        reference = heads["M01"].state_dict()
        for head in heads.values():
            self.assertIs(type(head), nn.Linear)
            self.assertTrue(all(torch.equal(reference[key], value) for key, value in head.state_dict().items()))

    def test_14_import_exposes_training_without_starting_it(self):
        self.assertTrue(callable(adaptation.train_all))
        self.assertTrue((Path(adaptation.__file__).parent / "train.py").exists())
        heads = instantiate_checkpoint_readouts(3, seed=1)
        self.assertTrue(all(parameter.grad is None for head in heads.values() for parameter in head.parameters()))

    def test_schema_supports_four_tasks_and_both_stimulus_forms(self):
        self.assertEqual({item.task_name for item in self.items}, set(TASK_NAMES))
        self.assertEqual({item.stimulus_kind for item in self.items}, {"word_pair", "sentence"})
        self.assertTrue(all(item.metadata == {"synthetic": True} for item in self.items))

    def test_config_and_future_output_schema_are_present_without_outputs(self):
        config = load_adaptation_config(Path(__file__).parent.parent / "configs" / "task_adaptation.example.toml")
        self.assertEqual(tuple(config["adaptation_sizes"]), DEFAULT_ADAPTATION_SIZES)
        self.assertEqual(tuple(config["initialization_seeds"]), DEFAULT_INITIALIZATION_SEEDS)
        self.assertEqual(
            {field.name for field in fields(AdaptationResult)},
            {
                "checkpoint_id", "task_name", "train_loss", "validation_loss",
                "train_accuracy", "validation_accuracy", "test_accuracy", "validation_AUC",
                "test_AUC", "best_epoch", "number_train_items", "number_validation_items",
                "number_test_items", "readout_state_dict",
            },
        )

    def test_seed_output_namespace_unifies_all_thirty_checkpoint_records(self):
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            layout = seed_output_layout(directory, "Meaning", digest, 2718)
            manifest = seed_run_manifest("Meaning", digest, 2718)
            self.assertEqual(
                layout.directory,
                Path(directory) / "task-meaning" / "split-aaaaaaaaaaaa" / "init-seed-0000002718",
            )
            self.assertEqual(layout.all_checkpoint_metrics.name, "all_checkpoint_metrics.tsv")
            self.assertEqual(tuple(layout.head_state_dicts), CHECKPOINT_IDS)
            self.assertEqual(manifest["checkpoint_ids"], list(CHECKPOINT_IDS))
            self.assertEqual(manifest["initialization_seed"], 2718)
            self.assertEqual(manifest["head_state_dicts"]["M30"], "heads/M30_linear_readout.pt")
            self.assertFalse(layout.directory.exists())


if __name__ == "__main__":
    unittest.main()
