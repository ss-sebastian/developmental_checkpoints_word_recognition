from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from devlm.adaptation.segmentation_probe import (
    BoundarySession,
    SegmentationProbeOptions,
    TransitionExample,
    determine_segmentation_onset,
    extract_transition_states,
    load_phase1_validation_sessions,
    parse_boundary_annotated_ipa,
    run_segmentation_probe,
    sample_transitions,
    session_bootstrap_ci,
    split_probe_sessions,
    train_segmentation_head,
)
from devlm.data import Session, Utterance
from devlm.features import FeatureTable
from devlm.model import CausalPhonemeGRU
from devlm.stream import build_session_stream


class SegmentationProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = FeatureTable(
            ["f1", "f2"], {"a": [1.0, 0.0], "b": [0.0, 1.0], "k": [-1.0, 0.0]},
        )
        self.vocabulary = {"a": 0, "b": 1, "k": 2}

    def test_word_boundary_is_label_only_and_trailing_marker_is_not_a_transition(self) -> None:
        phonemes, labels = parse_boundary_annotated_ipa(
            "a b WORD_BOUNDARY k a WORD_BOUNDARY"
        )
        self.assertEqual(phonemes, ("a", "b", "k", "a"))
        self.assertEqual(labels, (0, 1, 0))
        self.assertNotIn("WORD_BOUNDARY", phonemes)

    def test_loader_uses_only_phase1_validation_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            fields = [
                "corpus_id", "transcript_id", "id", "target_child_age",
                "ipa_transcription", "processed_gloss",
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for session_id in ("s1", "s2", "excluded"):
                    writer.writerow({
                        "corpus_id": "c", "transcript_id": session_id, "id": "1",
                        "target_child_age": "2.0", "ipa_transcription": "a WORD_BOUNDARY b k WORD_BOUNDARY",
                        "processed_gloss": "x",
                    })
            split = root / "session_split.json"
            split.write_text(json.dumps({
                "train": [["c", "excluded"]],
                "validation": [["c", "s1"], ["c", "s2"]],
            }), encoding="utf-8")
            sessions = load_phase1_validation_sessions(source, split)
        self.assertEqual({item.session.session_id for item in sessions}, {"s1", "s2"})
        self.assertEqual(sessions[0].session.utterances[0].phonemes, ("a", "b", "k"))
        self.assertEqual(sessions[0].transition_labels[0], (1, 0))

    def test_fixed_session_split_and_transition_sample_are_disjoint_and_reproducible(self) -> None:
        sessions = []
        for index in range(12):
            utterance = Utterance("c", str(index), 2.0, 1, "a b k a", "", ("a", "b", "k", "a"))
            sessions.append(BoundarySession(
                Session("c", str(index), 2.0, (utterance,)), ((0, 1, 0),),
            ))
        options = SegmentationProbeOptions(
            max_train_transitions=10, max_validation_transitions=6, max_test_transitions=6,
        )
        first = split_probe_sessions(sessions, options)
        second = split_probe_sessions(sessions, options)
        self.assertEqual(
            {key: [x.session.session_id for x in value] for key, value in first.items()},
            {key: [x.session.session_id for x in value] for key, value in second.items()},
        )
        keys = [{x.session.session_id for x in first[split]} for split in ("train", "validation", "test")]
        self.assertTrue(keys[0].isdisjoint(keys[1]) and keys[0].isdisjoint(keys[2]) and keys[1].isdisjoint(keys[2]))
        sample_a = sample_transitions(first, options)
        sample_b = sample_transitions(first, options)
        self.assertEqual(sample_a, sample_b)

    def test_transition_state_is_immediately_before_target_first_active_frame(self) -> None:
        utterance = Utterance("c", "s", 2.0, 1, "a b k", "", ("a", "b", "k"))
        boundary_session = BoundarySession(Session("c", "s", 2.0, (utterance,)), ((1, 0),))
        example = TransitionExample("c:s:1:1", "train", "c", "s", 0, 1, 1, "b", 1)
        torch.manual_seed(3)
        model = CausalPhonemeGRU(2, 4, 1, 3, 0.0).requires_grad_(False).eval()
        actual = extract_transition_states(
            model, [boundary_session], [example], self.features, self.vocabulary,
            noise_sigma=0.0, phoneme_envelope=None, input_noise_seed=1,
            sequence_chunk_frames=3, device=torch.device("cpu"),
        )
        stream = build_session_stream(
            boundary_session.session, self.features, self.vocabulary,
            np.random.default_rng(1), noise_sigma=0.0,
        )
        target_span = [span for span in stream.spans if span.utterance_index == 0][1]
        self.assertEqual(target_span.start_frame, 4)
        with torch.no_grad():
            expected, _ = model.gru(torch.from_numpy(stream.noisy_frames[:target_span.start_frame]).unsqueeze(0))
        self.assertTrue(torch.allclose(actual[0], expected[0, -1]))

    def test_probe_is_exactly_linear_and_reports_requested_metrics(self) -> None:
        generator = torch.Generator().manual_seed(7)
        labels = torch.tensor([0, 1] * 30, dtype=torch.float32)
        states = torch.randn(60, 128, generator=generator)
        states[:, 0] += labels * 3 - 1.5
        indices = {
            "train": torch.arange(0, 40),
            "validation": torch.arange(40, 50),
            "test": torch.arange(50, 60),
        }
        head, metrics, _ = train_segmentation_head(
            states, labels, indices,
            SegmentationProbeOptions(max_epochs=5, patience=2, batch_size=40),
            torch.device("cpu"),
        )
        self.assertIs(type(head), nn.Linear)
        self.assertEqual((head.in_features, head.out_features), (128, 1))
        self.assertEqual(set(metrics["test"]), {"auc", "balanced_accuracy", "boundary_prevalence", "accuracy"})

    def test_session_bootstrap_and_sustained_onset_rule(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        logits = np.asarray([-2, 2, -1, 1, -3, 3, -4, 4], dtype=float)
        sessions = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        intervals = session_bootstrap_ci(labels, logits, sessions, 100, 9)
        self.assertGreater(intervals["auc"][0], 0.5)
        rows = [{"checkpoint_id": "M00", "checkpoint_hours": 0.0, "test_auc": 0.52, "test_auc_ci_low": 0.48}]
        for index in range(1, 31):
            rows.append({
                "checkpoint_id": f"M{index:02d}", "checkpoint_hours": index * 9.9,
                "test_auc": 0.54 if index >= 2 else 0.51,
                "test_auc_ci_low": 0.51 if index >= 2 else 0.49,
            })
        onset = determine_segmentation_onset(rows)
        self.assertTrue(onset["detected"])
        self.assertEqual(onset["checkpoint_id"], "M02")

    def test_end_to_end_smoke_runs_m00_and_all_thirty_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            fields = [
                "corpus_id", "transcript_id", "id", "target_child_age",
                "ipa_transcription", "processed_gloss",
            ]
            validation = []
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for session_index in range(12):
                    session_id = f"s{session_index:02d}"
                    validation.append(["c", session_id])
                    writer.writerow({
                        "corpus_id": "c", "transcript_id": session_id, "id": "1",
                        "target_child_age": "2.0",
                        "ipa_transcription": "a WORD_BOUNDARY b k WORD_BOUNDARY a b WORD_BOUNDARY",
                        "processed_gloss": "synthetic",
                    })
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            (checkpoint_root / "session_split.json").write_text(
                json.dumps({"train": [], "validation": validation}), encoding="utf-8",
            )
            (checkpoint_root / "ipa_feature_mapping.json").write_text(json.dumps({
                "feature_names": list(self.features.feature_names),
                "phonemes": {key: value.tolist() for key, value in self.features.mapping.items()},
            }), encoding="utf-8")
            for checkpoint_index in range(30):
                torch.manual_seed(checkpoint_index)
                model = CausalPhonemeGRU(2, 128, 1, 3, 0.0)
                torch.save({
                    "model": model.state_dict(), "optimizer": {},
                    "metadata": {
                        "equivalent_input_duration_hours": checkpoint_index + 1.0,
                        "optimizer_step": checkpoint_index + 1,
                    },
                    "config": {
                        "seed": 91, "hidden_size": 128, "num_layers": 1,
                        "dropout": 0.0, "noise_sigma": 0.0,
                        "phoneme_envelope": [1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3],
                    },
                    "vocabulary": self.vocabulary,
                }, checkpoint_root / f"checkpoint_{checkpoint_index:02d}.pt")
            output = root / "outputs"
            zip_path = run_segmentation_probe(
                source, checkpoint_root, output,
                options=SegmentationProbeOptions(
                    max_train_transitions=12, max_validation_transitions=6,
                    max_test_transitions=6, max_epochs=1, patience=1,
                    batch_size=12, bootstrap_repetitions=10, device="cpu",
                ),
            )
            with (output / "summary" / "checkpoint_metrics.tsv").open(
                encoding="utf-8", newline="",
            ) as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(zip_path.name.endswith(".zip"))
        self.assertEqual([row["checkpoint_id"] for row in rows], ["M00", *[f"M{x:02d}" for x in range(1, 31)]])
        self.assertEqual(manifest["models"], 31)


if __name__ == "__main__":
    unittest.main()
