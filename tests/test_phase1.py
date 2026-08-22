from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from devlm.data import Session, Utterance, load_ipa_childes, split_sessions
from devlm.config import load_config
from devlm.features import FeatureTable
from devlm.stream import (
    DEFAULT_PHONEME_ENVELOPE,
    PHONEME_FRAMES,
    PHONEME_OVERLAP_FRAMES,
    build_session_stream,
)
from devlm.timing import EmpiricalPauseSampler, FRAME_MS


FIXTURES = Path(__file__).parent / "fixtures"


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.features = FeatureTable.from_json(FIXTURES / "features.json")
        self.vocab = {p: i for i, p in enumerate(sorted(self.features.mapping))}

    def pause_sampler(self, values, rng):
        return EmpiricalPauseSampler(values, rng)

    def session(self, utterances):
        return Session("c", "s", 20.0, tuple(
            Utterance("c", "s", 20.0, i + 1, "".join(ps), text, tuple(ps))
            for i, (text, ps) in enumerate(utterances)
        ))

    def stream(self, utterances, seed=1, noise=0.05, pause_values=(30,), envelope=None):
        rng = np.random.default_rng(seed)
        return build_session_stream(
            self.session(utterances), self.features, self.vocab,
            self.pause_sampler({"__default__": list(pause_values)}, rng), rng,
            noise_sigma=noise, phoneme_envelope=envelope,
        )

    def test_word_boundary_has_no_cue(self):
        stream = self.stream([("a book", ("a", "b"))], noise=0)
        first, second = stream.spans
        self.assertEqual(second.start_frame, first.end_frame - 1)
        self.assertTrue(stream.speech_mask[first.start_frame:second.end_frame].all())

    def test_utterance_boundary_is_zero_silence(self):
        stream = self.stream([("a", ("a",)), ("book", ("b",))], noise=0, pause_values=(30,))
        left, right = stream.spans
        silence = stream.clean_frames[left.end_frame:right.start_frame]
        self.assertEqual(len(silence), 3)
        self.assertTrue(np.array_equal(silence, np.zeros_like(silence)))

    def test_silence_receives_no_noise(self):
        stream = self.stream([("a", ("a",)), ("book", ("b",))], noise=0.5)
        left, right = stream.spans
        self.assertEqual(np.count_nonzero(stream.noisy_frames[left.end_frame:right.start_frame]), 0)

    def test_speech_receives_noise(self):
        stream = self.stream([("a", ("a",))], noise=0.5)
        self.assertFalse(np.array_equal(stream.clean_frames[stream.speech_mask], stream.noisy_frames[stream.speech_mask]))

    def test_each_phoneme_has_exactly_five_active_frames(self):
        stream = self.stream([("a book", ("a", "b"))], noise=0)
        self.assertEqual(PHONEME_FRAMES, 5)
        self.assertTrue(all(span.end_frame - span.start_frame == 5 for span in stream.spans))

    def test_adjacent_phonemes_overlap_exactly_one_frame_as_weighted_sum(self):
        stream = self.stream([("a book", ("a", "b"))], noise=0)
        first, second = stream.spans
        overlap_indices = set(range(first.start_frame, first.end_frame)) & set(range(second.start_frame, second.end_frame))
        self.assertEqual(PHONEME_OVERLAP_FRAMES, 1)
        self.assertEqual(overlap_indices, {second.start_frame})
        expected = (
            DEFAULT_PHONEME_ENVELOPE[-1] * self.features.vector("a")
            + DEFAULT_PHONEME_ENVELOPE[0] * self.features.vector("b")
        )
        np.testing.assert_allclose(stream.clean_frames[second.start_frame], expected)

    def test_frame_is_exactly_ten_ms(self):
        self.assertEqual(FRAME_MS, 10)
        self.assertEqual(self.stream([("a", ("a",))]).frame_ms, 10)

    def test_target_leakage_is_absent(self):
        stream = self.stream([("a book", ("a", "b"))], noise=0)
        source_span = stream.spans[0]
        target_span = stream.spans[1]
        self.assertEqual(stream.target_frames[0], target_span.start_frame - 1)
        source_envelope_index = stream.target_frames[0] - source_span.start_frame
        expected_without_target = DEFAULT_PHONEME_ENVELOPE[source_envelope_index] * self.features.vector("a")
        np.testing.assert_allclose(stream.clean_frames[stream.target_frames[0]], expected_without_target)
        self.assertEqual(target_span.start_frame, source_span.end_frame - 1)

    def test_session_split_has_no_overlap(self):
        sessions = load_ipa_childes(FIXTURES / "ipa_childes.synthetic.jsonl")
        train, val = split_sessions(sessions, 0.25, 9)
        train_ids = {(s.corpus_id, s.session_id) for s in train}
        val_ids = {(s.corpus_id, s.session_id) for s in val}
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertEqual([s.target_child_age_months for s in train], sorted(s.target_child_age_months for s in train))

    def test_seed_reproduces_pauses_and_initial_stream(self):
        a = self.stream([("a", ("a",)), ("book", ("b",))], seed=44, pause_values=(20, 30, 40))
        b = self.stream([("a", ("a",)), ("book", ("b",))], seed=44, pause_values=(20, 30, 40))
        self.assertEqual(a.spans, b.spans)
        self.assertTrue(np.array_equal(a.noisy_frames, b.noisy_frames))
        self.assertTrue(np.array_equal(a.target_ids, b.target_ids))

    def test_phoneme_duration_file_is_not_required(self):
        config = load_config(Path(__file__).parent.parent / "configs" / "smoke.toml")
        self.assertNotIn("phoneme_durations_path", config)

    def test_loader_filters_non_north_american_rows(self):
        rows = [
            {"corpus_id":"c","session_id":"us","target_child_age_months":10,"utterance_order":1,"ipa":"a","text":"a","phonemes":["a"],"language":"English","dialect":"North American English"},
            {"corpus_id":"c","session_id":"uk","target_child_age_months":10,"utterance_order":1,"ipa":"a","text":"a","phonemes":["a"],"language":"English","dialect":"British English"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual([s.session_id for s in load_ipa_childes(path)], ["us"])


if __name__ == "__main__":
    unittest.main()
