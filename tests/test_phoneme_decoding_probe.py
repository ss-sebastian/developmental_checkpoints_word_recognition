from __future__ import annotations

import unittest
import inspect

import numpy as np
import torch

from devlm.adaptation.phoneme_decoding_probe import (
    PhonemeDecodingProbeOptions, PhonemeEvent, all_phoneme_events,
    balance_phoneme_events, evaluate_multinomial_probe, extract_phoneme_peak_states,
    fit_multinomial_probe, fit_pca, fit_pooled_pca, fit_pooled_pca_from_scatters,
)
from devlm.adaptation.segmentation_probe import BoundarySession
from devlm.adaptation.segmentation_probe import SegmentationProbeOptions, _make_m00, split_probe_sessions
from devlm.data import Session, Utterance
from devlm.features import FeatureTable
from devlm.model import CausalPhonemeGRU
from devlm.stream import build_session_stream


class PhonemeDecodingProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = FeatureTable(["f1", "f2"], {"a": [1., 0.], "b": [0., 1.]})
        self.vocabulary = {"a": 0, "b": 1}

    def _partitions(self) -> dict[str, list[BoundarySession]]:
        result = {}
        for split in ("train", "validation", "test"):
            utterance = Utterance("c", split, 2., 1, "a b a b", "", ("a", "b", "a", "b"))
            result[split] = [BoundarySession(Session("c", split, 2., (utterance,)), ((0, 0, 0),))]
        return result

    def test_balancing_is_reproducible_and_equal_per_class(self) -> None:
        events = all_phoneme_events(self._partitions())
        options = PhonemeDecodingProbeOptions(min_tokens_per_phoneme=2, max_tokens_per_phoneme=2)
        first, second = balance_phoneme_events(events, options), balance_phoneme_events(events, options)
        self.assertEqual(first, second)
        for split in ("train", "validation", "test"):
            self.assertEqual(sum(x.split == split and x.phoneme == "a" for x in first), 2)
            self.assertEqual(sum(x.split == split and x.phoneme == "b" for x in first), 2)

    def test_peak_is_third_active_frame(self) -> None:
        partitions = self._partitions()
        event = [x for x in all_phoneme_events(partitions) if x.split == "train" and x.phoneme_index == 0]
        torch.manual_seed(3)
        model = CausalPhonemeGRU(2, 4, 1, 2, 0.).eval()
        actual = extract_phoneme_peak_states(model, partitions["train"], event, self.features, self.vocabulary, 0., None, 4, 100, torch.device("cpu"))
        stream = build_session_stream(partitions["train"][0].session, self.features, self.vocabulary, np.random.default_rng(4), noise_sigma=0.)
        with torch.no_grad():
            expected, _ = model.gru(torch.from_numpy(stream.noisy_frames[:3]).unsqueeze(0))
        self.assertTrue(torch.allclose(actual[0], expected[0, -1]))
        # The peak (start+2) is also strictly before the next phoneme's onset.
        self.assertLess(stream.spans[0].start_frame + 2, stream.spans[1].start_frame)

    def test_probe_fit_does_not_need_test_labels_and_pca_is_train_only(self) -> None:
        rng = np.random.default_rng(7)
        train = rng.normal(size=(40, 4)); train[:20, 0] -= 3; train[20:, 0] += 3
        validation = rng.normal(size=(20, 4)); validation[:10, 0] -= 3; validation[10:, 0] += 3
        test = rng.normal(size=(20, 4)); test[:10, 0] -= 3; test[10:, 0] += 3
        probe = fit_multinomial_probe(train, np.array([0] * 20 + [1] * 20), validation, np.array([0] * 10 + [1] * 10), ("a", "b"), PhonemeDecodingProbeOptions(lbfgs_max_iter=50))
        metrics = evaluate_multinomial_probe(probe, test, np.array([0] * 10 + [1] * 10))
        self.assertGreater(metrics["balanced_accuracy"], .9)
        pca = fit_pca(train)
        self.assertEqual(pca.components.shape, (2, 4))
        self.assertEqual(pca.transform(test).shape, (20, 2))
        self.assertNotIn("test", inspect.signature(fit_multinomial_probe).parameters)

    def test_session_split_is_disjoint_and_m00_reconstructs_seed(self) -> None:
        sessions = []
        for i in range(12):
            utterance = Utterance("c", str(i), 2., 1, "a b", "", ("a", "b"))
            sessions.append(BoundarySession(Session("c", str(i), 2., (utterance,)), ((0,),)))
        split = split_probe_sessions(sessions, SegmentationProbeOptions())
        ids = [{x.session.session_id for x in split[name]} for name in ("train", "validation", "test")]
        self.assertTrue(ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]))
        payload = {"config": {"seed": 23, "hidden_size": 4, "num_layers": 1, "dropout": 0.}}
        actual = _make_m00(payload, 2, 2, torch.device("cpu"))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23); expected = CausalPhonemeGRU(2, 4, 1, 2, 0.)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual.state_dict().values(), expected.state_dict().values())))

    def test_online_pooled_pca_matches_train_only_matrix_reference(self) -> None:
        rng = np.random.default_rng(11)
        train_a, train_b = rng.normal(size=(9, 4)), rng.normal(size=(7, 4))
        centers, reference_components, reference_evr = fit_pooled_pca([train_a, train_b])
        scatters = [(x - center).T @ (x - center) for x, center in zip((train_a, train_b), centers, strict=True)]
        components, evr = fit_pooled_pca_from_scatters(centers, scatters, [len(train_a), len(train_b)])
        # Eigenvector signs are arbitrary; absolute loadings and EVR must agree.
        self.assertTrue(np.allclose(np.abs(components), np.abs(reference_components)))
        self.assertTrue(np.allclose(evr, reference_evr))
        self.assertEqual(scatters[0].shape, (4, 4))  # no held-out observations enter the sufficient statistics
