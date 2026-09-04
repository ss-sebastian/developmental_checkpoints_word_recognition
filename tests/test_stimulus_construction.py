from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from openpyxl import Workbook

from devlm.stimulus_construction.constants import CONDITION_COUNTS, SPLIT_SIZES
from devlm.stimulus_construction.constants import (
    GRAMMATICALITY_PAIR_EXCLUSIONS, SENTENCE_PAIR_EXCLUSIONS,
)
from devlm.stimulus_construction.construct import (
    _original_reference_rows, _prepare_review_rows,
)
from devlm.stimulus_construction.resources import DsExclusions, load_subtlex_pos


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "task_adaptation_stimuli"


class StimulusConstructionTests(unittest.TestCase):
    def test_original_sentence_rows_preserve_exact_sentence_and_pair(self) -> None:
        source = {
            "stim_file": "stereo_PC01.wav", "trial_type": "SP_S",
            "stim_verb_forms": "be", "carrier_phrase": "n/a", "subject": "She",
            "verb1": "is", "verb2": "singing", "verb3": "n/a",
            "number": "one", "object": "song",
        }
        ds = DsExclusions(
            word_pairs={}, task_vocabulary={}, sentences={}, critical_pairs={},
            source_rows={"Plausibility": [source]},
        )
        row = _original_reference_rows(ds, "Plausibility")[0]
        self.assertEqual(row["sentence"], "She is singing one song")
        self.assertEqual((row["verb_surface"], row["object_surface"]), ("singing", "song"))
        self.assertEqual(row["stimulus_origin"], "ds003604_original")
        self.assertEqual(row["adaptation_eligibility"], "reference_only")
        self.assertEqual(row["split"], "")

    def test_review_rows_separate_automatic_checks_from_human_judgments(self) -> None:
        source = {
            "item_id": "CAND_PLAUS_1", "task": "Plausibility",
            "condition": "weak_congruence", "stimulus_origin": "plus",
            "sentence": "She catches one fish", "template_family": "present_3s",
            "subject": "she", "verb_surface": "catches", "object_surface": "fish",
            "number_word": "one", "verb_object_FSG": 0.11,
            "association_backward": 0.0,
        }
        row = _prepare_review_rows([source], "Plausibility")[0]
        self.assertEqual(row["auto_expected_makes_sense_yes_no"], "yes")
        self.assertEqual(row["auto_structure_complete_yes_no"], "yes")
        self.assertEqual(row["auto_usf_condition_pass_yes_no"], "yes")
        self.assertEqual(row["human_sentence_well_formed_yes_no"], "")
        self.assertEqual(row["human_sentence_makes_sense_yes_no"], "")

    def test_requested_final_design_arithmetic(self) -> None:
        self.assertEqual(sum(SPLIT_SIZES.values()), 620)
        for task, splits in CONDITION_COUNTS.items():
            for split, counts in splits.items():
                self.assertEqual(sum(counts.values()), SPLIT_SIZES[split], (task, split))
                positive = sum(
                    count for condition, count in counts.items()
                    if condition in {
                        "rhyme", "onset", "high_association", "low_association",
                        "strong_congruence", "weak_congruence", "grammatical",
                    }
                )
                self.assertEqual(positive, SPLIT_SIZES[split] // 2, (task, split))

    def test_subtlex_pos_loader_prefers_lowercase_and_handles_na(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pos.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "Word", "Dom_PoS_SUBTLEX", "Freq_dom_PoS_SUBTLEX",
                "Percentage_dom_PoS", "All_PoS_SUBTLEX",
            ])
            sheet.append(["Mark", "Name", 9, 0.9, "Name.Noun"])
            sheet.append(["mark", "Verb", 8, 0.8, "Verb.Noun"])
            sheet.append(["unknown", "Unclassified", "#N/A", "#N/A", "Unclassified"])
            workbook.save(path)
            loaded = load_subtlex_pos(path)
        self.assertEqual(loaded["mark"].dominant, "Verb")
        self.assertEqual(loaded["unknown"].dominant_frequency, 0)
        self.assertEqual(loaded["unknown"].dominant_proportion, 0.0)

    def test_manifest_forbids_premature_final_release_or_records_complete_release(self) -> None:
        manifest_path = OUTPUT / "combined" / "construction_manifest.json"
        if not manifest_path.exists():
            self.skipTest("real candidate artifacts are not present in this checkout")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn(manifest["status"], {"blocked_before_final_selection", "complete"})
        self.assertFalse(manifest["training_executed"])
        self.assertEqual(manifest["checkpoint_files_loaded"], 0)
        self.assertEqual(manifest["child_behavior_files_loaded"], 0)
        self.assertEqual(set(manifest["candidate_counts"].values()), {3100})
        for task in ("sound", "meaning", "plausibility", "grammaticality"):
            final_path = OUTPUT / task / "final_all.tsv"
            if manifest["status"] == "complete":
                self.assertTrue(final_path.exists())
                with final_path.open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 620)
            else:
                self.assertFalse(final_path.exists())

    def test_final_sentence_rows_are_from_the_human_reviewed_candidates(self) -> None:
        manifest_path = OUTPUT / "combined" / "construction_manifest.json"
        if not manifest_path.exists():
            self.skipTest("real candidate artifacts are not present in this checkout")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            self.skipTest("final selection has not been released")
        for task in ("plausibility", "grammaticality"):
            with (OUTPUT / task / "candidate_review.tsv").open(encoding="utf-8", newline="") as handle:
                reviewed = {
                    (row["condition"], row["sentence"], row["verb_lemma"], row["object_lemma"])
                    for row in csv.DictReader(handle, delimiter="\t")
                }
            with (OUTPUT / task / "final_all.tsv").open(encoding="utf-8", newline="") as handle:
                final = list(csv.DictReader(handle, delimiter="\t"))
            self.assertTrue(all(
                (row["condition"], row["sentence"], row["verb_lemma"], row["object_lemma"]) in reviewed
                for row in final
            ))

    def test_candidate_pool_counts_and_unique_ids(self) -> None:
        if not (OUTPUT / "sound" / "candidates.tsv").exists():
            self.skipTest("real candidate artifacts are not present in this checkout")
        for task in ("sound", "meaning", "plausibility", "grammaticality"):
            with (OUTPUT / task / "candidates.tsv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 3100, task)
            self.assertEqual(len({row["item_id"] for row in rows}), 3100, task)
            self.assertEqual(Counter(row["split"] for row in rows), {"": 3100})

    def test_human_rejected_sentence_pairs_are_absent_from_candidates(self) -> None:
        path = OUTPUT / "plausibility" / "candidates.tsv"
        if not path.exists():
            self.skipTest("real candidate artifacts are not present in this checkout")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        observed = {(row["verb_lemma"], row["object_lemma"]) for row in rows}
        self.assertFalse(observed.intersection(SENTENCE_PAIR_EXCLUSIONS))

        path = OUTPUT / "grammaticality" / "candidates.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        observed = {(row["verb_lemma"], row["object_lemma"]) for row in rows}
        self.assertFalse(observed.intersection(GRAMMATICALITY_PAIR_EXCLUSIONS))


if __name__ == "__main__":
    unittest.main()
