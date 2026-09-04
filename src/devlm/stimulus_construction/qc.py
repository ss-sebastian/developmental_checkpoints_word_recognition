from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import ks_2samp

from .constants import CONDITION_COUNTS, SPLIT_SIZES
from .matching import LexicalPair
from .resources import DsExclusions, Pronunciation


def _numeric(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def _continuous(row: dict[str, object], task: str) -> dict[str, float]:
    if task in {"Sound", "Meaning"}:
        return {
            "mean_subtlex_zipf": _numeric(row, "mean_content_subtlex"),
            "within_pair_subtlex_difference": abs(_numeric(row, "word1_subtlex") - _numeric(row, "word2_subtlex")),
            "mean_orthographic_length": (len(str(row["word1"])) + len(str(row["word2"]))) / 2,
            "mean_phoneme_count": (_numeric(row, "word1_n_phonemes") + _numeric(row, "word2_n_phonemes")) / 2,
            "mean_syllable_count": (_numeric(row, "word1_n_syllables") + _numeric(row, "word2_n_syllables")) / 2,
        }
    return {
        "mean_content_subtlex_zipf": _numeric(row, "mean_content_subtlex"),
        "sentence_word_count": float(len(str(row["sentence"]).split())),
        "sentence_phoneme_count": _numeric(row, "sentence_n_phonemes"),
    }


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    sd_a = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    sd_b = float(b.std(ddof=1)) if len(b) > 1 else 0.0
    pooled = np.sqrt((sd_a * sd_a + sd_b * sd_b) / 2)
    smd = 0.0 if pooled == 0 and np.isclose(a.mean(), b.mean()) else float("inf") if pooled == 0 else float((a.mean() - b.mean()) / pooled)
    return {
        "n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "sd_a": sd_a, "sd_b": sd_b, "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "min_a": float(a.min()), "max_a": float(a.max()), "min_b": float(b.min()), "max_b": float(b.max()),
        "smd": smd, "ks_statistic": float(ks_2samp(a, b).statistic),
    }


def _check_counts(task: str, rows: list[dict[str, object]]) -> None:
    if len(rows) != 620:
        raise RuntimeError(f"{task}: final count is {len(rows)}, required 620")
    if len({str(row["item_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{task}: duplicate item_id")
    identities = [(str(row.get("word1", "")), str(row.get("word2", "")), str(row.get("sentence", ""))) for row in rows]
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        detail = [
            (row["item_id"], row["condition"], row["verb_lemma"], row["object_lemma"], row["sentence"])
            for row in rows if identities[rows.index(row)] in set(duplicates[:1])
        ]
        raise RuntimeError(f"{task}: duplicate stimulus under different IDs: {detail}")
    for split, expected in SPLIT_SIZES.items():
        subset = [row for row in rows if row["split"] == split]
        if len(subset) != expected:
            raise RuntimeError(f"{task}:{split}: {len(subset)} items, required {expected}")
        labels = Counter(int(row["binary_label"]) for row in subset)
        if labels != {0: expected // 2, 1: expected // 2}:
            raise RuntimeError(f"{task}:{split}: label counts {dict(labels)} are not 50/50")
        conditions = Counter(str(row["condition"]) for row in subset)
        if conditions != Counter(CONDITION_COUNTS[task][split]):
            raise RuntimeError(f"{task}:{split}: subtype counts {dict(conditions)} do not match specification")


def _check_lexical_disjointness(task: str, rows: list[dict[str, object]]) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if task in {"Sound", "Meaning"}:
            by_split[str(row["split"])].update((str(row["word1"]), str(row["word2"])))
        else:
            by_split[str(row["split"])].update((str(row["verb_lemma"]), str(row["object_lemma"])))
    for split_a, split_b in combinations(by_split, 2):
        overlap = by_split[split_a].intersection(by_split[split_b])
        if overlap:
            raise RuntimeError(f"{task}: critical vocabulary crosses {split_a}/{split_b}: {sorted(overlap)[:10]}")


def _check_task_rules(task: str, rows: list[dict[str, object]], ds: DsExclusions, cmu: dict[str, Pronunciation]) -> None:
    for row in rows:
        if int(row["overlap_ds003604_exact"]) or int(row["overlap_ds003604_critical_vocab"]):
            raise RuntimeError(f"{task}:{row['item_id']}: ds003604 exclusion failed")
        if task in {"Sound", "Meaning"}:
            for key in ("word1_ipa", "word2_ipa"):
                if not json.loads(str(row[key])):
                    raise RuntimeError(f"{task}:{row['item_id']}: invalid {key}")
        else:
            if not json.loads(str(row["sentence_ipa"])):
                raise RuntimeError(f"{task}:{row['item_id']}: invalid sentence IPA")

        condition = str(row["condition"])
        if task == "Sound":
            a, b = cmu[str(row["word1"])], cmu[str(row["word2"])]
            valid = (
                condition == "rhyme" and a.onset != b.onset and a.rime == b.rime
                or condition == "onset" and a.phones[0] == b.phones[0] and a.rime != b.rime
                or condition == "unrelated" and a.phones[0] != b.phones[0] and a.rime != b.rime and not set(a.phones).intersection(b.phones)
            )
            if not valid:
                raise RuntimeError(f"Sound:{row['item_id']}: formal relation failed")
        elif task == "Meaning":
            value = float(row["FSG"])
            if condition == "high_association" and not 0.40 <= value <= 0.85:
                raise RuntimeError(f"Meaning:{row['item_id']}: high FSG={value}")
            if condition == "low_association" and not 0.14 <= value <= 0.39:
                raise RuntimeError(f"Meaning:{row['item_id']}: low FSG={value}")
            if condition == "unrelated" and (value != 0 or float(row["BSG"]) != 0):
                raise RuntimeError(f"Meaning:{row['item_id']}: unrelated association is not zero both ways")
        elif task == "Plausibility":
            value = float(row["verb_object_FSG"])
            if condition == "strong_congruence" and not 0.28 <= value <= 0.81:
                raise RuntimeError(f"Plausibility:{row['item_id']}: strong FSG={value}")
            if condition == "weak_congruence" and not 0.02 <= value <= 0.19:
                raise RuntimeError(f"Plausibility:{row['item_id']}: weak FSG={value}")
            if condition == "incongruent" and value != 0:
                raise RuntimeError(f"Plausibility:{row['item_id']}: incongruent FSG={value}")
            if row["violation_type"]:
                raise RuntimeError(f"Plausibility:{row['item_id']}: grammatical violation present")
        else:
            expected_label = 1 if condition == "grammatical" else 0
            if int(row["binary_label"]) != expected_label:
                raise RuntimeError(f"Grammaticality:{row['item_id']}: label/condition mismatch")
            if condition == "grammatical" and row["violation_type"] != "none":
                raise RuntimeError(f"Grammaticality:{row['item_id']}: grammatical item has violation")
            if condition != "grammatical" and not row["expected_form"]:
                raise RuntimeError(f"Grammaticality:{row['item_id']}: generated violation lacks expected form")


def validate_and_summarize(
    final: dict[str, list[dict[str, object]]],
    candidates: dict[str, list[dict[str, object]]],
    ds: DsExclusions,
    cmu: dict[str, Pronunciation],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for task, rows in final.items():
        if len(candidates[task]) < 3100:
            raise RuntimeError(f"{task}: candidate count {len(candidates[task])} is below 3,100")
        _check_counts(task, rows)
        _check_lexical_disjointness(task, rows)
        _check_task_rules(task, rows, ds, cmu)

        tables: list[dict[str, object]] = []
        conditions = sorted({str(row["condition"]) for row in rows})
        for condition_a, condition_b in combinations(conditions, 2):
            rows_a = [row for row in rows if row["condition"] == condition_a]
            rows_b = [row for row in rows if row["condition"] == condition_b]
            variables = _continuous(rows_a[0], task)
            for variable in variables:
                a = np.asarray([_continuous(row, task)[variable] for row in rows_a], dtype=float)
                b = np.asarray([_continuous(row, task)[variable] for row in rows_b], dtype=float)
                tables.append({
                    "table": "continuous", "condition_a": condition_a, "condition_b": condition_b,
                    "variable": variable, **_stats(a, b), "category": "", "count": "",
                })
        categorical = ["template_family", "subject", "number_word", "negation"] if task in {"Plausibility", "Grammaticality"} else []
        for variable in categorical:
            counts = Counter((str(row["condition"]), str(row[variable])) for row in rows)
            for (condition, category), count in sorted(counts.items()):
                tables.append({
                    "table": "categorical", "condition_a": condition, "condition_b": "",
                    "variable": variable, "category": category, "count": count,
                })
        finite_smds = [abs(float(row["smd"])) for row in tables if row["table"] == "continuous" and np.isfinite(float(row["smd"]))]
        max_smd = max(finite_smds, default=0.0)
        unseen = sum(
            int((_numeric(row, "childes_count_word1") == 0 or _numeric(row, "childes_count_word2") == 0))
            if task in {"Sound", "Meaning"}
            else int(_numeric(row, "childes_count_verb") == 0 or _numeric(row, "childes_count_object") == 0)
            for row in rows
        )
        for row in rows:
            row["qc_pass"] = 1
            if task in {"Sound", "Meaning"}:
                unseen_words = [word for key, word in (("childes_count_word1", row["word1"]), ("childes_count_word2", row["word2"])) if _numeric(row, key) == 0]
            else:
                unseen_words = [word for key, word in (("childes_count_verb", row["verb_lemma"]), ("childes_count_object", row["object_lemma"])) if _numeric(row, key) == 0]
            row["qc_notes"] = "unseen_phase1:" + ",".join(map(str, unseen_words)) if unseen_words else ""
        result[task] = {
            "summary": {
                "qc_pass": max_smd < 0.10,
                "candidate_count": len(candidates[task]), "final_count": len(rows),
                "split_counts": dict(Counter(str(row["split"]) for row in rows)),
                "condition_counts": dict(Counter(str(row["condition"]) for row in rows)),
                "label_counts_by_split": {
                    split: dict(Counter(int(row["binary_label"]) for row in rows if row["split"] == split))
                    for split in SPLIT_SIZES
                },
                "max_absolute_smd": max_smd, "smd_target": 0.10,
                "items_with_unseen_phase1_content_words": unseen,
                "ds003604_exact_overlap_count": 0, "ds003604_critical_vocabulary_overlap_count": 0,
            },
            "tables": tables,
        }
        if max_smd >= 0.10:
            failures = sorted(
                (row for row in tables if row["table"] == "continuous" and abs(float(row["smd"])) >= 0.10),
                key=lambda row: abs(float(row["smd"])), reverse=True,
            )
            detail = "; ".join(
                f"{row['condition_a']} vs {row['condition_b']} {row['variable']}={float(row['smd']):.3f} "
                f"(means {float(row['mean_a']):.3f}/{float(row['mean_b']):.3f})"
                for row in failures
            )
            raise RuntimeError(
                f"{task}: matching QC failed |SMD|<0.10; {detail}"
            )
    return result
