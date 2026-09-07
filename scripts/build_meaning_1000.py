from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from devlm.stimulus_construction.constants import ALL_COLUMNS
from devlm.stimulus_construction.linguistics import build_ipa_lexicon
from devlm.stimulus_construction.resources import count_childes_words
from devlm.stimulus_construction.selection import _word_feature, select_word_candidates


COUNTS = {
    "train": {"high_association": 145, "low_association": 145, "unrelated": 290},
    "validation": {"high_association": 40, "low_association": 40, "unrelated": 80},
    "test": {"high_association": 65, "low_association": 65, "unrelated": 130},
}
FEATURES = (
    "mean_subtlex_zipf", "within_pair_subtlex_difference",
    "mean_orthographic_length", "mean_phoneme_count", "mean_syllable_count",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            values = [row.get(column, "") for column in columns]
            while values and values[-1] in (None, ""):
                values.pop()
            writer.writerow(values)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    if pooled == 0:
        return 0.0 if np.isclose(a.mean(), b.mean()) else float("inf")
    return float((a.mean() - b.mean()) / pooled)


def audit(rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(rows) != 1000 or len({row["item_id"] for row in rows}) != 1000:
        raise RuntimeError("Meaning-1000 must contain 1,000 unique item IDs")
    if len({(row["word1"], row["word2"]) for row in rows}) != 1000:
        raise RuntimeError("Meaning-1000 contains duplicate cue-target pairs")
    tables: list[dict[str, object]] = []
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[str(row["split"])].update((str(row["word1"]), str(row["word2"])))
        if int(row["overlap_ds003604_exact"]) or int(row["overlap_ds003604_critical_vocab"]):
            raise RuntimeError(f"{row['item_id']}: ds003604 exclusion failed")
        if not json.loads(str(row["word1_ipa"])) or not json.loads(str(row["word2_ipa"])):
            raise RuntimeError(f"{row['item_id']}: missing final IPA")
        condition, fsg, bsg = str(row["condition"]), float(row["FSG"]), float(row["BSG"])
        if condition == "high_association" and not 0.40 <= fsg <= 0.85:
            raise RuntimeError(f"{row['item_id']}: high FSG={fsg}")
        if condition == "low_association" and not 0.14 <= fsg <= 0.39:
            raise RuntimeError(f"{row['item_id']}: low FSG={fsg}")
        if condition == "unrelated" and (fsg != 0 or bsg != 0):
            raise RuntimeError(f"{row['item_id']}: unrelated pair is not zero in both directions")
    for a, b in combinations(COUNTS, 2):
        overlap = by_split[a] & by_split[b]
        if overlap:
            raise RuntimeError(f"Critical vocabulary crosses {a}/{b}: {sorted(overlap)[:10]}")
    for split, expected in COUNTS.items():
        subset = [row for row in rows if row["split"] == split]
        observed = Counter(str(row["condition"]) for row in subset)
        if observed != Counter(expected):
            raise RuntimeError(f"{split}: condition counts {observed} != {expected}")
        labels = Counter(int(row["binary_label"]) for row in subset)
        if labels != {0: len(subset) // 2, 1: len(subset) // 2}:
            raise RuntimeError(f"{split}: labels are not 50/50: {labels}")
    for split in ("all", *COUNTS):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        for condition_a, condition_b in combinations(COUNTS["train"], 2):
            a_rows = [row for row in subset if row["condition"] == condition_a]
            b_rows = [row for row in subset if row["condition"] == condition_b]
            for feature in FEATURES:
                a = np.asarray([_word_feature(row, feature) for row in a_rows])
                b = np.asarray([_word_feature(row, feature) for row in b_rows])
                value = smd(a, b)
                tables.append({
                    "split": split, "condition_a": condition_a, "condition_b": condition_b,
                    "variable": feature, "n_a": len(a), "n_b": len(b),
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)),
                    "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                    "min_a": float(a.min()), "max_a": float(a.max()),
                    "min_b": float(b.min()), "max_b": float(b.max()),
                    "smd": value, "ks_statistic": float(ks_2samp(a, b).statistic),
                })
    failures = [row for row in tables if not np.isfinite(row["smd"]) or abs(row["smd"]) >= 0.10]
    if failures:
        worst = sorted(failures, key=lambda row: abs(row["smd"]), reverse=True)[0]
        raise RuntimeError(f"Meaning-1000 matching failed: {worst}")
    summary = {
        "schema_version": 1, "task": "Meaning", "candidate_count": 3100,
        "final_count": 1000, "split_counts": dict(Counter(row["split"] for row in rows)),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "label_counts_by_split": {
            split: dict(Counter(int(row["binary_label"]) for row in rows if row["split"] == split))
            for split in COUNTS
        },
        "ds003604_exact_overlap_count": 0, "ds003604_critical_vocabulary_overlap_count": 0,
        "cross_split_critical_vocabulary_overlap_count": 0,
        "max_absolute_smd": max(abs(row["smd"]) for row in tables), "smd_target": 0.10,
        "items_with_unseen_phase1_content_words": sum(
            int(float(row["childes_count_word1"]) == 0 or float(row["childes_count_word2"]) == 0)
            for row in rows
        ),
        "selection_seed": 1729,
        "selection_method": "single-seed mixed-integer constrained randomization from recorded 3100-item pool",
        "model_outputs_used_for_selection": False, "child_behavior_used_for_selection": False,
    }
    return summary, tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--original-semantics", type=Path, required=True)
    parser.add_argument("--childes", type=Path, required=True)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = read_tsv(args.candidates)
    originals = []
    for row in read_tsv(args.original_semantics):
        try:
            tuple(float(row[column]) for column in (
                "word_A_length", "word_B_length", "word_A_number_ phonemes",
                "word_B_number_ phonemes", "word_A_number_syllables", "word_B_number_syllables",
            ))
        except (KeyError, ValueError):
            continue
        originals.append(row)
    selected = select_word_candidates(
        "Meaning", candidates, originals, condition_counts=COUNTS,
    )
    split_order = {name: index for index, name in enumerate(COUNTS)}
    condition_order = {name: index for index, name in enumerate(COUNTS["train"])}
    selected.sort(key=lambda row: (
        split_order[str(row["split"])], condition_order[str(row["condition"])], str(row["item_id"]),
    ))
    childes_counts, childes_rows = count_childes_words(args.childes)
    original_pairs = {
        (row["word_A"].strip().lower(), row["word_B"].strip().lower()) for row in originals
    }
    original_vocabulary = {word for pair in original_pairs for word in pair}
    selected_words = {str(row[key]) for row in selected for key in ("word1", "word2")}
    ipa, ipa_failures = build_ipa_lexicon(selected_words, args.feature_table)
    if ipa_failures or set(ipa) != selected_words:
        raise RuntimeError(f"Selected words lack Phase 1-compatible IPA: {sorted(ipa_failures)[:20]}")
    for index, row in enumerate(selected, 1):
        row["item_id"] = f"MEANING_{index:04d}"
        row["adaptation_eligibility"] = "model_blind_selected"
        row["qc_pass"] = 1
        row["childes_count_word1"] = childes_counts[str(row["word1"])]
        row["childes_count_word2"] = childes_counts[str(row["word2"])]
        row["word1_ipa"] = json.dumps(ipa[str(row["word1"])], ensure_ascii=False)
        row["word2_ipa"] = json.dumps(ipa[str(row["word2"])], ensure_ascii=False)
        row["cue"], row["target"] = row["word1"], row["word2"]
        row["FSG"], row["BSG"] = row["association_forward"], row["association_backward"]
        pair = (str(row["word1"]).lower(), str(row["word2"]).lower())
        row["overlap_ds003604_exact"] = int(pair in original_pairs)
        row["overlap_ds003604_critical_vocab"] = int(bool(set(pair) & original_vocabulary))
        unseen = [word for word, count in (
            (row["word1"], row["childes_count_word1"]), (row["word2"], row["childes_count_word2"]),
        ) if count == 0]
        row["qc_notes"] = "unseen_phase1:" + ",".join(map(str, unseen)) if unseen else ""
    summary, tables = audit(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "final_all.tsv", selected, ALL_COLUMNS)
    for split in COUNTS:
        write_tsv(args.output_dir / f"{split}.tsv", [row for row in selected if row["split"] == split], ALL_COLUMNS)
    write_tsv(args.output_dir / "qc_tables.tsv", tables, list(tables[0]))
    final_path = args.output_dir / "final_all.tsv"
    summary["final_all_sha256"] = hashlib.sha256(final_path.read_bytes()).hexdigest()
    summary["ipa_childes_rows_scanned"] = childes_rows
    (args.output_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
