from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .constants import CONDITION_COUNTS, SEED, SPLIT_SIZES, TEMPLATE_FAMILIES


SPLITS = tuple(SPLIT_SIZES)


def _stable_unit(*parts: str) -> float:
    digest = hashlib.sha256((str(SEED) + "\0" + "\0".join(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _feature(row: dict[str, object], name: str) -> float:
    if name == "mean_content_subtlex_zipf":
        return float(row["mean_content_subtlex"])
    if name == "sentence_word_count":
        return float(len(str(row["sentence"]).split()))
    if name == "sentence_phoneme_count":
        return float(row["sentence_n_phonemes"])
    raise KeyError(name)


def _word_feature(row: dict[str, object], name: str) -> float:
    if name == "mean_subtlex_zipf":
        return float(row["mean_content_subtlex"])
    if name == "within_pair_subtlex_difference":
        return abs(float(row["word1_subtlex"]) - float(row["word2_subtlex"]))
    if name == "mean_orthographic_length":
        return (len(str(row["word1"])) + len(str(row["word2"]))) / 2
    if name == "mean_phoneme_count":
        return (float(row["word1_n_phonemes"]) + float(row["word2_n_phonemes"])) / 2
    if name == "mean_syllable_count":
        return (float(row["word1_n_syllables"]) + float(row["word2_n_syllables"])) / 2
    raise KeyError(name)


def select_word_candidates(
    task: str,
    candidates: list[dict[str, object]],
    original_source_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Select the fixed word-task manifests from their recorded 5x pools."""
    if len(candidates) != 3100:
        raise RuntimeError(f"{task}: {len(candidates)} candidate rows; required 3,100")
    rows = candidates
    n_rows = len(rows)
    words = sorted({str(row["word1"]) for row in rows} | {str(row["word2"]) for row in rows})
    word_index = {word: index for index, word in enumerate(words)}
    n_x = n_rows * len(SPLITS)
    n_variables = n_x + len(words) * len(SPLITS)

    def x_index(row_index: int, split_index: int) -> int:
        return row_index * len(SPLITS) + split_index

    def y_index(word: str, split_index: int) -> int:
        return n_x + word_index[word] * len(SPLITS) + split_index

    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def constraint(coefficients: dict[int, float], low: float, high: float) -> None:
        constraint_index = len(lower)
        for column, value in coefficients.items():
            matrix_rows.append(constraint_index)
            matrix_cols.append(column)
            matrix_data.append(value)
        lower.append(low)
        upper.append(high)

    for row_index in range(n_rows):
        constraint({x_index(row_index, s): 1.0 for s in range(len(SPLITS))}, 0.0, 1.0)
    for split_index, split in enumerate(SPLITS):
        for condition_name, required in CONDITION_COUNTS[task][split].items():
            constraint({
                x_index(row_index, split_index): 1.0
                for row_index, row in enumerate(rows) if row["condition"] == condition_name
            }, required, required)
    for row_index, row in enumerate(rows):
        for split_index in range(len(SPLITS)):
            for word in {str(row["word1"]), str(row["word2"])}:
                constraint(
                    {x_index(row_index, split_index): 1.0, y_index(word, split_index): -1.0},
                    -np.inf, 0.0,
                )
    for word in words:
        constraint({y_index(word, s): 1.0 for s in range(len(SPLITS))}, 0.0, 1.0)

    features = (
        "mean_subtlex_zipf", "within_pair_subtlex_difference",
        "mean_orthographic_length", "mean_phoneme_count", "mean_syllable_count",
    )
    for feature_name in features:
        values = np.asarray([_word_feature(row, feature_name) for row in rows], dtype=float)
        scale = max(float(values.std()), 0.25)
        if feature_name in {"mean_orthographic_length", "mean_phoneme_count", "mean_syllable_count"}:
            source_columns = {
                "mean_orthographic_length": ("word_A_length", "word_B_length"),
                "mean_phoneme_count": ("word_A_number_ phonemes", "word_B_number_ phonemes"),
                "mean_syllable_count": ("word_A_number_syllables", "word_B_number_syllables"),
            }[feature_name]
            target = float(np.mean([
                (float(row[source_columns[0]]) + float(row[source_columns[1]])) / 2
                for row in original_source_rows
            ]))
        else:
            target = float(np.mean([
                np.mean([_word_feature(row, feature_name) for row in rows if row["condition"] == condition])
                for condition in CONDITION_COUNTS[task]["train"]
            ]))
        tolerance = 0.015 * scale
        for condition_name in CONDITION_COUNTS[task]["train"]:
            required = sum(CONDITION_COUNTS[task][split][condition_name] for split in SPLITS)
            coefficients = {
                x_index(row_index, split_index): _word_feature(row, feature_name)
                for row_index, row in enumerate(rows) if row["condition"] == condition_name
                for split_index in range(len(SPLITS))
            }
            constraint(coefficients, required * (target - tolerance), required * (target + tolerance))

    matrix = coo_matrix(
        (matrix_data, (matrix_rows, matrix_cols)), shape=(len(lower), n_variables),
    ).tocsr()
    objective = np.zeros(n_variables, dtype=float)
    for row_index, row in enumerate(rows):
        for split_index, split in enumerate(SPLITS):
            objective[x_index(row_index, split_index)] = 1.0 + 1e-4 * _stable_unit(
                task, str(row["item_id"]), split,
            )
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0, "mip_rel_gap": 1e-4},
    )
    if result.x is None:
        raise RuntimeError(f"{task}: constrained candidate selection failed: {result.message}")
    selected: list[dict[str, object]] = []
    for row_index, source in enumerate(rows):
        for split_index, split in enumerate(SPLITS):
            if result.x[x_index(row_index, split_index)] > 0.5:
                row = dict(source)
                row["split"] = split
                selected.append(row)
    if len(selected) != 620:
        raise RuntimeError(f"{task}: optimizer selected {len(selected)} rows; required 620")
    return selected


def select_reviewed_sentence_candidates(
    task: str,
    candidates: list[dict[str, object]],
    approved_item_ids: set[str],
    original_reference_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select a fixed, balanced split only from human-reviewed candidates.

    This is a seeded mixed-integer constrained randomization.  It enforces the
    requested condition counts, prevents critical vocabulary from crossing
    splits, balances the original sentence-design factors, and tightly matches
    the nuisance-variable means across conditions.
    """
    rows = [row for row in candidates if str(row["item_id"]) in approved_item_ids]
    if len(rows) != 3100:
        raise RuntimeError(f"{task}: {len(rows)} approved candidate rows; required 3,100")

    n_rows = len(rows)
    words = sorted({str(row["verb_lemma"]) for row in rows} | {str(row["object_lemma"]) for row in rows})
    word_index = {word: index for index, word in enumerate(words)}
    n_x = n_rows * len(SPLITS)
    n_y = len(words) * len(SPLITS)
    n_variables = n_x + n_y

    def x_index(row_index: int, split_index: int) -> int:
        return row_index * len(SPLITS) + split_index

    def y_index(word: str, split_index: int) -> int:
        return n_x + word_index[word] * len(SPLITS) + split_index

    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def constraint(coefficients: dict[int, float], low: float, high: float) -> None:
        constraint_index = len(lower)
        for column, value in coefficients.items():
            matrix_rows.append(constraint_index)
            matrix_cols.append(column)
            matrix_data.append(value)
        lower.append(low)
        upper.append(high)

    # A candidate sentence can appear in at most one split.
    for row_index in range(n_rows):
        constraint({x_index(row_index, s): 1.0 for s in range(len(SPLITS))}, 0.0, 1.0)

    # Exact task-condition counts in every split.
    for split_index, split in enumerate(SPLITS):
        for condition_name, required in CONDITION_COUNTS[task][split].items():
            coefficients = {
                x_index(row_index, split_index): 1.0
                for row_index, row in enumerate(rows)
                if row["condition"] == condition_name
            }
            constraint(coefficients, float(required), float(required))

    # Link selected rows to split-level vocabulary indicators; each critical
    # word is then permitted in one split only.
    for row_index, row in enumerate(rows):
        for split_index in range(len(SPLITS)):
            for word in {str(row["verb_lemma"]), str(row["object_lemma"])}:
                constraint(
                    {x_index(row_index, split_index): 1.0, y_index(word, split_index): -1.0},
                    -np.inf, 0.0,
                )
    for word in words:
        constraint({y_index(word, s): 1.0 for s in range(len(SPLITS))}, 0.0, 1.0)

    # Match nuisance-variable means across conditions. The common target is
    # the midpoint of the feasible condition means; a 0.025 pooled-SD band
    # makes pairwise mean differences comfortably smaller than |SMD|=.10.
    features = (
        "mean_content_subtlex_zipf", "sentence_word_count", "sentence_phoneme_count",
    )
    conditions = tuple(CONDITION_COUNTS[task]["train"])
    for feature_name in features:
        values = np.asarray([_feature(row, feature_name) for row in rows], dtype=float)
        scale = max(float(values.std()), 1.0)
        condition_means = [
            float(np.mean([_feature(row, feature_name) for row in rows if row["condition"] == condition_name]))
            for condition_name in conditions
        ]
        target = float(np.mean(condition_means))
        tolerance = 0.015 * scale
        for condition_name in conditions:
            required = sum(CONDITION_COUNTS[task][split][condition_name] for split in SPLITS)
            coefficients = {
                x_index(row_index, split_index): _feature(row, feature_name)
                for row_index, row in enumerate(rows)
                if row["condition"] == condition_name
                for split_index in range(len(SPLITS))
            }
            constraint(
                coefficients,
                required * (target - tolerance),
                required * (target + tolerance),
            )

    # Reproduce the original factorial sentence frame rather than allowing the
    # randomizer to overrepresent a convenient template, subject, or number.
    categorical = {
        "template_family": tuple(TEMPLATE_FAMILIES),
        "subject": ("she", "he", "they"),
        "number_word": ("one", "two", "three", "four", "five", "six"),
    }
    for condition_name in CONDITION_COUNTS[task]["train"]:
        required = sum(CONDITION_COUNTS[task][split][condition_name] for split in SPLITS)
        reference = [row for row in original_reference_rows if row["condition"] == condition_name]
        if not reference:
            raise RuntimeError(f"{task}:{condition_name}: missing original reference rows")
        for field, categories in categorical.items():
            reference_counts = {
                category: sum(str(row[field]).strip().lower() == category for row in reference)
                for category in categories
            }
            raw_expected = {
                category: required * reference_counts[category] / len(reference)
                for category in categories
            }
            floors = {category: int(np.floor(value)) for category, value in raw_expected.items()}
            remaining = required - sum(floors.values())
            ranked = sorted(
                categories,
                key=lambda category: (
                    -(raw_expected[category] - floors[category]),
                    _stable_unit(task, condition_name, field, category),
                ),
            )
            expected_counts = dict(floors)
            for category in ranked[:remaining]:
                expected_counts[category] += 1
            for category, expected in expected_counts.items():
                # Permit a five-item margin for the correlated template,
                # subject, and number cells. This is the smallest tested band
                # that remains feasible together with |SMD|<.10 and strict
                # split-exclusive vocabulary.
                margin = 5
                coefficients = {
                    x_index(row_index, split_index): 1.0
                    for row_index, row in enumerate(rows)
                    if row["condition"] == condition_name and str(row[field]) == category
                    for split_index in range(len(SPLITS))
                }
                constraint(
                    coefficients,
                    max(0, expected - margin),
                    expected + margin,
                )

    matrix = coo_matrix(
        (matrix_data, (matrix_rows, matrix_cols)),
        shape=(len(lower), n_variables),
    ).tocsr()
    objective = np.zeros(n_variables, dtype=float)
    for row_index, row in enumerate(rows):
        for split_index, split in enumerate(SPLITS):
            # The constant term makes every feasible 620-item design nearly
            # equivalent; the seeded perturbation randomizes which valid one
            # is returned without spending minutes proving a meaningless
            # global ordering optimum.
            objective[x_index(row_index, split_index)] = 1.0 + 1e-4 * _stable_unit(
                task, str(row["item_id"]), split,
            )
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0, "mip_rel_gap": 1e-4},
    )
    if result.x is None:
        raise RuntimeError(f"{task}: constrained candidate selection failed: {result.message}")

    selected: list[dict[str, object]] = []
    for row_index, source in enumerate(rows):
        for split_index, split in enumerate(SPLITS):
            if result.x[x_index(row_index, split_index)] > 0.5:
                row = dict(source)
                row["split"] = split
                row["adaptation_eligibility"] = "human_reviewed_selected"
                selected.append(row)
    if len(selected) != 620:
        raise RuntimeError(f"{task}: optimizer selected {len(selected)} rows; required 620")
    return selected
