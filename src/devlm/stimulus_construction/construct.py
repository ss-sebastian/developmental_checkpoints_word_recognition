from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .constants import (
    ALL_COLUMNS, CONDITION_COUNTS, GRAMMATICALITY_PAIR_EXCLUSIONS,
    GRAMMATICALITY_PAIR_REPLACEMENTS,
    GRAMMATICALITY_SUPPLEMENTAL_PAIRS,
    MEANING_BOUNDS, OFFENSIVE_DENYLIST,
    PLAUSIBILITY_BOUNDS, POSITIVE_CONDITIONS, SEED, SPLIT_SIZES,
    SENTENCE_PAIR_EXCLUSIONS, TASK_PREFIX, TEMPLATE_FAMILIES,
)
from .linguistics import (
    build_ipa_lexicon, finiteness_violation_sentence, grammatical_sentence,
    inflect, is_base_verb, is_singular_noun_lemma, plurality_violation_sentence,
)
from .matching import LexicalPair, re_pair, select_matched_pair_sets, stable_key
from .resources import (
    Association, DsExclusions, Frequency, PartOfSpeech, Pronunciation,
    count_childes_words, load_cmudict, load_ds003604, load_subtlex,
    load_subtlex_pos, load_usf, normalize_sentence, sha256_file, verb_lemma,
    write_json,
)


SPLIT_ORDER = ("train", "validation", "test")


def blank_row() -> dict[str, object]:
    return {column: "" for column in ALL_COLUMNS}


def write_tsv(path: str | Path, rows: list[dict[str, object]], columns: list[str] = ALL_COLUMNS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t",
            extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _frequency_floor(task: str, ds: DsExclusions, frequencies: dict[str, Frequency]) -> float:
    words = ds.task_vocabulary[task]
    values = [frequencies[word].zipf for word in words if word in frequencies]
    if not values:
        raise RuntimeError(f"No ds003604 {task} words were found in SUBTLEX-US")
    return min(values)


def eligible_words(
    task: str,
    cmu: dict[str, Pronunciation],
    frequencies: dict[str, Frequency],
    lowercase_subtlex: set[str],
    ipa: dict[str, tuple[str, ...]],
    ds: DsExclusions,
    one_syllable: bool = False,
) -> set[str]:
    floor = _frequency_floor(task, ds, frequencies)
    return {
        word for word in cmu.keys() & frequencies.keys() & lowercase_subtlex & ipa.keys()
        if word not in ds.task_vocabulary[task]
        and word not in OFFENSIVE_DENYLIST
        and frequencies[word].zipf >= floor
        and (not one_syllable or cmu[word].syllables == 1)
    }


def _pair_features(row: dict[str, object], pair: LexicalPair, frequencies: dict[str, Frequency], cmu: dict[str, Pronunciation]) -> None:
    a, b = pair.word1, pair.word2
    row.update({
        "word1": a,
        "word2": b,
        "word1_subtlex": frequencies[a].zipf,
        "word2_subtlex": frequencies[b].zipf,
        "mean_content_subtlex": (frequencies[a].zipf + frequencies[b].zipf) / 2,
        "word1_n_phonemes": len(cmu[a].phones),
        "word2_n_phonemes": len(cmu[b].phones),
        "word1_n_syllables": cmu[a].syllables,
        "word2_n_syllables": cmu[b].syllables,
        "association_forward": "" if pair.association_forward is None else pair.association_forward,
        "association_backward": "" if pair.association_backward is None else pair.association_backward,
        "source_record_id": pair.source_record_id,
    })


def _sound_pools(words: set[str], cmu: dict[str, Pronunciation]) -> dict[str, list[LexicalPair]]:
    rhyme_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    onset_groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        pronunciation = cmu[word]
        rhyme_groups[pronunciation.rime].append(word)
        onset_groups[pronunciation.phones[0]].append(word)

    pools: dict[str, dict[tuple[str, str], LexicalPair]] = {key: {} for key in ("rhyme", "onset", "unrelated")}
    for rime, group in rhyme_groups.items():
        ordered = sorted(group)
        for word1, word2 in combinations(ordered, 2):
            if cmu[word1].onset != cmu[word2].onset:
                pair = LexicalPair(word1, word2, "rhyme", f"cmudict:rime:{'-'.join(rime)}")
                pools["rhyme"][(word1, word2)] = pair

    for first_phone, group in onset_groups.items():
        ordered = sorted(group, key=lambda word: stable_key("sound-onset-order", word))
        size = len(ordered)
        for index, word1 in enumerate(ordered):
            for offset in range(1, min(size, 17)):
                word2 = ordered[(index + offset) % size]
                a, b = sorted((word1, word2))
                if a != b and cmu[a].rime != cmu[b].rime:
                    pools["onset"][(a, b)] = LexicalPair(a, b, "onset", f"cmudict:first:{first_phone}")

    ordered = sorted(words, key=lambda word: stable_key("sound-unrelated-word", word))
    size = len(ordered)
    for index, word1 in enumerate(ordered):
        for offset in range(1, min(size, 401)):
            word2 = ordered[(index + offset * 37) % size]
            a, b = sorted((word1, word2))
            if a == b:
                continue
            pa, pb = cmu[a], cmu[b]
            if not set(pa.phones).intersection(pb.phones) and pa.phones[0] != pb.phones[0] and pa.rime != pb.rime:
                pools["unrelated"][(a, b)] = LexicalPair(a, b, "unrelated", "cmudict:formal-zero-shared-phone")
                if len(pools["unrelated"]) >= 4000:
                    break
        if len(pools["unrelated"]) >= 4000:
            break
    result = {name: list(values.values()) for name, values in pools.items()}
    if sum(map(len, result.values())) < 3100:
        raise RuntimeError(f"Sound candidate pool has only {sum(map(len, result.values())):,} candidates; need >=3,100")
    return result


def _association_pools(
    associations: list[Association],
    words: set[str],
    bounds: dict[str, tuple[float, float]],
    require_verb_noun: bool = False,
    parts_of_speech: dict[str, PartOfSpeech] | None = None,
) -> dict[str, list[LexicalPair]]:
    pools = {name: [] for name in bounds}
    for item in associations:
        if item.cue not in words or item.target not in words or not item.normed_target:
            continue
        if require_verb_noun:
            if (item.cue, item.target) in SENTENCE_PAIR_EXCLUSIONS:
                continue
            if parts_of_speech is None:
                raise ValueError("Official SUBTLEX-US PoS information is required for sentence tasks")
            cue_pos = parts_of_speech.get(item.cue)
            target_pos = parts_of_speech.get(item.target)
            if (
                cue_pos is None or target_pos is None
                or cue_pos.dominant != "Verb" or target_pos.dominant != "Noun"
                or not is_base_verb(item.cue)
                or not is_singular_noun_lemma(item.target)
            ):
                continue
            try:
                verb_forms = []
                for tag in ("VBG", "VBZ", "VBD"):
                    verb_forms.append(inflect(item.cue, tag))
                inflect(item.target, "NNS")
            except ValueError:
                continue
            if verb_forms[-1] == item.cue:
                # A bare-form/past-form contrast cannot instantiate an observable
                # ds003604-style finiteness omission for invariant spellings.
                continue
        for name, (low, high) in bounds.items():
            if low <= item.fsg <= high:
                pools[name].append(LexicalPair(
                    item.cue, item.target, name, f"USF:{item.source_record_id}", item.fsg, item.bsg,
                ))
                break
    return pools


def _zero_association_pool(
    words1: set[str],
    words2: set[str],
    normed_cues: set[str],
    association_lookup: set[tuple[str, str]],
    relation: str,
    limit: int = 5000,
) -> list[LexicalPair]:
    left = sorted(words1 & normed_cues, key=lambda word: stable_key("zero-left", relation, word))
    right = sorted(words2 & normed_cues, key=lambda word: stable_key("zero-right", relation, word))
    result: list[LexicalPair] = []
    seen: set[tuple[str, str]] = set()
    for index, word1 in enumerate(left):
        for offset in range(1, min(len(right), 101)):
            word2 = right[(index * 17 + offset * 43) % len(right)]
            pair = (word1, word2)
            if word1 == word2 or pair in seen or pair in association_lookup or (word2, word1) in association_lookup:
                continue
            seen.add(pair)
            result.append(LexicalPair(word1, word2, relation, "USF:absent-both-directions", 0.0, 0.0))
            if len(result) == limit:
                return result
    return result


def _formal_sound(pair: LexicalPair, cmu: dict[str, Pronunciation]) -> bool:
    a, b = cmu[pair.word1], cmu[pair.word2]
    if pair.relation == "rhyme":
        return a.onset != b.onset and a.rime == b.rime and pair.word1 != pair.word2
    if pair.relation == "onset":
        return a.phones[0] == b.phones[0] and a.rime != b.rime
    return a.phones[0] != b.phones[0] and a.rime != b.rime and not set(a.phones).intersection(b.phones)


def _select_word_task(
    task: str,
    positive_a: list[LexicalPair],
    positive_b: list[LexicalPair],
    frequencies: dict[str, Frequency],
    cmu: dict[str, Pronunciation],
    association_lookup: set[tuple[str, str]],
) -> list[tuple[str, LexicalPair]]:
    used: set[str] = set()
    selected: list[tuple[str, LexicalPair]] = []
    for split in ("validation", "test", "train"):
        condition_names = list(CONDITION_COUNTS[task][split])
        name_a, name_b = condition_names[:2]
        n = CONDITION_COUNTS[task][split][name_a]
        set_a, set_b = select_matched_pair_sets(
            positive_a, positive_b, n, used, frequencies, cmu, f"{task}:{split}",
        )
        positives = set_a + set_b
        if task == "Sound":
            negatives = re_pair(
                positives,
                lambda a, b: _formal_sound(LexicalPair(a, b, "unrelated", ""), cmu),
                frequencies, cmu, f"{task}:{split}",
            )
        else:
            negatives = re_pair(
                positives,
                lambda a, b: a != b and (a, b) not in association_lookup and (b, a) not in association_lookup,
                frequencies, cmu, f"{task}:{split}",
            )
        selected.extend((split, pair) for pair in set_a + set_b + negatives)
    return selected


def _template_schedule(count: int, label: str) -> list[str]:
    indexed = [(TEMPLATE_FAMILIES[index % 4], index) for index in range(count)]
    return [value for value, index in sorted(indexed, key=lambda item: stable_key(label, item[0], str(item[1])))]


def _expand_pairs(pairs: list[LexicalPair], count: int, label: str) -> list[tuple[LexicalPair, int]]:
    """Expand vetted lexical pairs into distinct deterministic template variants."""
    if not pairs:
        raise RuntimeError(f"{label}: no lexical pairs available")
    ordered = sorted(pairs, key=lambda pair: stable_key(label, pair.word1, pair.word2))
    expanded: list[tuple[LexicalPair, int]] = []
    occurrences: Counter[tuple[str, str]] = Counter()
    cursor = 0
    while len(expanded) < count:
        pair = ordered[cursor % len(ordered)]
        key = (pair.word1, pair.word2)
        variant = occurrences[key]
        if variant >= 72:
            raise RuntimeError(f"{label}: exhausted 72 distinct template/argument variants for {key}")
        occurrences[key] += 1
        expanded.append((pair, variant))
        cursor += 1
    return expanded


def _expand_pairs_to_phone_mean(
    pairs: list[LexicalPair], count: int, target_mean: float,
    cmu: dict[str, Pronunciation], frequencies: dict[str, Frequency], label: str,
) -> list[tuple[LexicalPair, int]]:
    """Deterministically weight lexical pairs to a morphology-adjusted phone target."""
    base = [pair for pair, _ in _expand_pairs(pairs, count, label + ":base")]
    phone_count = lambda pair: len(cmu[pair.word1].phones) + len(cmu[pair.word2].phones)
    target_sum = round(target_mean * count)
    current = sum(phone_count(pair) for pair in base)
    ordered = sorted(pairs, key=lambda pair: (phone_count(pair), stable_key(label, pair.word1, pair.word2)))
    pair_frequency = lambda pair: (frequencies[pair.word1].zipf + frequencies[pair.word2].zipf) / 2
    while current < target_sum:
        best: tuple[tuple[float, float, str], int, LexicalPair] | None = None
        counts = Counter(base)
        for index, old in enumerate(base):
            for new in reversed(ordered):
                if counts[new] >= 36:
                    continue
                delta = phone_count(new) - phone_count(old)
                if delta <= 0:
                    continue
                score = (
                    float(abs(target_sum - (current + delta))),
                    abs(pair_frequency(new) - pair_frequency(old)),
                    stable_key(label + ":swap", old.word1, new.word1),
                )
                if best is None or score < best[0]:
                    best = (score, index, new)
        if best is None or best[0][0] >= abs(target_sum - current):
            break
        _, index, new = best
        current += phone_count(new) - phone_count(base[index])
        base[index] = new
    occurrences: Counter[tuple[str, str]] = Counter()
    result = []
    for pair in base:
        key = (pair.word1, pair.word2)
        result.append((pair, occurrences[key]))
        occurrences[key] += 1
    return result


def _adjust_pair_frequency_mean(
    pair_variants: list[tuple[LexicalPair, int]], pool: list[LexicalPair],
    target_mean: float, cmu: dict[str, Pronunciation],
    frequencies: dict[str, Frequency], label: str,
) -> list[tuple[LexicalPair, int]]:
    """Improve frequency balance while changing aggregate base-phone count minimally."""
    pairs = [pair for pair, _ in pair_variants]
    pair_frequency = lambda pair: (frequencies[pair.word1].zipf + frequencies[pair.word2].zipf) / 2
    pair_phones = lambda pair: len(cmu[pair.word1].phones) + len(cmu[pair.word2].phones)
    target_sum = target_mean * len(pairs)
    current = sum(pair_frequency(pair) for pair in pairs)
    counts = Counter(pairs)
    ordered = sorted(pool, key=lambda pair: stable_key(label, pair.word1, pair.word2))
    for _ in range(len(pairs) * 4):
        best = None
        for index, old in enumerate(pairs):
            for new in ordered:
                if counts[new] >= 72:
                    continue
                updated = current + pair_frequency(new) - pair_frequency(old)
                if abs(target_sum - updated) >= abs(target_sum - current):
                    continue
                score = (
                    abs(target_sum - updated), abs(pair_phones(new) - pair_phones(old)),
                    stable_key(label + ":frequency-swap", old.word1, new.word1),
                )
                if best is None or score < best[0]:
                    best = (score, index, old, new, updated)
        if best is None:
            break
        _, index, old, new, current = best
        counts[old] -= 1
        counts[new] += 1
        pairs[index] = new
    occurrences: Counter[tuple[str, str]] = Counter()
    result = []
    for pair in pairs:
        key = (pair.word1, pair.word2)
        result.append((pair, occurrences[key]))
        occurrences[key] += 1
    return result


def _sentence_variant(variant: int) -> tuple[str, int]:
    return TEMPLATE_FAMILIES[variant % len(TEMPLATE_FAMILIES)], variant


def _subject(index: int) -> str:
    return ("she", "he", "they")[index % 3]


def _number(index: int) -> str:
    return ("one", "two", "three", "four", "five", "six")[index % 6]


def _sentence_row(
    task: str,
    split: str,
    condition: str,
    pair: LexicalPair,
    template: str,
    index: int,
    frequencies: dict[str, Frequency],
    cmu: dict[str, Pronunciation],
) -> dict[str, object]:
    subject, number = _subject(index // 4), _number(index // 12)
    sentence, verb_surface, object_surface, region = grammatical_sentence(
        pair.word1, pair.word2, template, subject, number,
    )
    row = blank_row()
    row.update({
        "task": task, "condition": condition, "binary_label": int(condition in POSITIVE_CONDITIONS),
        "split": split, "source_resource": "USF+ds003604-template-abstraction",
        "source_record_id": pair.source_record_id, "sentence": sentence,
        "template_id": template, "template_family": template, "critical_word": object_surface,
        "critical_region": region + " " + object_surface, "association_forward": pair.association_forward,
        "association_backward": pair.association_backward, "mean_content_subtlex": (
            frequencies[pair.word1].zipf + frequencies[pair.word2].zipf
        ) / 2, "subject": subject, "verb_lemma": pair.word1, "verb_surface": verb_surface,
        "object_lemma": pair.word2, "object_surface": object_surface, "number_word": number,
        "negation": int(template == "do_negation"), "verb_object_FSG": pair.association_forward,
        "verb_object_BSG": pair.association_backward, "congruence_level": condition,
    })
    return row


def _select_plausibility(
    strong: list[LexicalPair], weak: list[LexicalPair], frequencies: dict[str, Frequency],
    cmu: dict[str, Pronunciation], association_lookup: set[tuple[str, str]],
) -> list[dict[str, object]]:
    used: set[str] = set()
    rows: list[dict[str, object]] = []
    for split in ("validation", "test", "train"):
        n = CONDITION_COUNTS["Plausibility"][split]["strong_congruence"]
        lexical_count = math.ceil(n / 4)
        base_strong, base_weak = select_matched_pair_sets(
            strong, weak, lexical_count, used, frequencies, cmu, f"Plausibility:{split}",
            reservoir_factor_a=1, reservoir_factor_b=8,
            feature_weights=(3.0, 2.0, 1.0, 1.0, 1.5, 1.0, 1.0),
        )
        base_negatives = re_pair(
            base_strong + base_weak,
            lambda verb, obj: verb != obj and (verb, obj) not in association_lookup and (obj, verb) not in association_lookup,
            frequencies, cmu, f"Plausibility:{split}",
        )
        expanded = (
            ("strong_congruence", _expand_pairs(base_strong, n, f"Plausibility:{split}:strong")),
            ("weak_congruence", _expand_pairs(base_weak, n, f"Plausibility:{split}:weak")),
            ("incongruent", _expand_pairs(base_negatives, 2 * n, f"Plausibility:{split}:zero")),
        )
        for condition, pair_variants in expanded:
            for pair, variant in pair_variants:
                template, index = _sentence_variant(variant)
                rows.append(_sentence_row(
                    "Plausibility", split, condition, pair, template, index, frequencies, cmu,
                ))
    return rows


def _select_grammar(
    strong: list[LexicalPair], frequencies: dict[str, Frequency], cmu: dict[str, Pronunciation],
) -> list[dict[str, object]]:
    used: set[str] = set()
    rows: list[dict[str, object]] = []
    # Allocate the smaller splits first.  The lexical-disjointness constraint is
    # across splits (not within a split), so this deterministic order avoids a
    # large greedy train allocation consuming the only words available to a
    # smaller held-out split.
    for split in ("validation", "test", "train"):
        n_positive = CONDITION_COUNTS["Grammaticality"][split]["grammatical"]
        base_count = math.ceil(n_positive / 12)
        reservoir: list[LexicalPair] = []
        split_words: set[str] = set()
        for pair in sorted(strong, key=lambda value: stable_key("grammar", split, value.word1, value.word2)):
            if pair.word1 in used or pair.word2 in used:
                continue
            try:
                for tag in ("VBG", "VBZ", "VBD"):
                    inflect(pair.word1, tag)
                inflect(pair.word2, "NNS")
            except ValueError:
                continue
            reservoir.append(pair)
            split_words.update((pair.word1, pair.word2))
            if len(reservoir) == base_count:
                break
        if len(reservoir) < base_count:
            raise RuntimeError(
                f"Grammaticality:{split}: only {len(reservoir)} strong pairs with "
                f"split-exclusive vocabulary; need {base_count}"
            )
        used.update(split_words)
        grammatical_base = _expand_pairs(reservoir, n_positive, f"Grammar:{split}:grammatical-base")
        grammatical_base_phone_mean = sum(
            len(cmu[pair.word1].phones) + len(cmu[pair.word2].phones)
            for pair, _ in grammatical_base
        ) / len(grammatical_base)
        grammatical_variants = _expand_pairs_to_phone_mean(
            reservoir, n_positive, grammatical_base_phone_mean + 0.6,
            cmu, frequencies, f"Grammar:{split}:grammatical",
        )
        grammatical_structures: dict[tuple[str, str], set[int]] = defaultdict(set)
        for position, (pair, _) in enumerate(grammatical_variants):
            index = position % 72
            key = (pair.word1, pair.word2)
            while index in grammatical_structures[key]:
                index = (index + 1) % 72
            grammatical_structures[key].add(index)
            template, index = _sentence_variant(index)
            row = _sentence_row("Grammaticality", split, "grammatical", pair, template, index, frequencies, cmu)
            row.update({"grammar_subtype": "grammatical", "violation_type": "none"})
            rows.append(row)

        n_finite = CONDITION_COUNTS["Grammaticality"][split]["finiteness_violation"]
        grammatical_phone_mean = sum(
            len(cmu[pair.word1].phones) + len(cmu[pair.word2].phones)
            for pair, _ in grammatical_variants
        ) / len(grammatical_variants)
        grammatical_frequency_mean = sum(
            (frequencies[pair.word1].zipf + frequencies[pair.word2].zipf) / 2
            for pair, _ in grammatical_variants
        ) / len(grammatical_variants)
        finite_pairs = _expand_pairs_to_phone_mean(
            reservoir, n_finite, grammatical_phone_mean + 0.55,
            cmu, frequencies, f"Grammar:{split}:finite",
        )
        plural_pairs = _expand_pairs_to_phone_mean(
            reservoir, n_finite, grammatical_phone_mean + 0.70,
            cmu, frequencies, f"Grammar:{split}:plural",
        )
        finite_pairs = _adjust_pair_frequency_mean(
            finite_pairs, reservoir, grammatical_frequency_mean,
            cmu, frequencies, f"Grammar:{split}:finite",
        )
        plural_pairs = _adjust_pair_frequency_mean(
            plural_pairs, reservoir, grammatical_frequency_mean,
            cmu, frequencies, f"Grammar:{split}:plural",
        )
        conditions = (
            ("finiteness_violation", finite_pairs),
            ("plurality_violation", plural_pairs),
        )
        for condition, pair_variants in conditions:
            used_structures: dict[tuple[str, str], set[int]] = defaultdict(set)
            for position, (pair, _) in enumerate(pair_variants):
                variant = position % 72
                key = (pair.word1, pair.word2)
                while variant in used_structures[key]:
                    variant = (variant + 1) % 72
                used_structures[key].add(variant)
                template = TEMPLATE_FAMILIES[variant % 4]
                subject = _subject(variant // 4)
                if condition == "finiteness_violation":
                    number = _number(variant // 12)
                    add_error = variant % 5 == 0
                    sentence, verb_surface, object_surface, expected, observed, mechanism = finiteness_violation_sentence(
                        pair.word1, pair.word2, template, subject, number, add_error,
                    )
                    violation_location = "verb_phrase"
                else:
                    cycle = variant // 12
                    add_error = cycle == 0
                    violation_number = ("two", "three", "four", "five", "six")[max(cycle - 1, 0)]
                    sentence, verb_surface, object_surface, number, expected, region = plurality_violation_sentence(
                        pair.word1, pair.word2, template, subject, violation_number, add_error,
                    )
                    observed, mechanism, violation_location = object_surface, "added" if add_error else "omitted", "object_number"
                row = _sentence_row("Grammaticality", split, condition, pair, template, variant, frequencies, cmu)
                row.update({
                    "sentence": sentence, "binary_label": 0, "verb_surface": verb_surface,
                    "object_surface": object_surface, "critical_word": observed,
                    "subject": subject, "number_word": number,
                    "grammar_subtype": condition, "violation_type": mechanism,
                    "violation_location": violation_location, "expected_form": expected,
                    "observed_form": observed,
                })
                if condition == "plurality_violation":
                    row["number_word"] = number
                rows.append(row)
    return rows


def _assign_ids(rows: list[dict[str, object]], task: str) -> list[dict[str, object]]:
    order = {split: index for index, split in enumerate(SPLIT_ORDER)}
    rows = sorted(rows, key=lambda row: (
        order[str(row["split"])], str(row["condition"]), str(row.get("word1") or row.get("sentence")),
        str(row.get("word2", "")),
    ))
    for index, row in enumerate(rows, 1):
        row["item_id"] = f"{TASK_PREFIX[task]}_{index:04d}"
    return rows


def _add_ipa_and_counts(
    rows: list[dict[str, object]], ipa: dict[str, tuple[str, ...]], childes: Counter[str],
) -> None:
    sentence_words = {
        word.lower() for row in rows for word in str(row.get("sentence", "")).split() if word
    }
    missing = sentence_words - ipa.keys()
    if missing:
        raise RuntimeError(f"Final sentence words lack Phase 1-compatible IPA: {sorted(missing)[:20]}")
    for row in rows:
        if row["word1"]:
            word1, word2 = str(row["word1"]), str(row["word2"])
            row["word1_ipa"] = json.dumps(ipa[word1], ensure_ascii=False)
            row["word2_ipa"] = json.dumps(ipa[word2], ensure_ascii=False)
            row["childes_count_word1"] = childes[word1]
            row["childes_count_word2"] = childes[word2]
        else:
            words = [word.lower() for word in str(row["sentence"]).split()]
            phones = tuple(phone for word in words for phone in ipa[word])
            row["sentence_ipa"] = json.dumps(phones, ensure_ascii=False)
            row["sentence_n_phonemes"] = len(phones)
            verb, obj = str(row["verb_lemma"]), str(row["object_lemma"])
            row["childes_count_verb"] = childes[verb]
            row["childes_count_object"] = childes[obj]
            row["content_word_childes_counts"] = json.dumps({verb: childes[verb], obj: childes[obj]}, sort_keys=True)


def _candidate_rows_from_pairs(
    task: str, pools: dict[str, list[LexicalPair]], frequencies: dict[str, Frequency],
    cmu: dict[str, Pronunciation],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, pool in pools.items():
        quota = 5 * sum(CONDITION_COUNTS[task][split][condition] for split in SPLIT_ORDER)
        ordered = sorted(pool, key=lambda value: stable_key("candidate", task, condition, value.word1, value.word2))
        if len(ordered) < quota:
            raise RuntimeError(f"{task}:{condition} candidate count {len(ordered)}; need 5x={quota}")
        for pair in ordered[:quota]:
            row = blank_row()
            row.update({
                "item_id": f"CAND_{TASK_PREFIX[task]}_{len(rows)+1:06d}", "task": task,
                "condition": condition, "binary_label": int(condition in POSITIVE_CONDITIONS),
                "stimulus_origin": "plus",
                "adaptation_eligibility": "candidate_pending_human_review",
                "source_resource": "cmusphinx/cmudict" if task == "Sound" else "USF Appendix A",
            })
            _pair_features(row, pair, frequencies, cmu)
            rows.append(row)
    if len(rows) < 3100:
        raise RuntimeError(f"{task} candidate file has only {len(rows):,} rows; need >=3,100")
    return rows


def _rows_from_word_selection(
    task: str, selected: list[tuple[str, LexicalPair]], frequencies: dict[str, Frequency],
    cmu: dict[str, Pronunciation], ds: DsExclusions,
) -> list[dict[str, object]]:
    rows = []
    for split, pair in selected:
        row = blank_row()
        row.update({
            "task": task, "condition": pair.relation, "binary_label": int(pair.relation in POSITIVE_CONDITIONS),
            "split": split, "source_resource": "cmusphinx/cmudict" if task == "Sound" else "USF Appendix A",
            "overlap_ds003604_exact": int((pair.word1, pair.word2) in ds.word_pairs[task]),
            "overlap_ds003604_critical_vocab": int(pair.word1 in ds.task_vocabulary[task] or pair.word2 in ds.task_vocabulary[task]),
        })
        _pair_features(row, pair, frequencies, cmu)
        if task == "Sound":
            p1, p2 = cmu[pair.word1], cmu[pair.word2]
            row.update({
                "shared_onset": p1.phones[0] if p1.phones[0] == p2.phones[0] else "",
                "shared_rime": " ".join(p1.rime) if p1.rime == p2.rime else "",
                "shared_phoneme_count": len(set(p1.phones).intersection(p2.phones)),
                "phonological_relation_verified": int(_formal_sound(pair, cmu)),
            })
        else:
            row.update({
                "cue": pair.word1, "target": pair.word2, "FSG": pair.association_forward,
                "BSG": pair.association_backward,
            })
        rows.append(row)
    return _assign_ids(rows, task)


def _add_overlap_flags(rows: list[dict[str, object]], ds: DsExclusions) -> None:
    for row in rows:
        task = str(row["task"])
        sentence = str(row["sentence"]).lower()
        pair = (str(row["verb_lemma"]), str(row["object_lemma"]))
        row["overlap_ds003604_exact"] = int(sentence in ds.sentences[task])
        row["overlap_ds003604_critical_vocab"] = int(
            pair in ds.critical_pairs[task]
            or pair[0] in ds.task_vocabulary[task]
            or pair[1] in ds.task_vocabulary[task]
        )


def _original_content_verb(row: dict[str, str]) -> str:
    """Return the exact lexical-verb surface from a ds003604 metadata row."""
    auxiliaries = {"", "n/a", "is", "are", "was", "were", "be", "does", "do", "did", "not"}
    surfaces = [(row.get(f"verb{i}") or "").strip() for i in (1, 2, 3)]
    return next((surface for surface in reversed(surfaces) if surface.lower() not in auxiliaries), "")


def _noun_lemma(surface: str) -> str:
    from lemminflect import getAllLemmas
    lemmas = getAllLemmas(surface.lower()).get("NOUN", ())
    return str(lemmas[0]) if lemmas else surface.lower()


def _original_reference_rows(ds: DsExclusions, task: str) -> list[dict[str, object]]:
    """Reconstruct ds003604 items verbatim as evaluation/reference anchors.

    These rows intentionally retain the original sentence and critical lexical
    surfaces. They are never assigned to an adaptation split automatically.
    """
    condition_maps = {
        "Plausibility": {
            "SP_S": ("strong_congruence", 1),
            "SP_W": ("weak_congruence", 1),
            "SP_I": ("incongruent", 0),
        },
        "Grammaticality": {
            "G_G": ("grammatical", 1),
            "G_F": ("finiteness_violation", 0),
            "G_P": ("plurality_violation", 0),
        },
    }
    template_maps = {
        "be": "be_progressive", "do": "do_negation",
        "3-s": "present_3s", "ed": "simple_past",
    }
    prefix = TASK_PREFIX[task]
    result: list[dict[str, object]] = []
    for index, source in enumerate(ds.source_rows[task], 1):
        trial_type = (source.get("trial_type") or "").strip()
        condition, label = condition_maps[task][trial_type]
        verb_surface = _original_content_verb(source)
        object_surface = (source.get("object") or "").strip()
        sentence = normalize_sentence([
            source.get("carrier_phrase", ""), source.get("subject", ""),
            source.get("verb1", ""), source.get("verb2", ""),
            source.get("verb3", ""), source.get("number", ""),
            source.get("object", ""),
        ])
        row = blank_row()
        row.update({
            "item_id": f"ORIG_{prefix}_{index:04d}",
            "task": task,
            "condition": condition,
            "binary_label": label,
            "stimulus_origin": "ds003604_original",
            "adaptation_eligibility": "reference_only",
            "original_trial_type": trial_type,
            "original_stimulus_file": (source.get("stim_file") or "").strip(),
            "source_resource": "OpenNeuro ds003604 official stimulus metadata",
            "source_record_id": f"ds003604:{(source.get('stim_file') or '').strip()}",
            "sentence": sentence,
            "template_id": template_maps.get((source.get("stim_verb_forms") or "").strip(), ""),
            "template_family": template_maps.get((source.get("stim_verb_forms") or "").strip(), ""),
            "subject": (source.get("subject") or "").strip(),
            "verb_lemma": verb_lemma(verb_surface),
            "verb_surface": verb_surface,
            "object_lemma": _noun_lemma(object_surface),
            "object_surface": object_surface,
            "number_word": (source.get("number") or "").strip(),
            "negation": int((source.get("stim_verb_forms") or "").strip() == "do"),
            "critical_word": object_surface,
            "critical_region": f"{(source.get('number') or '').strip()} {object_surface}".strip(),
            "congruence_level": condition if task == "Plausibility" else "",
            "grammar_subtype": condition if task == "Grammaticality" else "",
            "violation_type": (source.get("stim_violation_type") or "none").strip()
            if task == "Grammaticality" else "",
        })
        result.append(row)
    return result


REVIEW_BASE_COLUMNS = [
    "item_id", "stimulus_origin", "adaptation_eligibility",
    "original_trial_type", "original_stimulus_file", "condition",
    "binary_label", "sentence", "template_family", "verb_surface",
    "object_surface", "verb_lemma", "object_lemma", "verb_object_FSG",
    "violation_type", "source_record_id",
]
PLAUSIBILITY_REVIEW_COLUMNS = REVIEW_BASE_COLUMNS + [
    "auto_expected_makes_sense_yes_no",
    "auto_structure_complete_yes_no",
    "auto_usf_condition_pass_yes_no",
    "human_sentence_well_formed_yes_no",
    "human_sentence_makes_sense_yes_no",
    "human_notes_optional",
]
GRAMMATICALITY_REVIEW_COLUMNS = REVIEW_BASE_COLUMNS + [
    "auto_expected_grammatical_yes_no",
    "auto_structure_complete_yes_no",
    "auto_error_annotation_pass_yes_no",
    "human_corrected_event_makes_sense_yes_no",
    "human_sentence_grammatical_yes_no",
    "human_no_extra_errors_yes_no",
    "human_notes_optional",
]


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _prepare_review_rows(rows: list[dict[str, object]], task: str) -> list[dict[str, object]]:
    """Add deterministic checks while leaving only judgment calls for humans."""
    prepared: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        complete = all(str(row.get(field, "")).strip() for field in (
            "sentence", "template_family", "subject", "verb_surface",
            "object_surface", "number_word",
        ))
        row["auto_structure_complete_yes_no"] = _yes_no(complete)
        if task == "Plausibility":
            condition = str(row["condition"])
            row["auto_expected_makes_sense_yes_no"] = _yes_no(condition != "incongruent")
            if row.get("stimulus_origin") == "ds003604_original":
                condition_pass = condition in {
                    "strong_congruence", "weak_congruence", "incongruent",
                }
            else:
                fsg = float(row.get("verb_object_FSG") or 0.0)
                bsg = float(row.get("association_backward") or 0.0)
                condition_pass = (
                    condition == "strong_congruence" and PLAUSIBILITY_BOUNDS[condition][0] <= fsg <= PLAUSIBILITY_BOUNDS[condition][1]
                    or condition == "weak_congruence" and PLAUSIBILITY_BOUNDS[condition][0] <= fsg <= PLAUSIBILITY_BOUNDS[condition][1]
                    or condition == "incongruent" and fsg == 0.0 and bsg == 0.0
                )
            row["auto_usf_condition_pass_yes_no"] = _yes_no(condition_pass)
            human_columns = PLAUSIBILITY_REVIEW_COLUMNS[-3:]
        else:
            condition = str(row["condition"])
            row["auto_expected_grammatical_yes_no"] = _yes_no(condition == "grammatical")
            violation = str(row.get("violation_type") or "").strip().lower()
            if row.get("stimulus_origin") == "ds003604_original":
                annotation_pass = condition in {
                    "grammatical", "finiteness_violation", "plurality_violation",
                }
            else:
                annotation_pass = (
                    condition == "grammatical" and violation == "none"
                    or condition == "finiteness_violation" and violation in {"added", "omitted", "substituted"}
                    or condition == "plurality_violation" and violation in {"added", "omitted"}
                )
            row["auto_error_annotation_pass_yes_no"] = _yes_no(annotation_pass)
            human_columns = GRAMMATICALITY_REVIEW_COLUMNS[-4:]
        for column in human_columns:
            row.setdefault(column, "")
        prepared.append(row)
    return prepared


def _write_review_tsv(
    path: Path, rows: list[dict[str, object]], columns: list[str],
) -> None:
    """Write review material without erasing prior human yes/no entries."""
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = {
                row["item_id"]: row for row in csv.DictReader(handle, delimiter="\t")
                if row.get("item_id")
            }
    merged = []
    for source in rows:
        row = dict(source)
        old = existing.get(str(row["item_id"]), {})
        same_stimulus = (
            old.get("sentence") == str(row.get("sentence", ""))
            and old.get("source_record_id") == str(row.get("source_record_id", ""))
        )
        if same_stimulus:
            for column in columns:
                if column.startswith("human_") and old.get(column, "").strip():
                    row[column] = old[column]
        merged.append(row)
    write_tsv(path, merged, columns)


def _load_completed_sentence_reviews(
    output_root: Path,
) -> dict[str, list[dict[str, str]]]:
    """Require complete, condition-consistent human review before final release."""
    completed: dict[str, list[dict[str, str]]] = {}
    for task in ("Plausibility", "Grammaticality"):
        path = output_root / task.lower() / "candidate_review.tsv"
        if not path.exists():
            raise RuntimeError(f"{task}: missing completed human review file: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 3100:
            raise RuntimeError(f"{task}: reviewed candidate count is {len(rows)}; required 3,100")
        for row in rows:
            if task == "Plausibility":
                valid = (
                    row["human_sentence_well_formed_yes_no"] == "yes"
                    and row["human_sentence_makes_sense_yes_no"]
                    == row["auto_expected_makes_sense_yes_no"]
                )
            else:
                valid = (
                    row["human_corrected_event_makes_sense_yes_no"] == "yes"
                    and row["human_sentence_grammatical_yes_no"]
                    == row["auto_expected_grammatical_yes_no"]
                    and row["human_no_extra_errors_yes_no"] == "yes"
                )
            if not valid:
                raise RuntimeError(
                    f"{task}:{row['item_id']}: incomplete or condition-inconsistent human review"
                )
        completed[task] = rows
    return completed


def _assert_sentence_final_is_reviewed(
    final: dict[str, list[dict[str, object]]],
    reviews: dict[str, list[dict[str, str]]],
) -> None:
    """Prevent release of generated sentence variants that humans never reviewed."""
    for task in ("Plausibility", "Grammaticality"):
        approved = {
            (row["condition"], row["sentence"], row["verb_lemma"], row["object_lemma"])
            for row in reviews[task]
        }
        unreviewed = [
            row for row in final[task]
            if (
                str(row["condition"]), str(row["sentence"]),
                str(row["verb_lemma"]), str(row["object_lemma"]),
            ) not in approved
        ]
        if unreviewed:
            examples = [
                (row["condition"], row["sentence"])
                for row in unreviewed[:5]
            ]
            raise RuntimeError(
                f"{task}: {len(unreviewed)} proposed final stimuli were not human-reviewed; "
                f"examples={examples}"
            )


def _original_design_alignment_rows(
    final: dict[str, list[dict[str, object]]], ds: DsExclusions,
) -> list[dict[str, object]]:
    """Create a direct, auditable comparison with the non-control source design."""
    condition_maps = {
        "Sound": {"P_R": "rhyme", "P_O": "onset", "P_U": "unrelated"},
        "Meaning": {"S_H": "high_association", "S_L": "low_association", "S_U": "unrelated"},
        "Plausibility": {"SP_S": "strong_congruence", "SP_W": "weak_congruence", "SP_I": "incongruent"},
        "Grammaticality": {"G_G": "grammatical", "G_F": "finiteness_violation", "G_P": "plurality_violation"},
    }
    template_map = {
        "be": "be_progressive", "do": "do_negation",
        "3-s": "present_3s", "ed": "simple_past",
    }
    output: list[dict[str, object]] = []
    for task, mapping in condition_maps.items():
        for source_condition, condition in mapping.items():
            originals = [row for row in ds.source_rows[task] if row["trial_type"] == source_condition]
            selected = [row for row in final[task] if row["condition"] == condition]
            if task in {"Sound", "Meaning"}:
                continuous = {
                    "mean_orthographic_length": (
                        lambda row: (float(row["word_A_length"]) + float(row["word_B_length"])) / 2,
                        lambda row: (len(str(row["word1"])) + len(str(row["word2"]))) / 2,
                    ),
                    "mean_phoneme_count": (
                        lambda row: (float(row["word_A_number_ phonemes"]) + float(row["word_B_number_ phonemes"])) / 2,
                        lambda row: (float(row["word1_n_phonemes"]) + float(row["word2_n_phonemes"])) / 2,
                    ),
                    "mean_syllable_count": (
                        lambda row: (float(row["word_A_number_syllables"]) + float(row["word_B_number_syllables"])) / 2,
                        lambda row: (float(row["word1_n_syllables"]) + float(row["word2_n_syllables"])) / 2,
                    ),
                }
                for variable, (source_value, final_value) in continuous.items():
                    original_mean = sum(map(source_value, originals)) / len(originals)
                    final_mean = sum(map(final_value, selected)) / len(selected)
                    output.append({
                        "task": task, "condition": condition, "comparison_type": "continuous_mean",
                        "variable": variable, "category": "", "original_n": len(originals),
                        "final_n": len(selected), "original_value": original_mean,
                        "final_value": final_mean, "absolute_difference": abs(final_mean - original_mean),
                        "note": "Directly comparable source-design variable",
                    })
            else:
                extractors = {
                    "template_family": (
                        lambda row: template_map[row["stim_verb_forms"].strip()],
                        lambda row: str(row["template_family"]),
                    ),
                    "subject": (
                        lambda row: row["subject"].strip().lower(), lambda row: str(row["subject"]),
                    ),
                    "number_word": (
                        lambda row: row["number"].strip().lower(), lambda row: str(row["number_word"]),
                    ),
                }
                for variable, (source_value, final_value) in extractors.items():
                    categories = sorted(set(map(source_value, originals)) | set(map(final_value, selected)))
                    for category in categories:
                        original_proportion = sum(source_value(row) == category for row in originals) / len(originals)
                        final_proportion = sum(final_value(row) == category for row in selected) / len(selected)
                        output.append({
                            "task": task, "condition": condition, "comparison_type": "categorical_proportion",
                            "variable": variable, "category": category, "original_n": len(originals),
                            "final_n": len(selected), "original_value": original_proportion,
                            "final_value": final_proportion,
                            "absolute_difference": abs(final_proportion - original_proportion),
                            "note": "Targeted to ds003604 within integer/lexical-disjointness tolerance",
                        })
    return output


def construct(config: dict[str, str | int]) -> dict[str, list[dict[str, object]]]:
    source_root = Path(str(config["source_root"]))
    output_root = Path(str(config["output_root"]))
    existing = [output_root / task.lower() / "final_all.tsv" for task in CONDITION_COUNTS]
    if any(path.exists() for path in existing):
        raise RuntimeError("Fixed stimulus manifests already exist; refusing to regenerate or replace their splits")
    cmu, variant_counts = load_cmudict(source_root / "cmudict/cmudict.dict")
    frequencies, lowercase_subtlex = load_subtlex(source_root / "subtlex_us/SUBTLEXusfrequencyabove1.xls")
    parts_of_speech = load_subtlex_pos(
        source_root / "subtlex_us/SUBTLEX-US frequency list with PoS and Zipf information.xlsx"
    )
    associations, normed_cues = load_usf(source_root / "usf")
    ds = load_ds003604(source_root / "ds003604")
    childes, childes_rows = count_childes_words(source_root / "ipa_childes/Eng-NA_processed.csv")

    preliminary = cmu.keys() & frequencies.keys() & lowercase_subtlex
    ipa, ipa_failures = build_ipa_lexicon(set(preliminary), source_root / "phase1/ipa_feature_mapping.json")
    task_words = {
        task: eligible_words(task, cmu, frequencies, lowercase_subtlex, ipa, ds, task == "Sound")
        for task in CONDITION_COUNTS
    }
    association_lookup = {(item.cue, item.target) for item in associations}

    sound_pools = _sound_pools(task_words["Sound"], cmu)
    meaning_pools = _association_pools(associations, task_words["Meaning"], MEANING_BOUNDS)
    plaus_pools = _association_pools(
        associations, task_words["Plausibility"], PLAUSIBILITY_BOUNDS, True, parts_of_speech,
    )
    grammar_pools = _association_pools(
        associations, task_words["Grammaticality"],
        {"strong_congruence": PLAUSIBILITY_BOUNDS["strong_congruence"]}, True,
        parts_of_speech,
    )
    grammar_candidate_pools = {
        condition: list(pairs) for condition, pairs in grammar_pools.items()
    }
    grammar_supplemental = _association_pools(
        associations, task_words["Grammaticality"],
        {"supplemental": (0.02, PLAUSIBILITY_BOUNDS["strong_congruence"][0] - 1e-9)},
        True, parts_of_speech,
    )["supplemental"]
    supplemental_lookup = {
        (pair.word1, pair.word2): pair for pair in grammar_supplemental
        if (pair.word1, pair.word2) in GRAMMATICALITY_SUPPLEMENTAL_PAIRS
    }
    missing_supplemental = GRAMMATICALITY_SUPPLEMENTAL_PAIRS - supplemental_lookup.keys()
    if missing_supplemental:
        raise RuntimeError(
            f"Missing source-backed Grammaticality supplemental pairs: {sorted(missing_supplemental)}"
        )
    grammar_pools["strong_congruence"].extend(supplemental_lookup.values())
    grammar_pools = {
        condition: [
            pair for pair in pairs
            if (pair.word1, pair.word2) not in GRAMMATICALITY_PAIR_EXCLUSIONS
        ]
        for condition, pairs in grammar_pools.items()
    }
    meaning_pools["unrelated"] = _zero_association_pool(
        task_words["Meaning"], task_words["Meaning"], normed_cues, association_lookup, "unrelated",
    )
    plaus_verbs = {pair.word1 for values in plaus_pools.values() for pair in values}
    plaus_objects = {pair.word2 for values in plaus_pools.values() for pair in values}
    plaus_pools["incongruent"] = _zero_association_pool(
        plaus_verbs, plaus_objects, normed_cues, association_lookup, "incongruent",
    )
    plaus_pools["incongruent"] = [
        pair for pair in plaus_pools["incongruent"]
        if (pair.word1, pair.word2) not in SENTENCE_PAIR_EXCLUSIONS
    ]

    candidates = {
        "Sound": _candidate_rows_from_pairs("Sound", sound_pools, frequencies, cmu),
        "Meaning": _candidate_rows_from_pairs("Meaning", meaning_pools, frequencies, cmu),
    }

    from .selection import select_word_candidates
    final: dict[str, list[dict[str, object]]] = {}
    for task in ("Sound", "Meaning"):
        selected_rows = select_word_candidates(task, candidates[task], ds.source_rows[task])
        selected_pairs = []
        for row in selected_rows:
            forward = row["association_forward"]
            backward = row["association_backward"]
            selected_pairs.append((
                str(row["split"]),
                LexicalPair(
                    str(row["word1"]), str(row["word2"]), str(row["condition"]),
                    str(row["source_record_id"]),
                    None if forward == "" else float(forward),
                    None if backward == "" else float(backward),
                ),
            ))
        final[task] = _rows_from_word_selection(task, selected_pairs, frequencies, cmu, ds)

    # Each sentence subtype receives exactly a 5x candidate pool before matching.
    for task, pools in (("Plausibility", plaus_pools), ("Grammaticality", grammar_pools)):
        rows: list[dict[str, object]] = []
        candidate_quotas = {
            condition: 5 * sum(CONDITION_COUNTS[task][split][condition] for split in SPLIT_ORDER)
            for condition in CONDITION_COUNTS[task]["train"]
        }
        for condition, quota in candidate_quotas.items():
            pair_pool = (
                pools[condition] if task == "Plausibility"
                else grammar_candidate_pools["strong_congruence"]
            )
            made = 0
            for source_pair in sorted(pair_pool, key=lambda value: stable_key("sent-candidate", task, condition, value.word1, value.word2)):
                pair = source_pair
                if task == "Grammaticality":
                    replacement = GRAMMATICALITY_PAIR_REPLACEMENTS.get(
                        (source_pair.word1, source_pair.word2)
                    )
                    if replacement is not None:
                        pair = supplemental_lookup[replacement]
                for variant in range(72):
                    template, index = _sentence_variant(variant)
                    if condition in {"strong_congruence", "weak_congruence", "incongruent", "grammatical"}:
                        row = _sentence_row(task, "", condition, pair, template, index, frequencies, cmu)
                        if task == "Grammaticality":
                            row.update({"grammar_subtype": "grammatical", "violation_type": "none"})
                    elif condition == "finiteness_violation":
                        template = TEMPLATE_FAMILIES[variant % 4]
                        subject = _subject(variant // 4)
                        number = _number(variant // 12)
                        sentence, verb_surface, object_surface, expected, observed, mechanism = finiteness_violation_sentence(
                            pair.word1, pair.word2, template, subject, number, variant % 5 == 0,
                        )
                        row = _sentence_row(task, "", condition, pair, template, index, frequencies, cmu)
                        row.update({"sentence": sentence, "verb_surface": verb_surface, "object_surface": object_surface,
                                    "subject": subject, "number_word": number,
                                    "grammar_subtype": condition, "violation_type": mechanism,
                                    "violation_location": "verb_phrase", "expected_form": expected, "observed_form": observed})
                    else:
                        template = TEMPLATE_FAMILIES[variant % 4]
                        subject = _subject(variant // 4)
                        cycle = variant // 12
                        add_error = cycle == 0
                        sentence, verb_surface, object_surface, number, expected, region = plurality_violation_sentence(
                            pair.word1, pair.word2, template, subject,
                            ("two", "three", "four", "five", "six")[max(cycle - 1, 0)],
                            add_error,
                        )
                        row = _sentence_row(task, "", condition, pair, template, index, frequencies, cmu)
                        row.update({"sentence": sentence, "verb_surface": verb_surface, "object_surface": object_surface,
                                    "subject": subject, "number_word": number, "grammar_subtype": condition,
                                    "violation_type": "added" if add_error else "omitted",
                                    "violation_location": "object_number", "expected_form": expected,
                                    "observed_form": object_surface})
                    rows.append(row)
                    made += 1
                    if made == quota:
                        break
                if made == quota:
                    break
            if made < quota:
                raise RuntimeError(f"{task}:{condition} candidate count {made}; need 5x={quota}")
        for index, row in enumerate(rows, 1):
            row["item_id"] = f"CAND_{TASK_PREFIX[task]}_{index:06d}"
            row["stimulus_origin"] = "plus"
            row["adaptation_eligibility"] = "candidate_pending_human_review"
        candidates[task] = rows

    # Candidate-first construction: the optimizer below needs the same exact
    # Phase 1-compatible sentence phoneme counts that downstream QC will use.
    surface_words = {
        word.lower()
        for rows in (final["Sound"], final["Meaning"], candidates["Plausibility"], candidates["Grammaticality"])
        for row in rows
        for word in (
            [str(row["word1"]), str(row["word2"])]
            if row["word1"] else str(row["sentence"]).split()
        )
        if word
    }
    missing_surfaces = surface_words - ipa.keys()
    if missing_surfaces:
        extra_ipa, extra_failures = build_ipa_lexicon(
            missing_surfaces, source_root / "phase1/ipa_feature_mapping.json",
        )
        ipa.update(extra_ipa)
        ipa_failures.update(extra_failures)
    _add_ipa_and_counts(final["Sound"], ipa, childes)
    _add_ipa_and_counts(final["Meaning"], ipa, childes)
    _add_ipa_and_counts(candidates["Plausibility"], ipa, childes)
    _add_ipa_and_counts(candidates["Grammaticality"], ipa, childes)

    original_references = {
        task: _original_reference_rows(ds, task)
        for task in ("Plausibility", "Grammaticality")
    }

    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    source_manifest = {
        "construction_date": date.today().isoformat(),
        "seed": SEED,
        "source_only": True,
        "checkpoint_files_loaded": 0,
        "child_behavior_files_loaded": 0,
        "model_outputs_used_for_selection": False,
        "resources": [
            {
                "path": str(path.relative_to(source_root)), "sha256": sha256_file(path),
                "bytes": path.stat().st_size, "download_date": date.today().isoformat(),
            }
            for path in source_files
        ],
        "resource_versions": {
            "cmudict": "cmusphinx/cmudict commit 74790861f652b15e4ac49015a90074ad62a27690",
            "USF": "Nelson, McEvoy & Schreiber Appendix A official files",
            "SUBTLEX-US": (
                "Brysbaert & New 2009 frequency table plus Brysbaert, New & Keuleers 2012 "
                "official PoS/Zipf extension (OSF 7wx25, version 1)"
            ),
            "ds003604": "OpenNeuroDatasets/ds003604 commit 0aaf6a478e605a92e626ab9a49775da614a5b9e4",
            "IPA-CHILDES": (
                "phonemetransformers/IPA-CHILDES revision "
                "2f4e63f61b3e4b11b470a511d6abcd18d1e3ad9e, Eng-NA/processed.csv"
            ),
        },
        "parsed_counts": {
            "cmudict_unambiguous_words": len(cmu), "cmudict_all_roots": len(variant_counts),
            "usf_associations": len(associations), "usf_normed_cues": len(normed_cues),
            "subtlex_words": len(frequencies), "subtlex_pos_words": len(parts_of_speech),
            "ipa_childes_rows": childes_rows,
            "phase1_ipa_compatible_base_words": len(ipa), "ipa_failures": len(ipa_failures),
        },
        "filtering_counts": {
            "eligible_words_by_task": {task: len(words) for task, words in task_words.items()},
            "sound_pair_pool_by_condition": {key: len(value) for key, value in sound_pools.items()},
            "meaning_pair_pool_by_condition": {key: len(value) for key, value in meaning_pools.items()},
            "plausibility_pair_pool_by_condition": {key: len(value) for key, value in plaus_pools.items()},
            "grammaticality_pair_pool_by_condition": {key: len(value) for key, value in grammar_pools.items()},
            "grammaticality_supplemental_pairs": [
                f"{verb}-{obj}" for verb, obj in sorted(GRAMMATICALITY_SUPPLEMENTAL_PAIRS)
            ],
            "grammaticality_pair_replacements": {
                f"{old_verb}-{old_obj}": f"{new_verb}-{new_obj}"
                for (old_verb, old_obj), (new_verb, new_obj)
                in sorted(GRAMMATICALITY_PAIR_REPLACEMENTS.items())
            },
            "ds003604_excluded_vocabulary_by_task": {
                task: len(words) for task, words in ds.task_vocabulary.items()
            },
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "sources/source_manifest.json", source_manifest)
    exclusion_rows = []
    for task in CONDITION_COUNTS:
        for word in sorted(ds.task_vocabulary[task]):
            exclusion_rows.append({"task": task, "word": word, "exclusion_type": "critical_vocabulary"})
    write_tsv(output_root / "sources/openneuro_exclusion_vocabulary.tsv", exclusion_rows, ["task", "word", "exclusion_type"])
    write_tsv(
        output_root / "sources/sentence_pair_exclusions.tsv",
        [
            {
                "task": "Plausibility", "verb_lemma": verb,
                "object_lemma": obj, "reason": reason,
            }
            for (verb, obj), reason in sorted(SENTENCE_PAIR_EXCLUSIONS.items())
        ] + [
            {
                "task": "Grammaticality", "verb_lemma": verb,
                "object_lemma": obj, "reason": reason,
            }
            for (verb, obj), reason in sorted(GRAMMATICALITY_PAIR_EXCLUSIONS.items())
        ],
        ["task", "verb_lemma", "object_lemma", "reason"],
    )

    # Candidate pools are retained even if final QC later identifies an unsatisfied constraint.
    for task, rows in candidates.items():
        write_tsv(output_root / task.lower() / "candidates.tsv", rows)

    # USF provides association strength, and the official SUBTLEX-US extension
    # provides coarse PoS. Neither resource provides verb subcategorization or
    # argument-structure acceptability. Consequently, direct-object templates
    # cannot be certified as grammatical for every candidate without an
    # additional user-approved resource or completed human review. The project
    # specification requires stopping instead of silently treating association
    # as transitivity.
    sentence_constraint = {
        "status": "blocked_before_final_selection",
        "candidate_counts": {task: len(rows) for task, rows in candidates.items()},
        "completed_final_counts": {task: 0 for task in candidates},
        "original_reference_counts": {
            task: len(rows) for task, rows in original_references.items()
        },
        "stimulus_origin_policy": {
            "ds003604_original": (
                "Exact original sentences and critical verb-object surfaces are retained as "
                "reference_only anchors and are not assigned to adaptation splits."
            ),
            "plus": (
                "Newly generated candidate stimuli; eligibility remains pending human review."
            ),
        },
        "human_review_policy": {
            "automatic_columns": (
                "Expected response, structural completeness, USF condition range, and "
                "grammatical-error annotation are generated by code."
            ),
            "plausibility_yes_no_columns": PLAUSIBILITY_REVIEW_COLUMNS[-3:-1],
            "grammaticality_yes_no_columns": GRAMMATICALITY_REVIEW_COLUMNS[-4:-1],
            "optional_notes_column": "human_notes_optional",
            "allowed_yes_no_values": ["yes", "no"],
        },
        "human_confirmed_sentence_pair_exclusions": {
            **{
                f"Plausibility:{verb}-{obj}": reason
                for (verb, obj), reason in sorted(SENTENCE_PAIR_EXCLUSIONS.items())
            },
            **{
                f"Grammaticality:{verb}-{obj}": reason
                for (verb, obj), reason in sorted(GRAMMATICALITY_PAIR_EXCLUSIONS.items())
            },
        },
        "constraint": (
            "The allowed sources do not contain verb valency/subcategorization or "
            "sentence-level acceptability judgments. USF FSG plus SUBTLEX-US dominant "
            "PoS cannot guarantee that generated verb-object sentences are grammatical "
            "and plausible."
        ),
        "required_resolution": (
            "User-approved verb-valency/acceptability resource, or completed human review "
            "of the sentence candidate pools followed by deterministic rematching."
        ),
        "training_executed": False,
        "checkpoint_files_loaded": 0,
        "child_behavior_files_loaded": 0,
    }
    (output_root / "combined").mkdir(parents=True, exist_ok=True)
    write_json(output_root / "combined/construction_manifest.json", sentence_constraint)
    for task in ("Plausibility", "Grammaticality"):
        review_columns = (
            PLAUSIBILITY_REVIEW_COLUMNS if task == "Plausibility"
            else GRAMMATICALITY_REVIEW_COLUMNS
        )
        candidate_reviews = _prepare_review_rows(candidates[task], task)
        original_reviews = _prepare_review_rows(original_references[task], task)
        _write_review_tsv(
            output_root / task.lower() / "candidate_review.tsv",
            candidate_reviews, review_columns,
        )
        _write_review_tsv(
            output_root / task.lower() / "original_reference.tsv",
            original_reviews, review_columns,
        )
        _write_review_tsv(
            output_root / task.lower() / "review_all.tsv",
            original_reviews + candidate_reviews, review_columns,
        )
    completed_reviews = _load_completed_sentence_reviews(output_root)
    from .selection import select_reviewed_sentence_candidates
    for task in ("Plausibility", "Grammaticality"):
        approved_item_ids = {row["item_id"] for row in completed_reviews[task]}
        final[task] = _assign_ids(
            select_reviewed_sentence_candidates(
                task, candidates[task], approved_item_ids, original_references[task],
            ),
            task,
        )
        _add_overlap_flags(final[task], ds)
    _assert_sentence_final_is_reviewed(final, completed_reviews)

    from .qc import validate_and_summarize
    qc = validate_and_summarize(final, candidates, ds, cmu)

    for task, rows in final.items():
        folder = output_root / task.lower()
        write_tsv(folder / "final_all.tsv", rows)
        for split in SPLIT_ORDER:
            write_tsv(folder / f"{split}.tsv", [row for row in rows if row["split"] == split])
        review_columns = (
            ["item_id", "condition", "binary_label", "split", "word1", "word2", "word1_ipa", "word2_ipa", "qc_notes"]
            if task in {"Sound", "Meaning"}
            else ["item_id", "condition", "binary_label", "split", "sentence", "template_family", "verb_lemma", "object_lemma", "violation_type", "qc_notes"]
        )
        write_tsv(folder / "review.tsv", rows, review_columns)
        write_json(folder / "qc_summary.json", qc[task]["summary"])
        write_tsv(folder / "qc_tables.tsv", qc[task]["tables"], [
            "table", "condition_a", "condition_b", "variable", "n_a", "n_b",
            "mean_a", "mean_b", "sd_a", "sd_b", "median_a", "median_b",
            "min_a", "max_a", "min_b", "max_b", "smd", "ks_statistic", "category", "count",
        ])
    combined = [row for task in CONDITION_COUNTS for row in final[task]]
    write_tsv(output_root / "combined/adaptation_all_tasks.tsv", combined)
    split_summary = [
        {
            "task": task, "split": split,
            "n_total": sum(row["split"] == split for row in final[task]),
            "n_label_0": sum(row["split"] == split and int(row["binary_label"]) == 0 for row in final[task]),
            "n_label_1": sum(row["split"] == split and int(row["binary_label"]) == 1 for row in final[task]),
        }
        for task in CONDITION_COUNTS for split in SPLIT_ORDER
    ]
    write_tsv(
        output_root / "combined/split_summary.tsv", split_summary,
        ["task", "split", "n_total", "n_label_0", "n_label_1"],
    )
    write_tsv(
        output_root / "combined/original_design_alignment.tsv",
        _original_design_alignment_rows(final, ds),
        [
            "task", "condition", "comparison_type", "variable", "category",
            "original_n", "final_n", "original_value", "final_value",
            "absolute_difference", "note",
        ],
    )
    construction_manifest = {
        "schema_version": 1,
        "construction_date": date.today().isoformat(),
        "seed": SEED,
        "status": "complete",
        "candidate_counts": {task: len(rows) for task, rows in candidates.items()},
        "final_counts": {task: len(rows) for task, rows in final.items()},
        "split_sizes": SPLIT_SIZES,
        "condition_counts": CONDITION_COUNTS,
        "matching_qc": {task: value["summary"] for task, value in qc.items()},
        "selection_was_model_blind": True,
        "selection_method": (
            "Seeded constrained randomization from the 3,100-row candidate pools; "
            "sentence tasks are restricted to human-reviewed rows."
        ),
        "original_design_matching": (
            "Condition counts are exact. Comparable lexical length/phoneme/syllable measures "
            "and sentence template/subject/number proportions are reported in "
            "combined/original_design_alignment.tsv. SUBTLEX Zipf is balanced internally but "
            "is not treated as numerically identical to the original raw-frequency scale."
        ),
        "checkpoint_files_loaded": 0,
        "child_behavior_files_loaded": 0,
        "training_executed": False,
        "phase1_resources_used": ["IPA feature mapping", "phoneme vocabulary", "IPA-CHILDES word counts"],
        "output_hashes": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in sorted(output_root.rglob("*.tsv"))
        },
    }
    write_json(output_root / "combined/construction_manifest.json", construction_manifest)
    return {"final": final, "candidates": candidates, "source_manifest": source_manifest}
