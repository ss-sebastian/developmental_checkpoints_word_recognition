from __future__ import annotations

import re
from pathlib import Path

from devlm.features import FeatureTable
from devlm.phase2.evidence import normalize_espeak_phones


def build_ipa_lexicon(
    words: set[str], feature_table_path: str | Path, language: str = "en-us",
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Use the project's existing espeak-to-Phase-1 convention without loading a model."""
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    table = FeatureTable.from_json(feature_table_path)
    ordered = sorted(words)
    rendered = phonemize(
        ordered, language=language, backend="espeak",
        separator=Separator(phone=" ", word="", syllable=""), strip=True,
        preserve_punctuation=False, with_stress=False, njobs=1,
    )
    if isinstance(rendered, str):
        rendered = [rendered]
    result: dict[str, tuple[str, ...]] = {}
    failures: dict[str, str] = {}
    for word, value in zip(ordered, rendered, strict=True):
        try:
            phones = normalize_espeak_phones(value)
            if not phones:
                raise ValueError("empty pronunciation")
            for phone in phones:
                table.vector(phone)
            result[word] = phones
        except ValueError as exc:
            failures[word] = str(exc)
    return result, failures


def inflect(lemma: str, tag: str) -> str:
    from lemminflect import getInflection

    forms = getInflection(lemma, tag=tag)
    if not forms:
        raise ValueError(f"Cannot inflect {lemma!r} as {tag}")
    form = str(forms[0]).lower()
    if not re.fullmatch(r"[a-z]+", form):
        raise ValueError(f"Inflection is not one alphabetic word: {lemma!r} -> {form!r}")
    return form


def is_verb(word: str) -> bool:
    from lemminflect import getAllLemmas

    return "VERB" in getAllLemmas(word)


def is_noun(word: str) -> bool:
    from lemminflect import getAllLemmas

    return "NOUN" in getAllLemmas(word)


def is_base_verb(word: str) -> bool:
    from lemminflect import getAllLemmas

    return word in getAllLemmas(word).get("VERB", ())


def is_singular_noun_lemma(word: str) -> bool:
    from lemminflect import getAllLemmas

    return word in getAllLemmas(word).get("NOUN", ())


def noun_surface(noun: str, number_word: str) -> str:
    return noun if number_word == "one" else inflect(noun, "NNS")


def grammatical_sentence(
    verb: str,
    noun: str,
    template_family: str,
    subject: str,
    number_word: str,
) -> tuple[str, str, str, str]:
    obj = noun_surface(noun, number_word)
    if template_family == "be_progressive":
        auxiliary = "are" if subject == "they" else "is"
        surface = inflect(verb, "VBG")
        sentence = f"{subject.capitalize()} {auxiliary} {surface} {number_word} {obj}"
        region = f"{auxiliary} {surface}"
    elif template_family == "do_negation":
        auxiliary = "do" if subject == "they" else "does"
        surface = verb
        sentence = f"{subject.capitalize()} {auxiliary} not {surface} {number_word} {obj}"
        region = f"{auxiliary} not {surface}"
    elif template_family == "present_3s":
        surface = verb if subject == "they" else inflect(verb, "VBZ")
        sentence = f"Every day {subject} {surface} {number_word} {obj}"
        region = surface
    elif template_family == "simple_past":
        surface = inflect(verb, "VBD")
        sentence = f"Last week {subject} {surface} {number_word} {obj}"
        region = surface
    else:
        raise ValueError(f"Unknown template family: {template_family}")
    return sentence, surface, obj, region


def finiteness_violation_sentence(
    verb: str,
    noun: str,
    template_family: str,
    subject: str,
    number_word: str,
    add_error: bool,
) -> tuple[str, str, str, str, str, str]:
    """Reproduce ds003604's attested 3-s/do/be/ed finiteness-error families."""
    grammatical, expected_verb, obj, _ = grammatical_sentence(
        verb, noun, template_family, subject, number_word,
    )
    if template_family == "present_3s":
        if subject == "they":
            expected = verb
            observed = inflect(verb, "VBZ")
        else:
            expected = inflect(verb, "VBZ")
            observed = verb
        sentence = f"Every day {subject} {observed} {number_word} {obj}"
    elif template_family == "do_negation":
        expected, observed = (("do", "does") if subject == "they" else ("does", "do"))
        sentence = f"{subject.capitalize()} {observed} not {verb} {number_word} {obj}"
    elif template_family == "be_progressive":
        progressive = inflect(verb, "VBG")
        expected, observed = (("are", "is") if subject == "they" else ("is", "are"))
        sentence = f"{subject.capitalize()} {observed} {progressive} {number_word} {obj}"
    elif template_family == "simple_past":
        expected = inflect(verb, "VBD")
        observed = inflect(verb, "VBG") if add_error else verb
        sentence = f"Last week {subject} {observed} {number_word} {obj}"
    else:
        raise ValueError(f"Unknown template family: {template_family}")
    mechanism = (
        "substituted" if template_family in {"do_negation", "be_progressive"}
        else "added" if subject == "they" and template_family == "present_3s"
        else "omitted"
    )
    return sentence, observed, obj, expected, observed, mechanism


def plurality_violation_sentence(
    verb: str,
    noun: str,
    template_family: str,
    subject: str,
    number_word: str,
    add_error: bool,
) -> tuple[str, str, str, str, str, str]:
    if add_error:
        number_word = "one"
        expected, observed = noun, inflect(noun, "NNS")
    else:
        if number_word == "one":
            raise ValueError("An omission plurality violation requires a plural number word")
        expected, observed = inflect(noun, "NNS"), noun
    grammatical, verb_surface, _, region = grammatical_sentence(
        verb, noun, template_family, subject, number_word,
    )
    sentence = grammatical.rsplit(" ", 1)[0] + " " + observed
    return sentence, verb_surface, observed, number_word, expected, region
