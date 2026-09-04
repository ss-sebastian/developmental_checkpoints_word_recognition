from __future__ import annotations


SEED = 1729
SPLIT_SIZES = {"train": 360, "validation": 100, "test": 160}
CONDITION_COUNTS = {
    "Sound": {
        "train": {"rhyme": 90, "onset": 90, "unrelated": 180},
        "validation": {"rhyme": 25, "onset": 25, "unrelated": 50},
        "test": {"rhyme": 40, "onset": 40, "unrelated": 80},
    },
    "Meaning": {
        "train": {"high_association": 90, "low_association": 90, "unrelated": 180},
        "validation": {"high_association": 25, "low_association": 25, "unrelated": 50},
        "test": {"high_association": 40, "low_association": 40, "unrelated": 80},
    },
    "Plausibility": {
        "train": {"strong_congruence": 90, "weak_congruence": 90, "incongruent": 180},
        "validation": {"strong_congruence": 25, "weak_congruence": 25, "incongruent": 50},
        "test": {"strong_congruence": 40, "weak_congruence": 40, "incongruent": 80},
    },
    "Grammaticality": {
        "train": {"grammatical": 180, "finiteness_violation": 90, "plurality_violation": 90},
        "validation": {"grammatical": 50, "finiteness_violation": 25, "plurality_violation": 25},
        "test": {"grammatical": 80, "finiteness_violation": 40, "plurality_violation": 40},
    },
}

COMMON_COLUMNS = [
    "item_id", "task", "condition", "binary_label", "split",
    "stimulus_origin", "adaptation_eligibility", "original_trial_type",
    "original_stimulus_file",
    "source_resource", "source_record_id", "word1", "word2", "sentence",
    "word1_ipa", "word2_ipa", "sentence_ipa", "template_id", "critical_word",
    "critical_region", "association_forward", "association_backward", "word1_subtlex",
    "word2_subtlex", "mean_content_subtlex", "word1_n_phonemes", "word2_n_phonemes",
    "sentence_n_phonemes", "word1_n_syllables", "word2_n_syllables",
    "childes_count_word1", "childes_count_word2", "subject", "verb_lemma",
    "verb_surface", "object_lemma", "object_surface", "number_word", "negation",
    "overlap_ds003604_exact", "overlap_ds003604_critical_vocab", "qc_pass", "qc_notes",
]
TASK_SPECIFIC_COLUMNS = [
    "shared_onset", "shared_rime", "shared_phoneme_count", "phonological_relation_verified",
    "cue", "target", "FSG", "BSG", "verb_object_FSG", "verb_object_BSG",
    "congruence_level", "template_family", "grammar_subtype", "violation_type",
    "violation_location", "expected_form", "observed_form", "childes_count_verb",
    "childes_count_object", "content_word_childes_counts",
]
ALL_COLUMNS = COMMON_COLUMNS + TASK_SPECIFIC_COLUMNS

TASK_PREFIX = {
    "Sound": "SOUND", "Meaning": "MEANING", "Plausibility": "PLAUS", "Grammaticality": "GRAM",
}

POSITIVE_CONDITIONS = {
    "rhyme", "onset", "high_association", "low_association",
    "strong_congruence", "weak_congruence", "grammatical",
}

MEANING_BOUNDS = {
    "high_association": (0.40, 0.85),
    "low_association": (0.14, 0.39),
}
PLAUSIBILITY_BOUNDS = {
    "strong_congruence": (0.28, 0.81),
    "weak_congruence": (0.02, 0.19),
}

# Pair-level exclusions confirmed during human review. These are not mere
# semantic incongruities: each pair would introduce a countability, valency,
# instrument/compound, or lexical-sense confound under the fixed direct-object
# and number+noun sentence templates.
SENTENCE_PAIR_EXCLUSIONS = {
    ("accelerate", "speed"): "unnatural direct object; increase speed or accelerate a vehicle",
    ("allow", "permission"): "unnatural collocation; grant permission",
    ("cover", "girl"): "USF relation is sense-ambiguous and may reflect cover girl",
    ("defrost", "ice"): "mass noun is incompatible with the fixed numbered-object frame",
    ("glide", "airplane"): "rare transitive use is unsuitable for a child judgment task",
    ("twinkle", "star"): "twinkle is intransitive in the intended event",
    ("weave", "hair"): "mass/count sense is incompatible with the numbered-object frame",
    ("decide", "tobacco"): "decide does not license this noun as a direct object",
    ("weigh", "guard"): "event is plausible and therefore is not a valid incongruent item",
}

# Additional pair-level exclusions from the Grammaticality review. These stay
# task-specific so that revising this candidate pool cannot silently change the
# already reviewed Plausibility pool.
GRAMMATICALITY_PAIR_EXCLUSIONS = {
    ("rattle", "baby"): "possible transitive use is unsafe and pragmatically unsuitable for neutral child stimuli",
    ("sweep", "broom"): "broom is normally the instrument, not the direct object of sweep",
    ("rattle", "snake"): "unnatural transitive event and potentially confusable with the compound rattlesnake",
    ("switch", "light"): "normally requires the particle on/off in the intended event",
    ("owe", "money"): "mass noun is incompatible with the fixed numbered-object frame",
    ("spend", "money"): "mass noun is incompatible with the fixed numbered-object frame",
    ("bloom", "flower"): "bloom is intransitive in the intended event",
    ("check", "money"): "numbered money is unnatural and the intended verb sense is underspecified",
    ("lie", "truth"): "lie does not license truth as a direct object; tell the truth is required",
    ("reflect", "mirror"): "mirror is normally the reflector, not the direct object of reflect",
    ("pay", "money"): "mass noun is incompatible with the fixed numbered-object frame",
    ("comb", "hair"): "hair is a mass noun in the intended grooming event and conflicts with the numbered-object frame",
    ("count", "number"): "counting numbered instances of number is pragmatically unnatural in the fixed templates",
    ("eat", "food"): "food is a mass noun in the intended event and conflicts with the numbered-object frame",
}

# One-to-one replacements preserve the reviewed candidate IDs, condition
# composition, and all unaffected pairs.  Replacement pairs are drawn from the
# source-backed supplemental pool below.
GRAMMATICALITY_PAIR_REPLACEMENTS = {
    ("rattle", "baby"): ("build", "house"),
    ("sweep", "broom"): ("repair", "television"),
    ("rattle", "snake"): ("celebrate", "birthday"),
    ("switch", "light"): ("draw", "picture"),
    ("owe", "money"): ("define", "word"),
    ("spend", "money"): ("dig", "hole"),
    ("bloom", "flower"): ("fold", "letter"),
    ("check", "money"): ("feed", "bird"),
    ("lie", "truth"): ("flip", "coin"),
    ("reflect", "mirror"): ("grab", "bag"),
    ("pay", "money"): ("mix", "drink"),
    ("comb", "hair"): ("throw", "ball"),
    ("count", "number"): ("notify", "contact"),
    ("eat", "food"): ("fry", "egg"),
}

# Source-backed supplemental candidates used only to keep the strict
# split-exclusive lexical allocation feasible after Grammaticality review.
# Every pair is attested in USF with nonzero association and was screened for
# a transitive, countable-object reading under the fixed numbered templates;
# the generated sentences remain pending the user's human review.
GRAMMATICALITY_SUPPLEMENTAL_PAIRS = frozenset({
    ("throw", "ball"),
    ("draw", "picture"),
    ("salute", "flag"),
    ("build", "house"),
    ("peel", "banana"),
    ("wrap", "gift"),
    ("fill", "cup"),
    ("paint", "wall"),
    ("feed", "bird"),
    ("make", "cake"),
    ("repair", "television"),
    ("sew", "shirt"),
    ("tell", "story"),
    ("welcome", "guest"),
    ("fold", "letter"),
    ("rip", "paper"),
    ("tie", "shoe"),
    ("fasten", "belt"),
    ("dig", "hole"),
    ("define", "word"),
    ("celebrate", "birthday"),
    ("collect", "stamp"),
    ("grab", "bag"),
    ("flip", "coin"),
    ("snatch", "purse"),
    ("raise", "child"),
    ("notify", "contact"),
    ("suggest", "idea"),
    ("mix", "drink"),
    ("fry", "egg"),
})

TEMPLATE_FAMILIES = ("be_progressive", "do_negation", "present_3s", "simple_past")

# Explicit lexical-hygiene exclusions only; no item is excluded based on model behavior.
OFFENSIVE_DENYLIST = frozenset({
    "abuse", "bitch", "cigarette", "cunt", "fuck", "fucker", "fucking",
    "cigar", "cum", "gun", "kill", "killer", "murder", "nigger", "nigga",
    "rape", "rapist", "retard", "retarded", "shit", "slut", "sluts",
    "stab", "whore",
})
