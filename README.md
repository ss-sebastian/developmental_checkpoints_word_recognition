# Developmental LM — Phase 1

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ss-sebastian/01_developmental_checkpoints_word_recognition/blob/main/colab/phase1_training.ipynb)

This directory implements **Phase 1 only**: a causal GRU trained solely by next-phoneme cross-entropy from noisy, continuous 10-ms articulatory-feature frames. It has no word-boundary input, semantic/behavioral supervision, pretrained model, or Phase 2 analysis.

## Required real data (not included)

Real training intentionally refuses to guess missing scientific inputs. Supply:

1. A North American English IPA-CHILDES export (`.jsonl`, `.csv`, or `.tsv`) with `corpus_id`, `session_id`, `target_child_age_months`, `utterance_order`, `ipa`, and `text`. Optional `language` and `dialect` fields are filtered when present. A JSON-list `phonemes` field is strongly recommended because IPA segmentation conventions differ across exports.
2. A complete, reproducible feature table JSON: `{"feature_names": [...], "phonemes": {"IPA": [numeric, ...]}}`. `FeatureTable.from_panphon(...)` can generate and save this mapping when the optional PanPhon dependency is installed.
3. An empirical utterance-pause JSON of the form `{"values_ms": {"__default__": [observations...]}}`. Values are sampled directly and rounded to 10-ms frames. The tiny pause file under `tests/fixtures/` is synthetic and authorized only for tests/smoke checks; it is not a scientific default.

An empirical phoneme-duration file is **not required and is not read by the current model**. Every phoneme has the fixed timing specified below.

The loader preserves text, IPA, target-child age, corpus/session IDs, and utterance order. Sessions are split before training and validation, then the training sessions retain developmental age order. Checkpoints are indexed by exposure counters—not invented human ages.

## Environment and commands

From this directory:

```bash
source .venv/bin/activate
python -m pip install -e '.[panphon]'
python -m unittest discover -s tests -v
python -m devlm.cli --config configs/smoke.toml
```

For real training, copy `configs/phase1.example.toml`, replace all required data paths, and run the same CLI. The default noise standard deviation is `0.05`. Noise is added only where speech is active; utterance pauses remain exact all-zero vectors.

## Timing and target semantics

Timing is fixed: one frame is 10 ms; every phoneme occupies exactly five frames (50 ms); adjacent phonemes overlap by exactly one frame (10 ms). The overlap frame is the weighted sum of both articulatory feature vectors. The default symmetric envelope is `[1/3, 2/3, 1, 2/3, 1/3]`; a different five-value symmetric envelope can be supplied as `phoneme_envelope` and is normalized to peak 1.0. Temporal extent and overlap are not configurable in this version.

Words are concatenated without silence, zeros, separators, or special symbols. Only adjacent CHILDES utterances receive empirically sampled all-zero pauses.

For target phoneme `i+1`, including when its onset is the one-frame overlap, the classifier reads the causal GRU state at frame `onset(i+1)-1`. Thus no frame containing the target's activation contributes to its prediction. The leakage invariant is enforced in stream construction and covered by a unit test.

Each checkpoint stores model/optimizer state and: optimizer step, cumulative frames/phonemes/utterances, equivalent input hours (`frames × 10 ms`), validation next-phoneme loss, and validation next-phoneme accuracy. The run is a single developmental pass; each session is one optimizer step in this minimal baseline.
