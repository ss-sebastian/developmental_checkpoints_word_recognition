# Developmental LM — Phase 1

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ss-sebastian/developmental_checkpoints_word_recognition/blob/main/colab/phase1_training.ipynb)

This directory implements **Phase 1 only**: a causal GRU trained solely by next-phoneme cross-entropy from noisy, continuous 10-ms articulatory-feature frames. It has no word-boundary input, semantic/behavioral supervision, pretrained model, or Phase 2 analysis.

## Real data preparation

For local training, supply:

1. A North American English IPA-CHILDES export (`.jsonl`, `.csv`, or `.tsv`) with `corpus_id`, `session_id`, `target_child_age_months`, `utterance_order`, `ipa`, and `text`. Optional `language` and `dialect` fields are filtered when present. A JSON-list `phonemes` field is strongly recommended because IPA segmentation conventions differ across exports.
2. A complete, reproducible feature table JSON: `{"feature_names": [...], "phonemes": {"IPA": [numeric, ...]}}`.

The Colab notebook instead downloads the public Hugging Face `phonemetransformers/IPA-CHILDES` EnglishNA subset directly, removes every `WORD_BOUNDARY` marker, converts it to the loader schema, and generates the feature table from its observed inventory using PanPhon. Multi-segment IPA-CHILDES phoneme tokens (for example, diphthongs) use the mean of their PanPhon segment vectors. No phoneme-duration or pause-duration file is required.

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

`device = "auto"` selects CUDA when available and otherwise uses CPU. The Colab notebook explicitly sets `device = "cuda"` so a missing GPU fails immediately instead of silently training on CPU. Model inputs, target indices, training, and validation all use the selected device.

The linked Colab notebook downloads and prepares its inputs under `/content`, trains entirely in local Colab runtime storage, then ZIPs and downloads every checkpoint together with the reproducibility metadata. It does not mount or write to Google Drive. Because `/content` is ephemeral, a runtime disconnection before the final download loses the local outputs.

The CLI reports data-loading progress every 100,000 utterances, displays a live session-level training bar with optimizer step, observed frames, equivalent input hours, and current loss, and prints validation metrics whenever it writes a checkpoint. Colab invokes Python unbuffered so these updates appear while the cell is running.

## Timing and target semantics

Timing is fixed: one frame is 10 ms; every phoneme occupies exactly five frames (50 ms); adjacent phonemes overlap by exactly one frame (10 ms). The overlap frame is the weighted sum of both articulatory feature vectors. The default symmetric envelope is `[1/3, 2/3, 1, 2/3, 1/3]`; a different five-value symmetric envelope can be supplied as `phoneme_envelope` and is normalized to peak 1.0. Temporal extent and overlap are not configurable in this version.

Words are concatenated without silence, zeros, separators, or special symbols. Only adjacent CHILDES utterances receive exactly three all-zero frames (30 ms). No noise is added to those frames.

For target phoneme `i+1`, including when its onset is the one-frame overlap, the classifier reads the causal GRU state at frame `onset(i+1)-1`. Thus no frame containing the target's activation contributes to its prediction. The leakage invariant is enforced in stream construction and covered by a unit test.

Each checkpoint stores model/optimizer state and: optimizer step, cumulative frames/phonemes/utterances, equivalent input hours (`frames × 10 ms`), validation next-phoneme loss, and validation next-phoneme accuracy. The run is a single developmental pass; each session is one optimizer step in this minimal baseline.

Checkpoint/evaluation timing is exposure-based rather than step-based. `target_checkpoint_count = 30` calculates an exact frame interval from the training split and fixed timing, aiming for roughly 30 checkpoints across the run. A full validation pass occurs only when a checkpoint is written and once at the end if needed; it does not run after every session.
