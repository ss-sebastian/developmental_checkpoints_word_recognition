# Independent acoustic log-Mel Phase 1 (50-hour pilot)

This is a separate causal/autoregressive model. It does not alter, load, or
replace the existing IPA-feature Phase 1 model. Its purpose is to test what
predictive learning can acquire from raw adult/caregiver child-directed speech
without being given IPA symbols, phoneme boundaries, word boundaries, a feature
table, artificial phoneme timing, or extra Gaussian noise.

## Default public source: BabySLM Providence

The Colab default is the public [BabySLM Providence audio archive](https://cognitive-ml.fr/downloads/baby-slm/training_sets/Providence/audio.zip), not TalkBank login. At inspection time it contains WAV clips in paths such as `audio/Alex_MOT/...` and also many non-MOT/FAT speaker-code folders. The preparer accepts only `MOT` and `FAT` folders, excludes every other code, and groups clips by the archive filename's recording/session stem before a deterministic session-disjoint split.

This is direct evidence for a **speaker-code filter**, not evidence that every retained caregiver turn is addressed to the child. The archive manifest records `directed_to_child=corpus_context` rather than `true`. The archive is audio-only and its filenames do not provide verified child-age metadata to this implementation. Its manifest therefore stores a deterministic `exposure_order`, not invented `target_child_age_months`; checkpoints from this source must not be mapped to individual child ages. The 50h/0.5h split is still leakage-safe at the parsed session-stem level.

The archive is about 13 GB and selected WAV extraction needs additional local runtime storage. The original archive is a temporary input; training outputs remain separate.

## Input contract

Give the runner a TSV or CSV manifest and PCM WAV files. The first pilot only
supports WAV deliberately, so its audio decoder is transparent and reproducible.
The manifest requires these columns:

```text
audio_path	corpus_id	session_id	target_child_age_months	source_corpus	speaker_role	directed_to_child	recording_order
recordings/corpA_s01_001.wav	corpusA	s01	18	TalkBank-Example	caregiver	true	1
```

`audio_path` is relative to the manifest (or absolute). `recording_order` is
optional. Each row must be an adult/caregiver speaker (`adult`, `caregiver`,
`mother`, `father`, `parent`, `grandparent`, or `other_adult`) and must set
`directed_to_child=true`. The loader explicitly rejects `CHI`/target-child
speech and any row without explicit child-directed metadata.

The existing IPA-CHILDES text export is **not** an audio source and cannot be
silently substituted. Likewise, CHILDES-Aligned is not used as a default: its
official release is child-speech-only and would change the exposure population.
`SOURCE_MODE = 'providence'` is an optional direct,
runtime-only preparation route for the TalkBank Providence corpus. Before
running it, add `TALK_BANK_EMAIL` and `TALK_BANK_PASSWORD` in the Colab Secrets
sidebar; these are fetched only into the live runtime. This avoids terminal
`getpass`, which can appear to hang or be invisible in Colab. If Secrets are
missing or denied, the notebook displays a masked `ipywidgets.Password` field;
click its Continue button and rerun that preparation cell. Credentials/cookies
are never printed, written to disk, logged, placed in configuration, or put in
the downloadable archive. The notebook POSTs credentials only to the TalkBank
login endpoint. It downloads the official Providence transcript ZIP, selects only
timestamped `MOT`, `FAT`, grandparent or other adult CHAT tiers, excludes
`CHI`, downloads only the linked media required for a 50-hour train/0.5-hour
validation selection, and cuts PCM WAV segments with `ffmpeg`. Selection uses
the actual 16-kHz/25-ms-window/10-ms-hop frame formula, then retains a one
minute safety margin for per-utterance window loss. The training runner—not the
preparer—then enforces the exact 50-hour (18,000,000-frame) cap.

This is not a claim that every selected adult tier has a child addressee:
Providence is a naturalistic parent-child corpus and CHAT provides the speaker
code, but not an addressee label per utterance. The generated manifest records
`directed_to_child=true` as a **corpus-context** inclusion decision. This is a
scientific limitation to retain in interpretation. If you require a manually
verified utterance-level addressee label, use the `upload_zip` route instead.

TalkBank media access can change or require an approved account. If the direct
route fails its explicit access checks, it must not be worked around with a
different corpus or child-only audio. Use `SOURCE_MODE = 'upload_zip'` to
upload properly licensed, preselected adult/caregiver CDS WAV files plus this
manifest.

## Model and loss

Waveform is resampled to 16 kHz, then converted to an 80-dimensional log-Mel
representation using a 25-ms window and exactly a 10-ms hop. Natural recording
silence stays in the waveform; no three-frame pause or artificial silence is
inserted. The first run also adds no synthetic Gaussian noise.

At time \(t\), a causal 128-unit GRU sees only frames up to \(t\) and has
separate linear heads for 30, 50 and 100 ms into the future:

\[
h_t \to (\hat{x}_{t+3}, \hat{x}_{t+5}, \hat{x}_{t+10}).
\]

The loss is the equal-weight mean of the three future log-Mel MSE values. There
is intentionally no one-frame (10-ms) prediction-only objective, because it
could be solved mainly by adjacent-frame acoustic copying. This remains an
autoregressive predictive model, not a bidirectional or masked model.

## Exposure and outputs

The 50-hour pilot caps training at exactly 18,000,000 10-ms log-Mel frames. It
processes eligible training sessions in manifest order after a deterministic
session-level holdout split: child-age order when verified ages are supplied,
or explicit `exposure_order` when they are not. The last recording is truncated
at the cap if needed. Validation uses an independent session split and is capped separately
at 0.5 hours. Its fixed per-frequency normalization is estimated from a
deterministic one-hour sample drawn only from the selected 50-hour training
exposure prefix, then saved and reused unchanged for training and validation.
With 30 requested checkpoints, evaluation/checkpointing is every approximately
1.67 hours of observed acoustic input.

Outputs include acoustic checkpoints, `metrics.jsonl`, `audio_session_split.json`,
`audio_source_manifest.json`, and `acoustic_run_manifest.json`. The latter
records the source-corpus and adult/CDS filters, configuration, counters and
manifest hash. `audio_training_exposure.json` records the exact capped training
audio prefix, including any final recording truncation.
