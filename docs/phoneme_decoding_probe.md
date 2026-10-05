# Peak-frame phoneme decoding probe

This is a diagnostic check of whether a frozen Phase 1 GRU state contains the
identity of the **currently active phoneme**. Each token contributes exactly one
state: `h[start_frame + 2]`, the third of the five phoneme frames, where the
input envelope reaches its unique peak of 1.0. The three all-zero utterance
pause frames are never examples.

The probe uses only Phase 1 validation sessions, then assigns entire sessions
to probe-train, validation, and test partitions. It retains only phonemes with
at least 20 tokens in every partition and class-balances tokens independently
within each partition. Therefore neither neighboring frames nor a session can
cross a partition boundary, and common phonemes cannot dominate the score.

For each frozen M00--M30 representation, a deterministic, L2-regularized
multinomial logistic classifier is fit on probe-train tokens. Its L2 value is
selected with validation cross-entropy; test labels are not passed to either
fit or selection. Report macro-F1, balanced accuracy (macro recall), top-1,
top-3, and cross-entropy. M00 is reconstructed from the Phase 1 initialization
seed rather than being a separately trained control.

The `figures/*_shared_pca.png` files provide the requested colored-dot map:
every translucent point is a held-out phoneme token, color identifies its IPA
phoneme, and a black-edged diamond labels each phoneme centroid. The faint
circle is a visual guide only, not a confidence boundary. PCA is fitted once on
the exact equal-checkpoint pooled within-checkpoint covariance of balanced
probe-train states from all M00--M30 models (never validation or test states).
Each checkpoint is centered by its own probe-train mean before projection, and
the resulting fixed two-dimensional basis is used for every
checkpoint. `pca_six_checkpoint_comparison` uses common axes and palette for
M00/M01/M05/M10/M20/M30. The faint circle is purely cosmetic, not a confidence
region. The PCA pass is online: it retains only each checkpoint's 128-vector
mean, 128×128 within-checkpoint scatter matrix, and count, never all hidden
states. PCA is an illustration only: a poor
two-dimensional separation does not establish that the 128-dimensional states
are not linearly decodable.

The result archive includes the fixed `configs/phoneme_event_manifest.tsv`,
per-checkpoint metrics, held-out token predictions, and long-form confusion
matrices. These files make it possible to inspect a seemingly good macro score
for a small group of systematically confused phonemes.

The trajectory figure contrasts hidden-state decoding with two controls: a
training-prevalence majority classifier and a linear classifier on the raw noisy
peak input vector. These controls are selected without test labels.

Run after extracting the Phase 1 output archive:

```bash
devlm-phoneme-decoding-probe \
  --source-csv /path/to/Eng-NA_processed.csv \
  --checkpoint-root /path/to/devlm_phase1_outputs \
  --output-root /path/to/phoneme_decoding_outputs \
  --device cuda
```

Do not interpret a high peak-frame score as learned phonological abstraction:
the current phoneme's feature vector is supplied to the GRU at that frame, so
M00 may also decode well. A useful follow-up is delayed decoding after the
phoneme has disappeared, or cross-context generalization tests.
