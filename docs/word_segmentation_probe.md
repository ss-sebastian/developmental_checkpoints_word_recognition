# Minimal word-segmentation probe

This analysis asks when a frozen Phase 1 hidden state first contains linearly
decodable information about an **utterance-internal word boundary**. It does not
test whether the language model can emit an unsupervised segmentation.

For every adjacent phoneme transition in the original IPA-CHILDES export, the
label is one when `WORD_BOUNDARY` occurs between the two phonemes and zero
otherwise. The marker is removed before frame construction and is never an
input feature. Transitions between utterances are excluded, so the three
zero-vector pause frames cannot reveal the label.

The readout state for a transition into phoneme B is the GRU state at
`B.start_frame - 1`: the frame immediately before B's first active frame. All
31 models use the same Phase 1 validation sessions, session-level probe split,
sampled transitions, input noise realization, and `Linear(128, 1)` training
settings. M00 reconstructs the original untrained GRU initialization from the
Phase 1 seed; M01--M30 are the frozen developmental checkpoints.

The primary outputs are ROC-AUC, balanced accuracy, and observed boundary
prevalence. Ordinary accuracy is included only as a secondary diagnostic.
Test uncertainty is estimated by resampling test sessions. Segmentation onset
is the earliest of three consecutive trained checkpoints whose test-AUC 95%
CI lower bound exceeds 0.5 and whose point AUC exceeds M00.

Run after extracting the Phase 1 output ZIP:

```bash
devlm-segmentation-probe \
  --source-csv /path/to/Eng-NA_processed.csv \
  --checkpoint-root /path/to/devlm_phase1_outputs \
  --output-root /path/to/segmentation_probe_outputs \
  --device cuda
```

The raw CSV must retain `WORD_BOUNDARY` in its `ipa_transcription` column. The
checkpoint directory must contain all 30 checkpoints, `session_split.json`, and
`ipa_feature_mapping.json`.

To keep the 31-model comparison tractable, transitions are uniformly sampled
with a fixed seed after the session split. Default caps are 120,000 train,
30,000 validation, and 30,000 test transitions. The complete selected-item
manifest is saved, so every checkpoint is evaluated on exactly the same data.

This probe deliberately excludes RT, SOA, target trajectories, n-gram and
phonological controls, nonlinear probes, semantic tasks, and lexical-disjoint
word splits.
