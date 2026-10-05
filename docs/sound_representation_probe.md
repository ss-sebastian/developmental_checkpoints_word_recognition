# Sound representation probe

This pipeline is restricted to the Sound task. It trains a `Linear(128, 1)` binary readout on frozen representations from an architecture-matched random M00 GRU and Phase 1 checkpoints M01–M30. Onset/rhyme trials are positive and unrelated trials are negative. No child responses, age, or reaction-time data enter the run.

## Primary stimuli

`data/sound_representation_probe/primary_manifest.tsv` is a model-blind subset of the existing Sound manifest. Its selector preserves the preassigned train/validation/test partitions and rejects lexical overlap between them. It retains only QC-passing, relation-verified, one-syllable words with positive CHILDES counts; the unrelated condition must have zero shared phonemes. Within each split it samples balanced onset/rhyme/unrelated counts while searching deterministically for low lexical reuse, bounded connected components, and improved balance on mean SUBTLEX frequency, phoneme count, and orthographic length. It never loads model checkpoints or uses probe performance.

Current selected counts are train 55 onset + 55 rhyme + 110 unrelated; validation 18 + 18 + 36; test 25 + 25 + 50 (392 total). The attested unrelated pool in the existing assigned test split only supports 50 balanced negatives, so test cannot reach 58 negatives without changing the source pool or split. The generated QC file records achieved lexical reuse, component sizes, and nuisance SMDs. Some residual SMDs remain sizeable because attested candidates are limited, notably rhyme phoneme-count SMD in validation and word-frequency SMD in test; interpret condition comparisons with this limitation in mind. The selector uses lexical metadata only and does not select by model or behavioral outcomes.

## Input and leakage controls

Each item is encoded as word 1 speech, exactly three zero-vector pause frames, then word 2 speech. The Phase 1 stream builder supplies the original five frames per phoneme, one-frame within-word overlap, and Gaussian noise on speech frames only. There is no cross-word phoneme overlap or hidden-state reset. The linear probe reads the final GRU hidden state after word 2.

For every checkpoint and initialization seed, representation standardization is fitted on train examples only. The fitting function does not accept test tensors. Each epoch evaluates train and validation only; best epoch and early stopping depend only on validation cross-entropy. The best validation-selected head is restored, train-only standardization is folded into its saved weights, and only then does the runner perform one test forward evaluation. The executable unit test checks this interface and ordering. Test representations may be extracted before fitting for efficiency, but neither they nor test labels enter optimization or model selection.

Three paired head-initialization seeds are used by default. All checkpoints receive the same items and deterministic noisy frames. Reported held-out metrics are overall AUC, balanced accuracy, and cross-entropy, plus onset-vs-unrelated and rhyme-vs-unrelated AUC. The developmental statistic is test AUC delta versus M00, with paired bootstrap resampling of lexical connected components. Representation onset is the earliest checkpoint among three consecutive checkpoints whose delta-AUC 95% lower bounds all exceed zero.

## Run

Prepare or regenerate the model-blind manifest:

```bash
devlm-sound-probe prepare \
  --source data/task_adaptation_stimuli/sound/final_all.tsv \
  --output data/sound_representation_probe/primary_manifest.tsv
```

Run after extracting the Phase 1 ZIP (containing exactly M01–M30 and the feature mapping):

```bash
devlm-sound-probe run \
  --stimuli data/sound_representation_probe/primary_manifest.tsv \
  --checkpoints-dir /path/to/phase1-extracted \
  --output-dir outputs/sound_representation_probe
```

The dedicated Colab `colab/sound_representation_probe/run_all.ipynb` performs the same steps, displays the summary tables, and downloads the complete result directory. Outputs include saved raw-hidden-state linear heads, per-epoch train/validation histories, per-seed metrics, held-out test predictions, checkpoint summaries, developmental delta-AUC intervals, and a run manifest recording the test-isolation guarantee.
