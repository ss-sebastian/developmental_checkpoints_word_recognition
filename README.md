# Developmental LM — Phases 1 and 2 behavioral evaluation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ss-sebastian/developmental_checkpoints_word_recognition/blob/main/colab/phase1_training.ipynb)

Phase 1 is a causal GRU trained solely by next-phoneme cross-entropy from noisy, continuous 10-ms articulatory-feature frames. Phase 2 evaluates the 30 frozen Phase 1 checkpoints on the OpenNeuro ds003604 Meaning Task by correlating two true-target probability measures with human trial-level RT. Phase 2 never retrains the GRU and contains no hidden-state probe, task head, classifier, RSA, MRI analysis, decision threshold, or SPRT.

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

Long sessions are processed with truncated backpropagation through time using `sequence_chunk_frames = 4096`. GRU hidden state is carried across chunks and detached at chunk boundaries, preventing cuDNN sequence-length failures and limiting GPU memory use. Optimizer steps therefore count chunks containing prediction targets, while checkpoint/evaluation scheduling remains based on cumulative observed frames.

The linked Colab notebook downloads and prepares its inputs under `/content`, trains entirely in local Colab runtime storage, then ZIPs and downloads every checkpoint together with the reproducibility metadata. It does not mount or write to Google Drive. Because `/content` is ephemeral, a runtime disconnection before the final download loses the local outputs.

The CLI reports data-loading progress every 100,000 utterances, displays a live session-level training bar with optimizer step, observed frames, equivalent input hours, and current loss, and prints validation metrics whenever it writes a checkpoint. Colab invokes Python unbuffered so these updates appear while the cell is running.

## Timing and target semantics

Timing is fixed: one frame is 10 ms; every phoneme occupies exactly five frames (50 ms); adjacent phonemes overlap by exactly one frame (10 ms). The overlap frame is the weighted sum of both articulatory feature vectors. The default symmetric envelope is `[1/3, 2/3, 1, 2/3, 1/3]`; a different five-value symmetric envelope can be supplied as `phoneme_envelope` and is normalized to peak 1.0. Temporal extent and overlap are not configurable in this version.

Words are concatenated without silence, zeros, separators, or special symbols. Only adjacent CHILDES utterances receive exactly three all-zero frames (30 ms). No noise is added to those frames.

For target phoneme `i+1`, including when its onset is the one-frame overlap, the classifier reads the causal GRU state at frame `onset(i+1)-1`. Thus no frame containing the target's activation contributes to its prediction. The leakage invariant is enforced in stream construction and covered by a unit test.

Each checkpoint stores model/optimizer state and: optimizer step, cumulative frames/phonemes/utterances, equivalent input hours (`frames × 10 ms`), validation next-phoneme loss, and validation next-phoneme accuracy. The run is a single developmental pass; each session is one optimizer step in this minimal baseline.

Checkpoint/evaluation timing is exposure-based rather than step-based. `target_checkpoint_count = 30` calculates an exact frame interval from the training split and fixed timing, aiming for roughly 30 checkpoints across the run. A full validation pass occurs only when a checkpoint is written and once at the end if needed; it does not run after every session.

## Phase 2 frozen-checkpoint behavioral evaluation

Install the Phase 2 dependencies and ensure `espeak`/`espeak-ng` is available, then copy and edit `configs/phase2.example.toml`:

```bash
python -m pip install -e '.[phase2]'
python -m devlm.phase2.cli --config configs/phase2.toml
```

The configured checkpoint directory must contain exactly 30 `.pt` files plus the Phase 1 `ipa_feature_mapping.json`. Checkpoints are sorted by their saved `cumulative_frames_seen` and labeled M01–M30. All GRU parameters are frozen, every experimental prime–target sequence starts from zero hidden state, and there is no reset or word-boundary input between its prime and target. Evaluation reuses the saved articulatory features, noise standard deviation, five-frame envelope, and one-frame overlap. Fixed behavior-independent word hashes make speech-frame Gaussian noise reproducible.

If `pronunciations_path` points to a JSON object, its values may be IPA strings or explicit IPA-token lists. Otherwise `phonemizer` with the configured espeak language converts the stimulus words. Its predictable American-English surface variants (for example, `ɚ`, `ɾ`, and uncombined affricates) are explicitly collapsed to phonemic counterparts already present in the Phase 1 inventory; no classifier class or feature vector is added. The evaluator fails explicitly for every remaining unrepresentable phone.

Only S_H, S_L, and S_U Meaning Task trials at sessions/ages 5, 7, and 9 are read. S_C is excluded. When the configured TSVs are absent, the evaluator downloads only the official stimulus-characteristics TSV and `task-Sem` events TSVs from the OpenNeuro ds003604 GitHub mirror; it never requests NIfTI or other MRI objects. Child response, accuracy, age, and RT files are loaded only after all frozen-model target probabilities have been generated.

For target IPA sequence \(\phi_1,\ldots,\phi_m\), v3 records each true next-phoneme log probability \(\ell_j=\log P(\phi_j\mid\text{prime},\phi_{<j})\). Each probability is read from the frame immediately before that phoneme first becomes active, so the first target prediction is −10 ms relative to target onset and contains no target activation. It does not construct, permute, or score null primes.

Two whole-target measures are evaluated:

- `log_probability_sum`: \(L^{sum}=\sum_j\ell_j=\log\prod_jP(\phi_j\mid\cdots)\), the stable log representation of the requested probability product;
- `mean_log_probability`: \(L^{mean}=m^{-1}\sum_j\ell_j\), which removes the mechanical accumulation with target phoneme count.

For every participant and checkpoint, each measure is correlated across trials with natural-log human RT. Outputs include directional Pearson \(r\), \(r^2\), and Fisher-z. The same correlations are also calculated after linearly residualizing both model score and log RT with respect to target phoneme count. Results are provided for all valid RT trials and, separately, correct-response trials. At least 10 usable trials are required by default; smaller samples remain in the table with null correlation values.

No behavioral parameter is fitted and no model decision time is claimed: this analysis tests whether frozen-model target predictability tracks human processing latency. The v1 actual-versus-permuted-prime SPRT and v2 calibrated-likelihood SPRT are archived under `src/devlm/phase2/v1/` and `src/devlm/phase2/v2/`. Neither is imported or invoked by the default CLI.

Phase 2 writes the following under its configured output directory:

- `tables/model_target_trajectory.tsv` with phoneme-level \(\ell_j\), cumulative \(L_j\), and running mean;
- `tables/model_target_probabilities.tsv` with the product, sum-log-probability, and mean-log-probability measures;
- `tables/behavior_probability_joined.tsv` with trial-level model and human data;
- participant-wise and checkpoint-by-age probability/RT correlation tables;
- Figures 1–4 as editable SVG, PDF, and 600-dpi TIFF;
- `phase2_run.json`, recording the analysis definition and run counts.

## Frozen-GRU task adaptation

`devlm.adaptation` trains binary readouts for Sound, Meaning, Plausibility, and Grammaticality from the fixed constructed stimulus manifest. The Colab entry point is `colab/task_adaptation_training.ipynb`: upload `adaptation_all_tasks.tsv` and the Phase 1 output ZIP containing exactly 30 checkpoints plus `ipa_feature_mapping.json`.

The manifest schema is documented in `docs/task_adaptation_manifest.md`; `tests/fixtures/task_adaptation.synthetic.tsv` is a tiny explicitly synthetic unit-test fixture. Both word-pair and sentence IPA specifications reuse the unchanged Phase 1 articulatory-feature, 10-ms frame, five-frame phoneme, overlap, and speech-noise pipeline. The completed sequence is encoded by the checkpoint GRU and only its final hidden state is passed to an exact `Linear(hidden_dim, 1)` readout. The Phase 1 language model is kept in evaluation mode, every parameter is frozen, and the exposed final state is detached.

The trainer uses one independently stored linear head for each M01–M30 checkpoint, with identical initialization and adaptation opportunity. It reads the already fixed 360/100/160 train/validation/test partitions from the stimulus manifest and verifies them before training. Age, participant identity, and RT are forbidden adaptation inputs. Checkpoint identity is never a readout feature. Every head receives the same maximum budget and validation-loss early-stopping rule; achieved accuracy is never forced to match.

The initial full-data run uses three paired initialization seeds. All checkpoints within a seed start with identical linear-head weights, and all seeds/checkpoints receive the same deterministic noisy stimulus frames. Metrics and head weights are written after every completed head, so an interrupted Colab training cell can resume. Behavioral alignment, child accuracy/RT analysis, model decision latency, threshold processes, checkpoint-to-age selection, and RSA remain deliberately unimplemented.

Initialization robustness uses paired seeds: one initialization seed is applied identically to all M01–M30 heads, and multiple seeds provide robustness repetitions rather than data cross-validation. Outputs are grouped under `task-<task>/init-seed-<seed>/heads/`, with unified `all_task_checkpoint_metrics.tsv` and `run_manifest.json` files at the output root.

## Task-adaptation stimulus construction

`devlm-construct-stimuli` is model-blind stimulus-construction infrastructure. It reads official CMUdict, USF Free Association Norms, SUBTLEX-US frequency and PoS files, Phase 1 IPA/CHILDES resources, and only the small ds003604 stimulus-characteristics tables. It never loads checkpoints, behavioral responses, or training code.

The four deterministic candidate pools contain 3,100 rows each. After the sentence reviews are complete, a seeded constrained randomization selects exactly 620 rows from each recorded pool. It enforces the requested split/condition counts, prevents critical words from crossing train/validation/test, balances the specified nuisance variables to `|SMD| < 0.10`, and restricts Plausibility and Grammaticality to human-reviewed candidates. The fixed `final_all.tsv` and split manifests are training inputs; unselected candidate rows are not.

For Plausibility and Grammaticality, the 60 original non-control ds003604 items per task are reconstructed exactly from the official metadata, including each original sentence and lexical verb–object surface pair. They are labeled `ds003604_original` and `reference_only`; newly generated items are labeled `plus` and `candidate_pending_human_review`. Each task folder contains `original_reference.tsv`, a plus-only `candidate_review.tsv`, and a combined `review_all.tsv`. Original reference items receive no adaptation split, preventing leakage if the same ds003604 items are later used for behavioral evaluation.

The review TSVs precompute expected yes/no responses, sentence-structure completeness, USF condition checks, and grammatical-error annotation checks. Human reviewers fill only the columns prefixed `human_`, using `yes` or `no`; notes are optional. Regeneration preserves any existing nonblank `human_` entries by `item_id`.

`combined/original_design_alignment.tsv` directly compares the final sets with the non-control ds003604 design. Sound/Meaning are aligned on comparable orthographic length, phoneme count, and syllable count; sentence tasks target the original condition-specific template, subject, and number proportions subject to integer, nuisance-balance, and lexical-disjointness feasibility. SUBTLEX Zipf is balanced across the new conditions but is not presented as numerically identical to ds003604's raw-frequency measure because those scales differ.

Confirmed lexical countability, valency, and lexical-sense failures are recorded in `sources/sentence_pair_exclusions.tsv` and removed before candidate selection. Semantic oddity alone is not an exclusion for an incongruent item, but every Plausibility sentence must retain a grammatical direct-object frame. Existing human entries are preserved only when both the sentence and source record still match the item identifier.
