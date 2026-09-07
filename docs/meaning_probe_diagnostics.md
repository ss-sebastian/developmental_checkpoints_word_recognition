# Formal Meaning-probe diagnostics

`devlm-meaning-probe` is the auditable runner for the final 1,000-item Meaning dataset. It is distinct from the earlier exploratory four-task/620-item adaptation command.

## Input contract

The fixed TSV is `data/task_adaptation_stimuli/meaning_1000/final_all.tsv`. It contains exactly 1,000 unique Meaning rows: train 580 (145 High, 145 Low, 290 Unrelated), validation 160 (40/40/80), and test 260 (65/65/130). Every split is exactly 50/50 at the binary level. Required fields are `item_id`, `task`, `condition`, `binary_label`, `split`, `word1_ipa`, `word2_ipa`, `FSG`, and `BSG`, plus either `cue`/`target` or `word1`/`word2`. Conditions accept `High`/`high_association`, `Low`/`low_association`, and `Unrelated`; High and Low must have label 1 and Unrelated label 0. IPA fields are JSON arrays of tokens. Child variables are not accepted or propagated.

The checkpoint directory must resolve to exactly 30 unique Phase 1 checkpoints. A single deterministic stimulus-frame batch is shared across them. The runner records SHA256 values for the dataset bytes, padded stimulus frames plus lengths/item order, every checkpoint, every checkpoint's Phase 1 config, and the probe training config.

## Selection and test isolation

The Phase 1 GRU remains frozen. Hidden dimensions are standardized using training items only while fitting; the affine transform is folded into the saved raw-state linear head. Every epoch evaluates only train and validation. The stored best epoch is the exact minimum validation-loss epoch; `min_delta` controls patience only. Once training stops, that state is restored, train and validation are evaluated, and test is evaluated exactly once. Test values never select epoch, hyperparameters, checkpoint, or train size.

`gradient_norm` is the mean across minibatches of the global L2 norm of all linear-head gradients in that epoch. Individual High, Low and Unrelated rows report accuracy and mean probability/logit; their AUC is `NaN` because each contains one binary class by design. High-versus-Unrelated and Low-versus-Unrelated rows additionally report a valid logit-based ROC-AUC.

## Bundle

The output parent receives `meaning_probe_results.zip`, whose root contains:

```text
summary/checkpoint_probe_metrics.csv
summary/condition_metrics.csv
summary/run_status.csv
histories/M01_history.csv ... M30_history.csv
predictions/M01_predictions.tsv ... M30_predictions.tsv
heads/M01_linear_probe.pt ... M30_linear_probe.pt
cached_hidden/M01_hfinal.npz ... M30_hfinal.npz
configs/training_config.json
configs/dataset_manifest.json
configs/checkpoint_manifest.json
configs/environment.json
run_manifest.json
run_log.txt
```

Every cache contains `item_ids`, an exactly `[1000, hidden_dim]` float32 `h_final`, and dataset, stimulus-frame and checkpoint hashes. A failed checkpoint is never skipped: `run_status.csv` records `failed` and its reason, and `run_log.txt` retains the traceback. Warnings such as undefined AUC or exhaustion of `max_epochs` without early stopping are also retained.
