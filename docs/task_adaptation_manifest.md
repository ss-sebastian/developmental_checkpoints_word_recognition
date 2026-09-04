# Future task-adaptation manifest

This interface is infrastructure only. The checked-in fixture is entirely synthetic and is not adaptation data.

Use a UTF-8 TSV with exactly these required fields:

| Field | Meaning |
|---|---|
| `item_id` | Unique stimulus-item identifier |
| `task_name` | `Sound`, `Meaning`, `Plausibility`, or `Grammaticality` |
| `stimulus_kind` | `word_pair` or `sentence` |
| `prime_phonemes` | JSON array of IPA tokens for a word-pair prime; blank for sentences |
| `target_phonemes` | JSON array of IPA tokens for a word-pair target; blank for sentences |
| `sentence_phonemes` | JSON array of IPA tokens for a sentence; blank for word pairs |
| `binary_label` | `0` (negative category) or `1` (positive category) |
| `split_group` | Group kept intact across item-level train/validation/test splitting |
| `metadata_json` | Task-specific JSON object; must not contain age, participant, or RT fields |

The IPA arrays are passed through the existing Phase 1 articulatory-feature and frame-generation pipeline. A word pair is presented as one continuous prime–target phoneme sequence with no added boundary. The readout receives only the frozen GRU's final hidden state.

Create and save one deterministic grouped split per task before any future adaptation. Reopen that same split for M01–M30; never derive splits from participant repetitions. Split ratios, optimization settings, and dataset-specific label meanings remain provisional until real datasets are chosen.

The example optimizer family and all numerical optimization values are placeholders. Whatever values are scientifically selected later must be held identical across M01–M30, and early stopping must use only a validation-loss plateau—not achieved accuracy.

Initialization robustness and data cross-validation are distinct. For each configured initialization seed, all M01–M30 heads start from the same seed-specific weights and reuse the same saved item split. The future output namespace is:

```text
outputs/task_adaptation/
  task-<task>/
    split-<first-12-SHA256-chars>/
      init-seed-<10-digit-seed>/
        run_manifest.json
        all_checkpoint_metrics.tsv
        heads/
          M01_linear_readout.pt
          ...
          M30_linear_readout.pt
```

Thus `all_checkpoint_metrics.tsv` is one unified M01–M30 record for a single initialization seed, while the path also makes the task and saved split unambiguous. Actual files are not created by the current infrastructure. If item-fold cross-validation is later selected scientifically, fold identity must receive a separate namespace rather than being encoded as an initialization seed.

The future learning-curve sizes are fixed in the interface at 25, 50, 100, 200, and 400 items. Their eventual selection must use adaptation-validation performance only, never child behavior.
