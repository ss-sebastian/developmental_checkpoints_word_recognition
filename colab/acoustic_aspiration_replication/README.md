# Acoustic aspiration replication Colab

Open `run_all.ipynb`, select a T4 runtime, and Run all. It asks for the ZIP
made by acoustic Phase 1, validates exactly 30 checkpoints, downloads official
MALD word/pseudoword WAV and TextGrid archives directly from UAlberta, and
downloads one ZIP of results when complete. No Drive is mounted or required.

The formal setting uses all matched MALD items and nested ten-fold probes; it
is deliberately long. The notebook exposes `SMOKE_MAX_ITEMS` only for a
clearly labelled code-path smoke test. Checkpoint-level result caches make a
runtime restart resumable provided `/content` survives.
