# Acoustic aspiration-probe replication

`devlm.acoustic_aspiration_probe` is an evaluation-only, bounded replication of
Martin et al. (Interspeech 2023). It reads the 30 speech-trained acoustic GRU
checkpoints and does not create random-weight or non-speech branches, alter a
checkpoint, or change Phase 1.

It downloads the four pinned official MALD UAlberta Scholaris archives (words
and pseudowords, WAV and time-aligned TextGrids), checks size and MD5, caches
partial downloads with HTTP Range resume, and safely extracts ZIPs. The exact
Table-1 patterns are applied to MALD ARPAbet `phone` tiers: initial P/T/K is
the operational aspirated context, `S P/T/K` at word onset the post-s context,
and initial B/D/G the voiced context. Empty/silence labels are ignored and
adjacent identical phones are merged. A `# s C V` pattern is **not** widened to
an arbitrary internal sC cluster.

For GRUs, a phone vector is the mean of every hidden state whose explicit
left-aligned 25-ms frontend window overlaps its TextGrid phone interval; hops
are 10 ms. The 80-D baseline uses `torchaudio.compliance.kaldi.fbank` at
25 ms/10 ms/80 bins. The formal run fails clearly if that backend is
unavailable: it must not silently substitute project log-Mel. Only an explicit
small `--max-items` smoke run may use a manifest-labelled compatibility
fallback, uniformly for every baseline item.

Formal runs use every matched item. The `--max-items` argument is an explicit
smoke-only cap and its presence is recorded in `run_manifest.json`. Outer and
inner CV are 10-fold; item-ID/label-derived folds are fixed across all
checkpoints and Fbank. PCA is fitted only to each fold's training data. L2
`C={0.01,0.1,1,10}` and the fold seed are deterministic implementation choices:
the paper states nested 10-fold L2 tuning but does not report this grid/seed.
No class resampling is applied. Candidate PCA dimensions are powers of two plus
the terminal width (128 for GRU, 80 for Fbank); one `d*` is selected by summed
checkpoint control scores.

Every outer holdout receives a newly generated deterministic stratified
**ten-fold** inner assignment (not the remaining nine outer folds). Result and
pooled-vector cache names include a fingerprint of stimuli, interval labels,
CV settings, smoke/formal mode, and checkpoint content, so a smoke cache can
never be reused by a formal run.

The Colab uses a T4 for frozen-state extraction but CPU/scikit-learn for the
cross-validated probes. Per-representation control/final caches allow a
restart to retain completed checkpoints. This is not an identity replication:
a causal one-layer future-log-Mel GRU and checkpoint-exposure axis differ from
HuBERT Base's bidirectionally contextualized masked objective and layer axis.
