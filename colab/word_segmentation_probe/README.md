# Word-segmentation probe Colab

Open `run_all.ipynb`, select a GPU runtime, and choose **Runtime → Run all**.
The only manual input is the Phase 1 ZIP containing exactly 30 checkpoints,
`session_split.json`, and `ipa_feature_mapping.json`.

The notebook does not mount Google Drive. It downloads IPA-CHILDES directly,
keeps only the Phase 1 validation sessions while preserving `WORD_BOUNDARY`,
runs M00 and M01–M30, plots the requested metrics, and downloads
`segmentation_probe_results.zip` from Colab's local `/content` storage.
