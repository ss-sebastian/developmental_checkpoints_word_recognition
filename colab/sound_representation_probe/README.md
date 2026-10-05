# Sound representation probe Colab

Open `run_all.ipynb` and choose **Runtime → Run all**. Upload the Phase 1 output ZIP containing the 30 checkpoints plus `ipa_feature_mapping.json` when prompted. The notebook uses the dedicated model-blind primary manifest in `data/sound_representation_probe/`, runs the Sound-only M00–M30 frozen probe, and downloads `sound_representation_probe_results.zip`.

The notebook does not mount Google Drive and does not access behavioral responses, age, or RT. Test items are not used for head fitting or epoch selection; each selected head evaluates test once after validation-based restoration.
