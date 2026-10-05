from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from devlm.acoustic.config import load_config
from devlm.acoustic.data import load_audio_manifest, split_audio_sessions
from devlm.acoustic.features import _mel_filterbank, log_mel_frame_count, log_mel_spectrogram
from devlm.acoustic.babyslm import _clip_from_name, _select_frames, _session_split as babyslm_session_split, migrate_cached_babyslm_manifest
from devlm.acoustic.providence import _download_media, _frames_from_duration_ms, _request_session, parse_chat_segments, parse_providence_corpus
from devlm.acoustic.train import FRAME_MS, round_robin_validation_items, train


def write_wav(path: Path, seconds: float = 0.12, sample_rate: int = 16_000) -> None:
    values = (0.1 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sample_rate)) / sample_rate) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(values.tobytes())


class AcousticPhase1Tests(unittest.TestCase):
    def manifest(self, root: Path, *, child_directed: str = "true", speaker: str = "caregiver") -> Path:
        rows = []
        for index in range(4):
            wav = root / f"audio_{index}.wav"
            write_wav(wav)
            rows.append({
                "audio_path": wav.name, "corpus_id": "synthetic", "session_id": f"s{index}",
                "target_child_age_months": str(12 + index), "source_corpus": "synthetic-test",
                "speaker_role": speaker, "directed_to_child": child_directed, "recording_order": "1",
            })
        path = root / "manifest.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_log_mel_has_ten_ms_hop_and_no_label_input(self):
        waveform = torch.zeros(1_600)
        features = log_mel_spectrogram(waveform, 16_000, target_sample_rate=16_000, n_mels=80, n_fft=800, win_length=400, hop_length=160)
        self.assertEqual(FRAME_MS, 10)
        self.assertEqual(features.shape, (8, 80))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.all(_mel_filterbank(16_000, 800, 80).sum(dim=1) > 0))

    def test_manifest_rejects_child_or_not_directed_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsupported directed_to_child"):
                load_audio_manifest(self.manifest(root, child_directed="false"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "target-child/CHI"):
                load_audio_manifest(self.manifest(root, speaker="CHI"))

    def test_session_split_is_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            items = load_audio_manifest(self.manifest(Path(directory)))
            train_items, validation_items = split_audio_sessions(items, 0.25, 3)
            self.assertTrue({item.session_key for item in train_items}.isdisjoint({item.session_key for item in validation_items}))

    def test_providence_parser_keeps_timestamped_adult_tiers_and_excludes_chi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat = root / "sample.cha"
            chat.write_text(
                "@Begin:\n@Participants:\tCHI Target_Child, MOT Mother, FAT Father\n"
                "@ID:\teng|Providence|CHI|1;06.00|female|||Target_Child|||\n"
                "@Media:\tsample, audio\n*MOT:\thello there . \x150_1000\x15\n"
                "*CHI:\tba . \x151000_1500\x15\n*FAT:\tlook at that . \x151500_2600\x15\n@End:\n",
                encoding="utf-8",
            )
            segments = parse_chat_segments(chat, root)
            self.assertEqual(len(segments), 2)
            self.assertEqual([segment.speaker_role for segment in segments], ["mother", "father"])
            self.assertTrue(all(segment.target_child_age_months == 18 for segment in segments))
            self.assertTrue(all(segment.media_relative_path == "sample" for segment in segments))

    def test_providence_corpus_skips_missing_age_file_and_retains_valid_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.cha").write_text(
                "@Begin:\n@Participants:\tCHI Target_Child, MOT Mother\n"
                "@ID:\teng|Providence|CHI|2;00.00|female|||Target_Child|||\n"
                "@Media:\tvalid, audio\n*MOT:\tgood . \x150_1000\x15\n@End:\n",
                encoding="utf-8",
            )
            (root / "missing_age.cha").write_text(
                "@Begin:\n@Participants:\tMOT Mother\n@Media:\tmissing, audio\n"
                "*MOT:\tgood . \x150_1000\x15\n@End:\n",
                encoding="utf-8",
            )
            segments, skipped, first_paths = parse_providence_corpus(root)
            self.assertEqual(len(segments), 1)
            self.assertEqual(skipped["missing_chi_age"], 1)
            self.assertEqual(first_paths["missing_chi_age"], "missing_age.cha")

    def test_providence_login_rejects_http_200_json_failure_even_with_cookie(self):
        class Response:
            status_code = 200

            def json(self):
                return {"success": False}

        class Session:
            cookies = {"talkbank": "anonymous"}

            def post(self, *args, **kwargs):
                return Response()

        with self.assertRaisesRegex(RuntimeError, "login failed"):
            _request_session("user@example.org", "not-a-real-password", session_factory=Session)

    def test_providence_frame_accounting_reflects_window_loss(self):
        # Zero-padding each DFT must not extend the explicit 25-ms frame.
        self.assertEqual(_frames_from_duration_ms(100), 8)
        self.assertEqual(_frames_from_duration_ms(1000), 98)

    def test_feature_frame_count_uses_the_25_ms_window_not_dft_length(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "one_second.wav"
            write_wav(wav, seconds=1.0)
            self.assertEqual(
                log_mel_frame_count(wav, target_sample_rate=16_000, n_fft=800, win_length=400, hop_length=160),
                98,
            )

    def test_providence_media_rejects_invalid_payload_and_tries_next_container(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            attempts: list[str] = []

            def downloader(_session, url, destination):
                attempts.append(url.rsplit("/", 1)[-1])
                destination.write_bytes(b"not necessarily media")

            def probe(path):
                return (path.suffix == ".mov", "valid audio" if path.suffix == ".mov" else "invalid payload")

            output = _download_media(None, "Ethan/001104.mp3", cache, downloader=downloader, probe=probe)
            self.assertEqual(output.suffix, ".mov")
            self.assertEqual(attempts, ["001104.mp3", "001104.wav", "001104.mp4", "001104.mov"])
            self.assertEqual(list(cache.iterdir()), [output])

    def test_babyslm_filename_filter_only_accepts_mot_fat_and_groups_session(self):
        mot = _clip_from_name("audio/Alex_MOT/Alex_010427_05_02_2002_MOT_10888_15525.wav")
        fat = _clip_from_name("audio/Alex_FAT/Alex_010427_05_02_2002_FAT_15525_19516.wav")
        self.assertIsNotNone(mot)
        self.assertIsNotNone(fat)
        self.assertEqual(mot.session_id, fat.session_id)
        self.assertIsNone(_clip_from_name("audio/Alex_CHI/Alex_010427_05_02_2002_CHI_1_2.wav"))
        train_sessions, validation_sessions = babyslm_session_split([mot, fat, _clip_from_name("audio/Lily_MOT/Lily_020001_01_01_2003_MOT_0_1000.wav")], 0.5, 4)
        self.assertTrue(train_sessions.isdisjoint(validation_sessions))
        self.assertTrue(_select_frames([mot, fat], {mot.session_id}, 10))

    def test_audio_manifest_allows_explicit_exposure_order_when_age_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "a.wav"
            write_wav(wav)
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "audio_path\tcorpus_id\tsession_id\tsource_corpus\tspeaker_role\tdirected_to_child\texposure_order\n"
                "a.wav\tBabySLM-Providence\tsession-1\tBabySLM\tmother\ttrue\t1\n",
                encoding="utf-8",
            )
            item = load_audio_manifest(manifest)[0]
            self.assertIsNone(item.target_child_age_months)
            self.assertEqual(item.exposure_order, 1)

    def test_cached_babyslm_manifest_is_repartitioned_without_copying_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings = root / "recordings"
            recordings.mkdir()
            rows = []
            # A cache with a few long sessions and several short sessions per
            # child: the duration-aware migration should reserve the short
            # sessions for diverse validation and retain the long training pool.
            for child in ("Alex", "Ethan", "Lily"):
                for session_index, seconds in enumerate((0.3, 0.4, 5.0)):
                    session = f"{child}_{session_index:02d}"
                    for clip_index in range(1):
                        wav = recordings / f"{session}_{clip_index}.wav"
                        write_wav(wav, seconds=seconds)
                        rows.append({
                            "audio_path": wav.relative_to(root).as_posix(), "corpus_id": "BabySLM-Providence",
                            "session_id": session, "target_child_age_months": "", "exposure_order": str(len(rows) + 1),
                            "source_corpus": "BabySLM Providence official audio.zip", "speaker_role": "mother",
                            "directed_to_child": "corpus_context", "recording_order": str(clip_index + 1),
                            "split": "train" if session_index < 2 else "validation",
                        })
            source = root / "babyslm_providence_caregiver_manifest.tsv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            original_audio = {path.relative_to(root) for path in recordings.glob("*.wav")}
            target = migrate_cached_babyslm_manifest(root, seed=31, validation_hours=0.0001)
            first_text = target.read_text(encoding="utf-8")
            self.assertEqual(target, migrate_cached_babyslm_manifest(root, seed=31, validation_hours=0.0001))
            self.assertEqual(first_text, target.read_text(encoding="utf-8"))
            items = load_audio_manifest(target)
            train_items, validation_items = split_audio_sessions(items, 0.1, 31)
            self.assertTrue({item.session_key for item in train_items}.isdisjoint({item.session_key for item in validation_items}))
            self.assertEqual({path.relative_to(root) for path in recordings.glob("*.wav")}, original_audio)
            validation_order = [item.session_id for item in validation_items]
            self.assertEqual(len(set(validation_order[:6])), 6)
            report = json.loads((root / "babyslm_repartition_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["cache_action"], "reused existing WAVs only; no audio was copied, fabricated, or downloaded")
            self.assertEqual(report["validation_sessions"], 6)
            self.assertEqual(len(report["validation_children"]), 3)
            self.assertGreater(
                report["train_hours_available"],
                0.85 * (report["train_hours_available"] + report["validation_hours_available"]),
            )
            self.assertIn("duration-aware", report["validation_selection"])
            self.assertIn("not duration-representative", report["validation_selection_duration_bias"])

    def test_validation_order_round_robins_session_groups_before_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for session in range(4):
                for clip in range(3):
                    wav = root / f"s{session}_{clip}.wav"
                    write_wav(wav)
                    rows.append({
                        "audio_path": wav.name, "corpus_id": "synthetic", "session_id": f"s{session}",
                        "target_child_age_months": str(12 + session), "source_corpus": "synthetic-test",
                        "speaker_role": "caregiver", "directed_to_child": "true", "recording_order": str(clip),
                    })
            manifest = root / "round_robin.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            ordered = round_robin_validation_items(load_audio_manifest(manifest), 42)
            self.assertEqual(len({item.session_id for item in ordered[:4]}), 4)

    def test_acoustic_smoke_training_writes_independent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root)
            config_path = root / "smoke.toml"
            output_dir = root / "outputs"
            config_path.write_text("\n".join([
                "[acoustic_phase1]", f'audio_manifest_path = "{manifest}"', f'output_dir = "{output_dir}"',
                "seed = 7", 'device = "cpu"', "validation_fraction = 0.25", "sample_rate = 16000", "n_mels = 16",
                "n_fft = 800", "win_length = 400", "hop_length = 160", "hidden_size = 8", "num_layers = 1", "dropout = 0.0",
                "learning_rate = 0.001", "gradient_clip_norm = 1.0", "sequence_chunk_frames = 32",
                "future_horizons_frames = [3, 5]", "max_train_hours = 0.00002", "max_validation_hours = 0.00002",
                "normalization_max_hours = 0.00002", "target_checkpoint_count = 2", "",
            ]), encoding="utf-8")
            metrics = train(load_config(config_path))
            output = root / "outputs"
            self.assertTrue(list(output.glob("acoustic_checkpoint_step_*.pt")))
            self.assertTrue((output / "acoustic_run_manifest.json").is_file())
            self.assertEqual(metrics["future_horizons_ms"], [30, 50])
            exposure = json.loads((output / "audio_training_exposure.json").read_text(encoding="utf-8"))
            self.assertIn("child-age order", exposure["selection"])

    def test_config_disallows_only_next_frame_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text("\n".join([
                "[acoustic_phase1]", 'audio_manifest_path = "manifest.tsv"', 'output_dir = "out"', "seed = 1", 'device = "cpu"',
                "validation_fraction = 0.2", "sample_rate = 16000", "n_mels = 80", "n_fft = 800", "win_length = 400", "hop_length = 160",
                "hidden_size = 8", "num_layers = 1", "dropout = 0.0", "learning_rate = 0.001", "gradient_clip_norm = 1.0",
                "sequence_chunk_frames = 64", "future_horizons_frames = [1]", "max_train_hours = 1.0", "max_validation_hours = 1.0", "normalization_max_hours = 1.0",
                "target_checkpoint_count = 1", "",
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "greater than 1"):
                load_config(path)

    def test_config_rejects_empty_mel_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_mel.toml"
            path.write_text("\n".join([
                "[acoustic_phase1]", 'audio_manifest_path = "manifest.tsv"', 'output_dir = "out"', "seed = 1", 'device = "cpu"',
                "validation_fraction = 0.2", "sample_rate = 16000", "n_mels = 80", "n_fft = 400", "win_length = 400", "hop_length = 160",
                "hidden_size = 8", "num_layers = 1", "dropout = 0.0", "learning_rate = 0.001", "gradient_clip_norm = 1.0",
                "sequence_chunk_frames = 64", "future_horizons_frames = [3, 5]", "max_train_hours = 1.0", "max_validation_hours = 1.0", "normalization_max_hours = 1.0",
                "target_checkpoint_count = 1", "",
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty bins"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
