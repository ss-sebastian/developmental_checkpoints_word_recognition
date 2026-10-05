from __future__ import annotations

import csv
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from devlm.acoustic.config import load_config
from devlm.acoustic.data import load_audio_manifest, split_audio_sessions
from devlm.acoustic.features import log_mel_spectrogram
from devlm.acoustic.providence import _frames_from_duration_ms, _request_session, parse_chat_segments
from devlm.acoustic.train import FRAME_MS, train


def write_wav(path: Path, seconds: float = 0.08, sample_rate: int = 16_000) -> None:
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
        features = log_mel_spectrogram(waveform, 16_000, target_sample_rate=16_000, n_mels=80, n_fft=400, hop_length=160)
        self.assertEqual(FRAME_MS, 10)
        self.assertEqual(features.shape, (8, 80))
        self.assertTrue(torch.isfinite(features).all())

    def test_manifest_rejects_child_or_not_directed_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "not explicitly child-directed"):
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
        # 100 ms contains ten 10-ms hops, but an independent 25-ms window gives only eight frames.
        self.assertEqual(_frames_from_duration_ms(100), 8)
        self.assertEqual(_frames_from_duration_ms(1000), 98)

    def test_acoustic_smoke_training_writes_independent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root)
            config_path = root / "smoke.toml"
            output_dir = root / "outputs"
            config_path.write_text("\n".join([
                "[acoustic_phase1]", f'audio_manifest_path = "{manifest}"', f'output_dir = "{output_dir}"',
                "seed = 7", 'device = "cpu"', "validation_fraction = 0.25", "sample_rate = 16000", "n_mels = 16",
                "n_fft = 128", "hop_length = 160", "hidden_size = 8", "num_layers = 1", "dropout = 0.0",
                "learning_rate = 0.001", "gradient_clip_norm = 1.0", "sequence_chunk_frames = 32",
                "future_horizons_frames = [3, 5]", "max_train_hours = 0.00002", "max_validation_hours = 0.00002",
                "normalization_max_hours = 0.00002", "target_checkpoint_count = 2", "",
            ]), encoding="utf-8")
            metrics = train(load_config(config_path))
            output = root / "outputs"
            self.assertTrue(list(output.glob("acoustic_checkpoint_step_*.pt")))
            self.assertTrue((output / "acoustic_run_manifest.json").is_file())
            self.assertEqual(metrics["future_horizons_ms"], [30, 50])

    def test_config_disallows_only_next_frame_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text("\n".join([
                "[acoustic_phase1]", 'audio_manifest_path = "manifest.tsv"', 'output_dir = "out"', "seed = 1", 'device = "cpu"',
                "validation_fraction = 0.2", "sample_rate = 16000", "n_mels = 80", "n_fft = 400", "hop_length = 160",
                "hidden_size = 8", "num_layers = 1", "dropout = 0.0", "learning_rate = 0.001", "gradient_clip_norm = 1.0",
                "sequence_chunk_frames = 64", "future_horizons_frames = [1]", "max_train_hours = 1.0", "max_validation_hours = 1.0", "normalization_max_hours = 1.0",
                "target_checkpoint_count = 1", "",
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "greater than 1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
