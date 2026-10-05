"""Prepare an adult/caregiver-only Providence CDS WAV manifest at run time.

TalkBank credentials are accepted only as function arguments.  This module does
not print, write, or put credentials/cookies into a command line, config or
output manifest.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


LOGIN_URL = "https://sla2.talkbank.org/logInUser"
TRANSCRIPT_ZIP_URL = "https://talkbank.org/data/phon/Eng-NA/Providence?f=zip"
MEDIA_ROOT_URL = "https://media.talkbank.org/phon/Eng-NA/Providence"
ADULT_CODES = {"MOT", "FAT", "GRM", "GRF", "ADU", "INV"}
ADULT_ROLE_WORDS = {"mother", "father", "parent", "caregiver", "adult", "grandmother", "grandfather", "investigator"}
TIMESTAMP = re.compile(r"\x15(\d+)[_:](\d+)\x15")
TIER = re.compile(r"^\*([A-Za-z0-9]+):\s*(.*)$")


@dataclass(frozen=True)
class ProvidenceSegment:
    session_id: str
    target_child_age_months: float
    speaker_role: str
    media_relative_path: str
    start_ms: int
    end_ms: int
    recording_order: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _age_months(value: str) -> float | None:
    match = re.search(r"(\d+);(\d+)", value)
    return int(match.group(1)) * 12 + int(match.group(2)) if match else None


def _speaker_role(code: str, role: str) -> str | None:
    lowered = role.strip().lower()
    if code == "MOT" or "mother" in lowered:
        return "mother"
    if code == "FAT" or "father" in lowered:
        return "father"
    if code in {"GRM", "GRF"} or "grand" in lowered:
        return "grandparent"
    if code in ADULT_CODES or any(word in lowered for word in ADULT_ROLE_WORDS):
        return "adult" if code in {"ADU", "INV"} else "caregiver"
    return None


def _metadata(lines: list[str]) -> tuple[float, dict[str, str], str | None]:
    roles: dict[str, str] = {}
    child_age: float | None = None
    media: str | None = None
    for line in lines:
        if line.startswith("@Participants:"):
            for participant in line.split(":", 1)[1].split(","):
                fields = participant.strip().split(maxsplit=1)
                if fields:
                    roles[fields[0].upper()] = fields[1] if len(fields) > 1 else ""
        elif line.startswith("@ID:"):
            fields = line.split(":", 1)[1].strip().split("|")
            if len(fields) >= 4:
                code = fields[2].strip().upper()
                role = fields[7].strip() if len(fields) > 7 else ""
                roles.setdefault(code, role)
                if code == "CHI":
                    child_age = _age_months(fields[3])
        elif line.startswith("@Media:") and media is None:
            value = line.split(":", 1)[1].strip().split(",", 1)[0].strip()
            if value:
                media = value
    if child_age is None:
        raise ValueError("Providence CHAT file has no parseable CHI age")
    return child_age, roles, media


def parse_chat_segments(path: str | Path, transcript_root: str | Path, *, minimum_duration_ms: int = 100) -> list[ProvidenceSegment]:
    """Extract timestamped MOT/FAT/other adult CHAT tiers; CHI is excluded."""
    path, transcript_root = Path(path), Path(transcript_root)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    child_age, roles, media = _metadata(lines)
    if media is None:
        return []
    current_code: str | None = None
    current_text: list[str] = []
    tiers: list[tuple[str, str]] = []
    for line in lines + ["@End:"]:
        match = TIER.match(line)
        if match:
            if current_code is not None:
                tiers.append((current_code, " ".join(current_text)))
            current_code, current_text = match.group(1).upper(), [match.group(2)]
        elif current_code is not None and (line.startswith("\t") or line.startswith(" ")):
            current_text.append(line.strip())
        elif line.startswith("@") and current_code is not None:
            tiers.append((current_code, " ".join(current_text)))
            current_code, current_text = None, []
    relative_parent = path.relative_to(transcript_root).parent.as_posix()
    # The download root itself is Providence, so remove a duplicated archive root.
    if relative_parent in {"", "."} or relative_parent.lower() == "providence":
        relative_parent = ""
    elif relative_parent.lower().startswith("providence/"):
        relative_parent = relative_parent.split("/", 1)[1]
    media_name = media
    relative_media = "/".join(part for part in (relative_parent, media_name) if part)
    session_id = path.relative_to(transcript_root).with_suffix("").as_posix()
    segments: list[ProvidenceSegment] = []
    for order, (code, text) in enumerate(tiers, 1):
        role = _speaker_role(code, roles.get(code, ""))
        if role is None or code == "CHI":
            continue
        timestamps = TIMESTAMP.findall(text)
        if not timestamps:
            continue
        start_ms, end_ms = (int(value) for value in timestamps[-1])
        if end_ms - start_ms < minimum_duration_ms:
            continue
        segments.append(ProvidenceSegment(session_id, child_age, role, relative_media, start_ms, end_ms, order))
    return segments


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe transcript ZIP member: {member.filename!r}") from exc
    archive.extractall(destination)


def _login_succeeded(response) -> bool:
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return False
    return isinstance(payload, dict) and payload.get("success") is True


def _request_session(email: str, password: str, *, session_factory=None):
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - documented installation path
        raise RuntimeError("Providence preparation requires requests; install the acoustic optional dependency") from exc
    session = (session_factory or requests.Session)()
    response = session.post(LOGIN_URL, json={"email": email, "pswd": password}, timeout=60)
    if response.status_code >= 400 or not _login_succeeded(response) or not session.cookies:
        raise RuntimeError("TalkBank login failed. Check credentials/account access; no credentials were saved.")
    return session


def _download_bytes(session, url: str, destination: Path) -> None:
    response = session.get(url, stream=True, timeout=120)
    content_type = response.headers.get("Content-Type", "").lower()
    if response.status_code >= 400 or "text/html" in content_type:
        raise RuntimeError(f"TalkBank did not return media/transcript data for {url}; verify account access and corpus permissions.")
    with destination.open("wb") as handle:
        for block in response.iter_content(1 << 20):
            if block:
                handle.write(block)


def _media_candidates(relative_media: str) -> list[str]:
    path = Path(relative_media)
    if path.suffix:
        return [path.as_posix()]
    return [f"{path.as_posix()}{suffix}" for suffix in (".wav", ".mp3", ".mp4", ".mov")]


def _download_media(session, relative_media: str, cache_dir: Path) -> Path:
    for candidate in _media_candidates(relative_media):
        cached = cache_dir / candidate.replace("/", "__")
        if cached.is_file():
            return cached
        url = f"{MEDIA_ROOT_URL}/{quote(candidate)}"
        try:
            _download_bytes(session, url, cached)
            return cached
        except RuntimeError:
            cached.unlink(missing_ok=True)
    raise RuntimeError(f"No supported media file could be downloaded for {relative_media!r}")


def _session_split(segments: list[ProvidenceSegment], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    import random
    sessions = sorted({segment.session_id for segment in segments})
    if len(sessions) < 2:
        raise ValueError("Providence preparation needs at least two timestamped sessions")
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    n_validation = min(len(shuffled) - 1, max(1, round(len(shuffled) * fraction)))
    validation = set(shuffled[:n_validation])
    return set(sessions) - validation, validation


def _frames_from_duration_ms(duration_ms: int, *, sample_rate: int = 16_000, n_fft: int = 400, hop_length: int = 160) -> int:
    samples = round(duration_ms * sample_rate / 1000)
    return max(0, 1 + (samples - n_fft) // hop_length) if samples >= n_fft else 0


def _minimum_duration_for_frames(frames: int, *, sample_rate: int = 16_000, n_fft: int = 400, hop_length: int = 160) -> int:
    if frames <= 0:
        return 0
    samples = n_fft + (frames - 1) * hop_length
    return math.ceil(samples * 1000 / sample_rate)


def _select_duration(segments: list[ProvidenceSegment], sessions: set[str], maximum_frames: int) -> list[tuple[ProvidenceSegment, int]]:
    selected: list[tuple[ProvidenceSegment, int]] = []
    remaining = maximum_frames
    for segment in sorted((segment for segment in segments if segment.session_id in sessions), key=lambda s: (s.target_child_age_months, s.session_id, s.start_ms, s.recording_order)):
        if remaining <= 0:
            break
        frames = _frames_from_duration_ms(segment.duration_ms)
        if frames <= remaining:
            duration = segment.duration_ms
            remaining -= frames
        else:
            duration = min(segment.duration_ms, _minimum_duration_for_frames(remaining))
            remaining -= _frames_from_duration_ms(duration)
        selected.append((segment, duration))
    return selected


def prepare_providence(
    output_dir: str | Path, *, email: str, password: str, train_hours: float = 50.0,
    validation_hours: float = 0.5, validation_fraction: float = 0.1, seed: int = 20261005,
) -> Path:
    """Download/cut an adult-tier Providence manifest. Returns its TSV path."""
    if train_hours <= 0 or validation_hours <= 0:
        raise ValueError("train_hours and validation_hours must be positive")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = _request_session(email, password)
    with tempfile.TemporaryDirectory(prefix="devlm-providence-") as temporary:
        temporary_path = Path(temporary)
        transcript_zip = temporary_path / "providence.zip"
        _download_bytes(session, TRANSCRIPT_ZIP_URL, transcript_zip)
        if not zipfile.is_zipfile(transcript_zip):
            raise RuntimeError("TalkBank transcript endpoint did not return a ZIP archive; verify access and current endpoint.")
        transcript_root = temporary_path / "transcripts"
        transcript_root.mkdir()
        with zipfile.ZipFile(transcript_zip) as archive:
            _safe_extract(archive, transcript_root)
        segments = [segment for chat in transcript_root.rglob("*.cha") for segment in parse_chat_segments(chat, transcript_root)]
        if not segments:
            raise RuntimeError("No timestamped adult/caregiver Providence CHAT tiers were found; corpus layout may have changed.")
        train_sessions, validation_sessions = _session_split(segments, validation_fraction, seed)
        # Per-clip 25-ms analysis windows cost a few frames at every utterance
        # boundary. Select with that exact frame rule and retain one extra minute;
        # the trainer then truncates the actual selected WAV frames exactly at cap.
        safety_frames = 6_000
        chosen_train = _select_duration(segments, train_sessions, round(train_hours * 3_600_000 / 10) + safety_frames)
        chosen_validation = _select_duration(segments, validation_sessions, round(validation_hours * 3_600_000 / 10) + safety_frames)
        if not chosen_train or not chosen_validation:
            raise RuntimeError("Providence selection yielded empty train or validation audio")
        cache = temporary_path / "media"
        cache.mkdir()
        audio_dir = output_dir / "recordings"
        audio_dir.mkdir(exist_ok=True)
        rows: list[dict[str, str]] = []
        source_cache: dict[str, Path] = {}
        for split, selected in (("train", chosen_train), ("validation", chosen_validation)):
            for index, (segment, duration_ms) in enumerate(selected, 1):
                source = source_cache.get(segment.media_relative_path)
                if source is None:
                    source = _download_media(session, segment.media_relative_path, cache)
                    source_cache[segment.media_relative_path] = source
                destination = audio_dir / f"{split}_{len(rows):07d}.wav"
                command = [
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{segment.start_ms / 1000:.3f}",
                    "-i", str(source), "-t", f"{duration_ms / 1000:.3f}", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
                ]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except FileNotFoundError as exc:
                    raise RuntimeError("ffmpeg is required to cut Providence media in Colab") from exc
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(f"ffmpeg failed while cutting Providence media; stderr: {exc.stderr[-500:]}") from exc
                rows.append({
                    "audio_path": destination.relative_to(output_dir).as_posix(), "corpus_id": "Providence",
                    "session_id": segment.session_id, "target_child_age_months": f"{segment.target_child_age_months:.3f}",
                    "source_corpus": "TalkBank Providence", "speaker_role": segment.speaker_role,
                    # Providence is a parent-child naturalistic corpus; this is corpus-context metadata, not a claim
                    # that CHAT codes addressee for every individual utterance.
                    "directed_to_child": "true", "recording_order": str(index), "split": split,
                })
    manifest = output_dir / "providence_adult_cds_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_corpus": "TalkBank Providence",
        "selection": "timestamped MOT/FAT/other adult CHAT tiers only; CHI excluded",
        "directionality": "parent-child corpus context; per-utterance addressee is not encoded by CHAT",
        "requested_train_hours": train_hours,
        "requested_validation_hours": validation_hours,
        "selection_accounting": "10-ms log-Mel frame rule (16 kHz, 25-ms window, 10-ms hop) plus 60-s safety margin; trainer truncates exactly at configured caps",
        "segments": len(rows),
        "speaker_roles": dict(Counter(row["speaker_role"] for row in rows)),
        "credentials_saved": False,
    }
    # Exact exposure is computed from WAV frames later by the training runner; avoid a false duration claim here.
    (output_dir / "providence_preparation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return manifest
