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

from tqdm import tqdm


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


class ChatMetadataError(ValueError):
    """A single unusable CHAT file; callers may skip it without guessing metadata."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


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
        raise ChatMetadataError("missing_chi_age", "Providence CHAT file has no parseable CHI age")
    return child_age, roles, media


def parse_chat_segments(path: str | Path, transcript_root: str | Path, *, minimum_duration_ms: int = 100) -> list[ProvidenceSegment]:
    """Extract timestamped MOT/FAT/other adult CHAT tiers; CHI is excluded."""
    path, transcript_root = Path(path), Path(transcript_root)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    child_age, roles, media = _metadata(lines)
    if media is None:
        raise ChatMetadataError("missing_media", "Providence CHAT file has no @Media declaration")
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
    adult_tier_count = 0
    timestamped_adult_tier_count = 0
    for order, (code, text) in enumerate(tiers, 1):
        role = _speaker_role(code, roles.get(code, ""))
        if role is None or code == "CHI":
            continue
        adult_tier_count += 1
        timestamps = TIMESTAMP.findall(text)
        if not timestamps:
            continue
        timestamped_adult_tier_count += 1
        start_ms, end_ms = (int(value) for value in timestamps[-1])
        if end_ms - start_ms < minimum_duration_ms:
            continue
        segments.append(ProvidenceSegment(session_id, child_age, role, relative_media, start_ms, end_ms, order))
    if not adult_tier_count:
        raise ChatMetadataError("no_eligible_adult_tier", "Providence CHAT file has no MOT/FAT/eligible adult tier")
    if not timestamped_adult_tier_count:
        raise ChatMetadataError("missing_adult_timestamps", "Providence CHAT file has no timestamped eligible adult tier")
    return segments


def parse_providence_corpus(transcript_root: str | Path) -> tuple[list[ProvidenceSegment], Counter[str], dict[str, str]]:
    """Parse a corpus defensively: malformed individual files never end a run."""
    transcript_root = Path(transcript_root)
    skipped: Counter[str] = Counter()
    first_path: dict[str, str] = {}
    segments: list[ProvidenceSegment] = []
    for chat in sorted(transcript_root.rglob("*.cha")):
        relative = chat.relative_to(transcript_root).as_posix()
        try:
            segments.extend(parse_chat_segments(chat, transcript_root))
        except ChatMetadataError as exc:
            skipped[exc.reason] += 1
            first_path.setdefault(exc.reason, relative)
        except (UnicodeError, ValueError):
            # Do not manufacture a child age or media mapping from a filename.
            skipped["malformed_chat"] += 1
            first_path.setdefault("malformed_chat", relative)
    return segments, skipped, first_path


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
    try:
        response = session.post(LOGIN_URL, json={"email": email, "pswd": password}, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError("TalkBank login request failed or timed out; check network access and try again.") from exc
    if response.status_code >= 400 or not _login_succeeded(response) or not session.cookies:
        raise RuntimeError("TalkBank login failed. Check credentials/account access; no credentials were saved.")
    return session


def _download_bytes(session, url: str, destination: Path) -> None:
    try:
        response = session.get(url, stream=True, timeout=120)
    except Exception as exc:
        raise RuntimeError("TalkBank media/transcript download failed or timed out; check network access and try again.") from exc
    content_type = response.headers.get("Content-Type", "").lower()
    rejected_types = ("text/", "application/json", "application/xml", "text/xml")
    if response.status_code >= 400 or any(marker in content_type for marker in rejected_types):
        raise RuntimeError(f"TalkBank did not return media/transcript data for {url}; verify account access and corpus permissions.")
    with destination.open("wb") as handle:
        for block in response.iter_content(1 << 20):
            if block:
                handle.write(block)


def _media_candidates(relative_media: str) -> list[str]:
    path = Path(relative_media)
    stem = path.with_suffix("").as_posix() if path.suffix else path.as_posix()
    # CHAT may name a stale/redirecting extension. Preserve it first, then try
    # the same base across TalkBank's usual audio/video containers.
    candidates = [path.as_posix()] + [f"{stem}{suffix}" for suffix in (".wav", ".mp3", ".mp4", ".mov")]
    return list(dict.fromkeys(candidates))


def _probe_decodable_audio(path: Path) -> tuple[bool, str]:
    """Require an audio stream with positive duration before admitting a cache file."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_type:format=duration", "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to validate TalkBank media in Colab") from exc
    if result.returncode != 0:
        return False, "ffprobe could not decode file"
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        duration = float(payload.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, "ffprobe returned invalid metadata"
    if not streams:
        return False, "no audio stream"
    if not duration > 0:
        return False, "non-positive duration"
    return True, f"duration={duration:.3f}s"


def _download_media(session, relative_media: str, cache_dir: Path, *, downloader=_download_bytes, probe=_probe_decodable_audio) -> Path:
    failures: list[str] = []
    for candidate in _media_candidates(relative_media):
        cached = cache_dir / candidate.replace("/", "__")
        if cached.is_file():
            valid, reason = probe(cached)
            if valid:
                return cached
            cached.unlink(missing_ok=True)
            failures.append(f"{candidate}: cached {reason}")
        url = f"{MEDIA_ROOT_URL}/{quote(candidate)}"
        try:
            downloader(session, url, cached)
            valid, reason = probe(cached)
            if valid:
                return cached
            failures.append(f"{candidate}: {reason}")
        except RuntimeError as exc:
            failures.append(f"{candidate}: download rejected")
        finally:
            # A valid return occurs before finally; all invalid/error candidates
            # must not poison a later retry or be mistaken for media cache.
            if cached.is_file():
                valid, _ = probe(cached)
                if not valid:
                    cached.unlink(missing_ok=True)
    detail = "; ".join(failures) if failures else "no candidates"
    raise RuntimeError(f"No decodable TalkBank media was available for {relative_media!r}. Tried: {detail}")


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


def _frames_from_duration_ms(duration_ms: int, *, sample_rate: int = 16_000, n_fft: int = 800, win_length: int = 400, hop_length: int = 160) -> int:
    samples = round(duration_ms * sample_rate / 1000)
    if win_length > n_fft:
        raise ValueError("win_length must not exceed n_fft")
    return max(0, 1 + (samples - win_length) // hop_length) if samples >= win_length else 0


def _minimum_duration_for_frames(frames: int, *, sample_rate: int = 16_000, n_fft: int = 800, win_length: int = 400, hop_length: int = 160) -> int:
    if frames <= 0:
        return 0
    if win_length > n_fft:
        raise ValueError("win_length must not exceed n_fft")
    samples = win_length + (frames - 1) * hop_length
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
    print("Providence 1/5: authenticating to TalkBank...", flush=True)
    session = _request_session(email, password)
    print("Providence 1/5 complete: login accepted.", flush=True)
    with tempfile.TemporaryDirectory(prefix="devlm-providence-") as temporary:
        temporary_path = Path(temporary)
        transcript_zip = temporary_path / "providence.zip"
        print("Providence 2/5: downloading official transcript archive...", flush=True)
        _download_bytes(session, TRANSCRIPT_ZIP_URL, transcript_zip)
        if not zipfile.is_zipfile(transcript_zip):
            raise RuntimeError("TalkBank transcript endpoint did not return a ZIP archive; verify access and current endpoint.")
        transcript_root = temporary_path / "transcripts"
        transcript_root.mkdir()
        with zipfile.ZipFile(transcript_zip) as archive:
            _safe_extract(archive, transcript_root)
        chat_files = sorted(transcript_root.rglob("*.cha"))
        print(f"Providence 3/5: parsing {len(chat_files):,} CHAT transcripts for timestamped adult tiers...", flush=True)
        segments, skipped_chats, first_skipped_path = parse_providence_corpus(transcript_root)
        if skipped_chats:
            skip_report = "; ".join(
                f"{reason}={count} (first: {first_skipped_path[reason]})"
                for reason, count in sorted(skipped_chats.items())
            )
            print(f"Providence 3/5: skipped unusable CHAT files: {skip_report}", flush=True)
        if not segments:
            raise RuntimeError(
                "No usable timestamped adult/caregiver Providence CHAT tiers remain after skipping unusable files. "
                "Check corpus access/layout; no child age or media mapping was inferred from filenames."
            )
        print(
            f"Providence 3/5 complete: {len(segments):,} adult/caregiver segments from "
            f"{len({segment.session_id for segment in segments}):,} sessions; CHI segments excluded.",
            flush=True,
        )
        if len({segment.session_id for segment in segments}) < 2:
            raise RuntimeError("Fewer than two usable Providence sessions remain after CHAT validation; cannot make a session-level train/validation split.")
        train_sessions, validation_sessions = _session_split(segments, validation_fraction, seed)
        # Each separately cut utterance uses the same 25-ms-window/10-ms-hop
        # frame accounting as training; its 800-point DFT is zero-padding only.
        # Select with that exact frame rule and retain one extra minute;
        # the trainer then truncates the actual selected WAV frames exactly at cap.
        safety_frames = 6_000
        chosen_train = _select_duration(segments, train_sessions, round(train_hours * 3_600_000 / 10) + safety_frames)
        chosen_validation = _select_duration(segments, validation_sessions, round(validation_hours * 3_600_000 / 10) + safety_frames)
        if not chosen_train or not chosen_validation:
            raise RuntimeError("Providence selection yielded empty train or validation audio")
        selected = [("train", segment, duration) for segment, duration in chosen_train] + [("validation", segment, duration) for segment, duration in chosen_validation]
        expected_train_frames = sum(_frames_from_duration_ms(duration) for _, duration in chosen_train)
        expected_validation_frames = sum(_frames_from_duration_ms(duration) for _, duration in chosen_validation)
        print(
            f"Providence 4/5: selected {len(chosen_train):,} train segments "
            f"(~{expected_train_frames * 10 / 3_600_000:.3f} h) and {len(chosen_validation):,} validation segments "
            f"(~{expected_validation_frames * 10 / 3_600_000:.3f} h). Downloading and cutting media...",
            flush=True,
        )
        cache = temporary_path / "media"
        cache.mkdir()
        audio_dir = output_dir / "recordings"
        audio_dir.mkdir(exist_ok=True)
        rows: list[dict[str, str]] = []
        source_cache: dict[str, Path] = {}
        completed_frames = 0
        progress = tqdm(selected, desc="Providence 5/5: cutting adult CDS WAV", unit="segment", dynamic_ncols=True)
        split_orders: Counter[str] = Counter()
        for split, segment, duration_ms in progress:
            split_orders[split] += 1
            index = split_orders[split]
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
                # Providence is a parent-child corpus; CHAT has no utterance-level addressee code.
                "directed_to_child": "true", "recording_order": str(index), "split": split,
            })
            completed_frames += _frames_from_duration_ms(duration_ms)
            progress.set_postfix(
                split=split,
                prepared_hours=f"{completed_frames * 10 / 3_600_000:.3f}",
                source=Path(segment.media_relative_path).name[:28],
            )
        print(f"Providence 5/5 complete: prepared {len(rows):,} PCM WAV segments (~{completed_frames * 10 / 3_600_000:.3f} h before final training cap).", flush=True)
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
