"""Credential-free preparation of the official BabySLM Providence audio ZIP.

The archive is audio-only: it labels speakers in directory names but does not
provide child ages.  This module therefore creates a deterministic exposure
order, never a fabricated age trajectory.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import wave
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tqdm import tqdm


BABYSLM_PROVIDENCE_AUDIO_URL = "https://cognitive-ml.fr/downloads/baby-slm/training_sets/Providence/audio.zip"
DEFAULT_CAREGIVER_CODES = ("MOT", "FAT")
FEATURE_WIN_LENGTH = 400
FEATURE_HOP = 160


@dataclass(frozen=True)
class BabySLMClip:
    archive_name: str
    session_id: str
    speaker_code: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _clip_from_name(name: str, caregiver_codes: tuple[str, ...] = DEFAULT_CAREGIVER_CODES) -> BabySLMClip | None:
    path = PurePosixPath(name)
    parts = path.parts
    if len(parts) != 3 or parts[0] != "audio" or path.suffix.lower() != ".wav":
        return None
    folder_parts = parts[1].rsplit("_", 1)
    if len(folder_parts) != 2 or folder_parts[1] not in caregiver_codes:
        return None
    speaker_code = folder_parts[1]
    stem_parts = path.stem.rsplit("_", 3)
    if len(stem_parts) != 4 or stem_parts[-3] != speaker_code:
        return None
    try:
        start_ms, end_ms = int(stem_parts[-2]), int(stem_parts[-1])
    except ValueError:
        return None
    if end_ms <= start_ms:
        return None
    # This is a recording/session identifier, not an age. It supports leakage-safe grouping.
    session_id = stem_parts[0]
    return BabySLMClip(name, session_id, speaker_code, start_ms, end_ms)


def _frames_from_duration_ms(duration_ms: int) -> int:
    samples = round(duration_ms * 16_000 / 1000)
    return max(0, 1 + (samples - FEATURE_WIN_LENGTH) // FEATURE_HOP) if samples >= FEATURE_WIN_LENGTH else 0


def _minimum_duration_for_frames(frames: int) -> int:
    if frames <= 0:
        return 0
    return 25 + 10 * (frames - 1)


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as reader:
        samples, rate = reader.getnframes(), reader.getframerate()
    samples = samples if rate == 16_000 else max(1, round(samples * 16_000 / rate))
    return max(0, 1 + (samples - FEATURE_WIN_LENGTH) // FEATURE_HOP) if samples >= FEATURE_WIN_LENGTH else 0


def _round_robin_session_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    """Interleave session groups so a capped validation pass is not prefix-biased."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(row)
    rng = random.Random(seed)
    keys = sorted(grouped)
    rng.shuffle(keys)
    for key in keys:
        grouped[key].sort(key=lambda row: int(row.get("recording_order", "0")))
    result: list[dict[str, str]] = []
    while keys:
        next_keys: list[str] = []
        for key in keys:
            if grouped[key]:
                result.append(grouped[key].pop(0))
            if grouped[key]:
                next_keys.append(key)
        keys = next_keys
    return result


def migrate_cached_babyslm_manifest(output_dir: str | Path, *, seed: int = 20261005, validation_hours: float = 0.5) -> Path:
    """Repartition a valid extracted BabySLM cache without downloading/duplicating WAVs.

    The old initial preparation selected a deterministic archive prefix.  This
    migration treats those already-extracted clips as a fixed pool, moves whole
    sessions to validation, interleaves validation sessions, and writes a
    seeded train permutation.  It never copies, synthesizes, or re-downloads a
    WAV.  If the retained pool itself lacks enough children/sessions, the JSON
    report records that limitation instead of pretending to add diversity.
    """
    root = Path(output_dir).resolve()
    source = root / "babyslm_providence_caregiver_manifest.tsv"
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"audio_path", "session_id", "speaker_role", "source_corpus"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Existing BabySLM manifest is incomplete")
    seen: set[Path] = set()
    frames_by_path: dict[Path, int] = {}
    for row in rows:
        path = (root / row["audio_path"]).resolve()
        if root not in path.parents or path.suffix.lower() != ".wav" or not path.is_file() or path in seen:
            raise ValueError("Existing BabySLM cache has missing, unsafe, non-WAV, or duplicate audio paths")
        seen.add(path)
        frames_by_path[path] = _wav_frames(path)
    by_session: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row)
    frames_by_session = {
        session: sum(frames_by_path[(root / row["audio_path"]).resolve()] for row in session_rows)
        for session, session_rows in by_session.items()
    }
    by_child: dict[str, list[str]] = {}
    for session in sorted(by_session):
        by_child.setdefault(session.split("_", 1)[0], []).append(session)
    # Existing caches can be only ~50.5 h. Holding out a fixed fraction of
    # sessions wastes several hours. Prefer two short, session-disjoint groups
    # per available child identifier, then add only short sessions required to
    # fill the requested validation pool. This is intentionally duration-biased
    # and must not be interpreted as a representative child/session sample.
    requested_validation_frames = round(validation_hours * 3_600_000 / 10)
    sessions_per_child_target = 2
    validation_sessions: list[str] = []
    for child in sorted(by_child):
        remaining_capacity = len(by_session) - 1 - len(validation_sessions)
        if remaining_capacity <= 0:
            break
        candidates = sorted(by_child[child], key=lambda session: (frames_by_session[session], session))
        validation_sessions.extend(candidates[:min(sessions_per_child_target, len(candidates), remaining_capacity)])
    validation_set = set(validation_sessions)
    validation_frames = sum(frames_by_session[session] for session in validation_set)
    remaining_sessions = sorted(
        (session for session in by_session if session not in validation_set),
        key=lambda session: (frames_by_session[session], session),
    )
    for session in remaining_sessions:
        if validation_frames >= requested_validation_frames or len(validation_set) >= len(by_session) - 1:
            break
        validation_set.add(session)
        validation_frames += frames_by_session[session]
    validation_rows = _round_robin_session_rows([row for row in rows if row["session_id"] in validation_set], seed)
    train_rows = [row for row in rows if row["session_id"] not in validation_set]
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    # Keep the source WAVs; only write a new metadata view with a deterministic
    # clip order. Training's exact caps enforce final 50h/0.5h exposure.
    reordered = [("train", row) for row in train_rows] + [("validation", row) for row in validation_rows]
    for order, (split, row) in enumerate(reordered, 1):
        row["split"] = split
        row["exposure_order"] = str(order)
        row["recording_order"] = str(order)
    target = root / "babyslm_providence_repartitioned_manifest.tsv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows([row for _, row in reordered])
    train_frames = sum(frames_by_path[(root / row["audio_path"]).resolve()] for row in train_rows)
    validation_children = sorted({session.split("_", 1)[0] for session in validation_set})
    (root / "babyslm_repartition_manifest.json").write_text(json.dumps({
        "source_manifest": source.name, "output_manifest": target.name, "seed": seed,
        "train_sessions": len({row["session_id"] for row in train_rows}),
        "validation_sessions": len(validation_set), "validation_children": validation_children,
        "train_hours_available": train_frames * 10 / 3_600_000,
        "validation_hours_available": validation_frames * 10 / 3_600_000,
        "requested_validation_hours": validation_hours,
        "requested_validation_frames": requested_validation_frames,
        "validation_pool_meets_requested_duration": validation_frames >= requested_validation_frames,
        "validation_selection": "duration-aware: up to two shortest session groups per inferred child identifier, then shortest remaining sessions until the requested validation duration",
        "validation_sessions_per_child_target": sessions_per_child_target,
        "validation_selection_duration_bias": "Validation-session selection deliberately favors shorter sessions to retain training exposure; it is not duration-representative.",
        "validation_order": "seeded round-robin across selected session groups",
        "training_order": "seeded random permutation of train clips",
        "cache_action": "reused existing WAVs only; no audio was copied, fabricated, or downloaded",
        "diversity_caveat": (
            "The retained cache contains fewer than two child identifiers inferred from session ids; "
            "validation is still session-disjoint and round-robin, but cannot be child-diverse without additional cached audio."
            if len(validation_children) < 2 else None
        ),
    }, indent=2) + "\n", encoding="utf-8")
    return target


def _session_split(clips: list[BabySLMClip], validation_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    sessions = sorted({clip.session_id for clip in clips})
    if len(sessions) < 2:
        raise ValueError("BabySLM Providence selection requires at least two clip session groups")
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    n_validation = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    validation = set(shuffled[:n_validation])
    return set(sessions) - validation, validation


def _select_frames(clips: list[BabySLMClip], sessions: set[str], maximum_frames: int) -> list[tuple[BabySLMClip, int]]:
    selected: list[tuple[BabySLMClip, int]] = []
    remaining = maximum_frames
    for clip in sorted((clip for clip in clips if clip.session_id in sessions), key=lambda clip: (clip.session_id, clip.start_ms, clip.end_ms, clip.archive_name)):
        if remaining <= 0:
            break
        frames = _frames_from_duration_ms(clip.duration_ms)
        if not frames:
            continue
        if frames <= remaining:
            selected.append((clip, clip.duration_ms))
            remaining -= frames
        else:
            selected.append((clip, min(clip.duration_ms, _minimum_duration_for_frames(remaining))))
            remaining = 0
    return selected


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe BabySLM ZIP member: {name!r}")
    return path


def download_archive(destination: str | Path, *, url: str = BABYSLM_PROVIDENCE_AUDIO_URL) -> Path:
    """Stream the public ZIP with visible progress and validate its ZIP structure."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("BabySLM download requires requests; install the acoustic optional dependency") from exc
    destination = Path(destination)
    try:
        response = requests.get(url, stream=True, timeout=(30, 300))
    except requests.RequestException as exc:
        raise RuntimeError("BabySLM Providence archive request failed or timed out") from exc
    if response.status_code != 200:
        raise RuntimeError(f"BabySLM Providence archive request failed with HTTP {response.status_code}")
    content_type = response.headers.get("Content-Type", "").lower()
    if "zip" not in content_type and "octet-stream" not in content_type:
        raise RuntimeError("BabySLM Providence URL did not return a ZIP-like content type")
    total = int(response.headers.get("Content-Length", 0) or 0)
    with destination.open("wb") as handle, tqdm(total=total or None, unit="B", unit_scale=True, desc="Downloading BabySLM Providence audio.zip") as progress:
        for block in response.iter_content(1 << 20):
            if block:
                handle.write(block)
                progress.update(len(block))
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError("BabySLM Providence download is not a valid ZIP archive")
    return destination


def prepare_babyslm_providence(
    output_dir: str | Path, *, train_hours: float = 50.0, validation_hours: float = 0.5,
    validation_fraction: float = 0.1, seed: int = 20261005,
    archive_url: str = BABYSLM_PROVIDENCE_AUDIO_URL,
) -> Path:
    """Create a 50h/0.5h caregiver-code-only PCM WAV manifest from BabySLM."""
    if train_hours <= 0 or validation_hours <= 0:
        raise ValueError("train_hours and validation_hours must be positive")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "babyslm_providence_audio.zip"
    print("BabySLM 1/4: downloading the official Providence audio archive (about 13 GB)...", flush=True)
    download_archive(archive_path, url=archive_url)
    print("BabySLM 2/4: inspecting archive entries and selecting only MOT/FAT-labelled clips...", flush=True)
    with zipfile.ZipFile(archive_path) as archive:
        clips = [clip for info in archive.infolist() if (clip := _clip_from_name(info.filename)) is not None]
        if not clips:
            raise RuntimeError("No MOT/FAT-labelled WAV clips found in the BabySLM Providence archive")
        train_sessions, validation_sessions = _session_split(clips, validation_fraction, seed)
        safety_frames = 6_000
        chosen_train = _select_frames(clips, train_sessions, round(train_hours * 360_000) + safety_frames)
        chosen_validation = _select_frames(clips, validation_sessions, round(validation_hours * 360_000) + safety_frames)
        if not chosen_train or not chosen_validation:
            raise RuntimeError("BabySLM session split did not yield usable train and validation clips")
        selected = [("train", clip, duration) for clip, duration in chosen_train] + [("validation", clip, duration) for clip, duration in chosen_validation]
        print(
            f"BabySLM 2/4 complete: {len(clips):,} MOT/FAT clips from {len({clip.session_id for clip in clips}):,} session groups; "
            f"selecting {len(chosen_train):,} train and {len(chosen_validation):,} validation clips.", flush=True,
        )
        recordings = output_dir / "recordings"
        recordings.mkdir(exist_ok=True)
        rows: list[dict[str, str]] = []
        prepared_frames = 0
        print("BabySLM 3/4: extracting selected WAV clips...", flush=True)
        for exposure_order, (split, clip, duration_ms) in enumerate(tqdm(selected, desc="Extracting BabySLM caregiver clips", unit="clip", dynamic_ncols=True), 1):
            member = _safe_member_path(clip.archive_name)
            destination = recordings / f"{split}_{exposure_order:07d}.wav"
            with archive.open(member.as_posix()) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            # The archive already contains speaker-turn clips. Do not crop WAV
            # bytes again: duration is applied as a final cap in the trainer.
            role = "mother" if clip.speaker_code == "MOT" else "father"
            rows.append({
                "audio_path": destination.relative_to(output_dir).as_posix(), "corpus_id": "BabySLM-Providence",
                "session_id": clip.session_id, "target_child_age_months": "", "exposure_order": str(exposure_order),
                "source_corpus": "BabySLM Providence official audio.zip", "speaker_role": role,
                "directed_to_child": "corpus_context", "recording_order": str(exposure_order), "split": split,
            })
            prepared_frames += _frames_from_duration_ms(duration_ms)
    print(f"BabySLM 3/4 complete: extracted {len(rows):,} clips (~{prepared_frames * 10 / 3_600_000:.3f} h before final cap).", flush=True)
    # The public archive is reproducibly available by URL and is not a training
    # output. Reclaim Colab disk before the long training phase.
    archive_path.unlink(missing_ok=True)
    print("BabySLM 3/4: removed temporary 13-GB archive after validated extraction.", flush=True)
    manifest = output_dir / "babyslm_providence_caregiver_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": "official BabySLM Providence audio.zip",
        "archive_url": archive_url,
        "speaker_filter": "only archive directories labelled MOT or FAT; all other speaker codes excluded",
        "directed_to_child_caveat": "MOT/FAT identify speaker role in archive paths; per-utterance addressee is not supplied by this audio-only archive",
        "age_caveat": "archive contains no child-age metadata used here; exposure_order is deterministic clip/session order, not developmental age",
        "train_selection": "deterministic session-disjoint split then 50-hour target plus 60-second safety margin; trainer applies exact cap",
        "seed": seed,
        "speaker_codes_found": dict(Counter(clip.speaker_code for clip in clips)),
        "clips_extracted": len(rows),
    }
    (output_dir / "babyslm_preparation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("BabySLM 4/4 complete: caregiver manifest ready.", flush=True)
    return manifest
