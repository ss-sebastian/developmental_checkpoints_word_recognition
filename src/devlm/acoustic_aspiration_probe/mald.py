"""Official MALD acquisition and minimal, auditable TextGrid parsing."""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


# UAlberta Scholaris / MALD deposits.  Checksums and byte sizes are pinned so
# a resumed Colab run cannot silently use an HTML error page or revised asset.
MALD_RESOURCES = {
    "words_audio": ("https://ualberta.scholaris.ca/server/api/core/bitstreams/bc2dfd53-0a66-4baa-b682-f37e463d7973/content", 1_114_689_099, "d334cc24e264cfb10789b3a867a41ced"),
    "pseudowords_audio": ("https://ualberta.scholaris.ca/server/api/core/bitstreams/74e01434-a108-400b-a9f4-0e00fb9b526c/content", 418_915_294, "027b18ca30178962965e477a18e36464"),
    "words_textgrids": ("https://ualberta.scholaris.ca/server/api/core/bitstreams/71874084-4afc-4346-8b5c-198fbf079039/content", 10_464_955, "236583b872057bdc44be2b89cbbb2545"),
    "pseudowords_textgrids": ("https://ualberta.scholaris.ca/server/api/core/bitstreams/669cd367-4eed-4609-a259-f6c6a2c0b2f5/content", 3_812_183, "b53aece5ec5b985f8d17133f6284aa48"),
}


@dataclass(frozen=True)
class PhoneInterval:
    start_s: float
    end_s: float
    label: str


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract zip only after rejecting traversal/symlink members."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe MALD ZIP member: {member.filename!r}")
        bundle.extractall(destination)


def download_resource(name: str, cache_dir: str | Path, *, timeout_s: int = 60) -> Path:
    """Download one pinned MALD ZIP with streaming progress and cache checks."""
    if name not in MALD_RESOURCES:
        raise KeyError(name)
    url, expected_bytes, expected_md5 = MALD_RESOURCES[name]
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{name}.zip"
    if target.is_file() and target.stat().st_size == expected_bytes and _md5(target) == expected_md5:
        print(f"MALD cache verified: {target.name}", flush=True)
        return target
    partial = target.with_suffix(".part")
    resume_at = partial.stat().st_size if partial.exists() else 0
    if resume_at >= expected_bytes:
        partial.unlink(missing_ok=True); resume_at = 0
    print(f"Downloading official MALD {name.replace('_', ' ')}…", flush=True)
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    with requests.get(url, headers=headers, stream=True, timeout=(15, timeout_s)) as response:
        # A compliant resumed response must identify the requested range.  A
        # 200 means this mirror ignored Range, so deliberately restart instead
        # of appending a second copy of the archive.
        if resume_at and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {resume_at}-")):
            partial.unlink(missing_ok=True); resume_at = 0
            response.close()
            return download_resource(name, cache_dir, timeout_s=timeout_s)
        response.raise_for_status()
        mode = "ab" if resume_at else "wb"
        with partial.open(mode) as handle, tqdm(total=expected_bytes, initial=resume_at, unit="B", unit_scale=True, desc=name, dynamic_ncols=True) as bar:
            for block in response.iter_content(1 << 20):
                if block:
                    handle.write(block); bar.update(len(block))
    if partial.stat().st_size != expected_bytes or _md5(partial) != expected_md5:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Official MALD {name} failed pinned size/MD5 validation")
    partial.replace(target)
    return target


def prepare_mald(cache_dir: str | Path) -> tuple[Path, Path]:
    """Download/extract official MALD audio and phone TextGrids, resumably."""
    cache = Path(cache_dir); downloads = cache / "downloads"; extracted = cache / "extracted"
    for name in MALD_RESOURCES:
        archive = download_resource(name, downloads)
        marker = extracted / f".{name}.complete"
        if not marker.exists():
            print(f"Extracting {name.replace('_', ' ')}…", flush=True)
            _safe_extract(archive, extracted)
            marker.write_text("verified official archive\n", encoding="utf-8")
    audio = extracted / "recordings"
    if not audio.exists():
        # Official archive roots have changed between MALD deposits. Search
        # rather than encode an unverified layout, then validate actual WAVs.
        wavs = list(extracted.rglob("*.wav"))
        if not wavs:
            raise FileNotFoundError("MALD audio archives extracted but no WAV files were found")
        audio = extracted
    textgrids = extracted
    if not any(textgrids.rglob("*.TextGrid")):
        raise FileNotFoundError("MALD TextGrid archives extracted but no TextGrid files were found")
    print(f"MALD ready: {sum(1 for _ in audio.rglob('*.wav')):,} WAVs; {sum(1 for _ in textgrids.rglob('*.TextGrid')):,} TextGrids", flush=True)
    return audio, textgrids


def parse_phone_textgrid(path: str | Path) -> list[PhoneInterval]:
    """Read MALD's short ooTextFile ``phone`` tier, merge adjacent equals.

    Silence labels and blank intervals are removed.  This intentionally keeps
    phone edges, matching Martin et al.; no central-frame trimming is applied.
    """
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    try:
        phone_at = lines.index('"phone"')
    except ValueError:
        return []
    # MALD uses short TextGrid: tier name, xmin, xmax, count, then triples.
    try:
        count = int(float(lines[phone_at + 3]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed MALD phone tier: {path}") from exc
    start = phone_at + 4; intervals: list[PhoneInterval] = []
    silences = {"", "sil", "sp", "spn", "pau", "<sil>"}
    for index in range(count):
        try:
            begin, end, raw = lines[start + 3 * index:start + 3 * index + 3]
            label = raw.strip().strip('"').upper()
            interval = PhoneInterval(float(begin), float(end), label)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Malformed phone interval {index} in {path}") from exc
        if interval.end_s <= interval.start_s or label.lower() in silences:
            continue
        if intervals and intervals[-1].label == label and abs(intervals[-1].end_s - interval.start_s) < 1e-6:
            previous = intervals[-1]
            intervals[-1] = PhoneInterval(previous.start_s, interval.end_s, label)
        else:
            intervals.append(interval)
    return intervals
