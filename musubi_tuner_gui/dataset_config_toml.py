"""Pure TOML logic for musubi-tuner dataset config files.

No Gradio imports here on purpose: this module is unit-testable headlessly
and owns everything about reading/writing/validating the dataset TOML
described in musubi-tuner/docs/dataset_config.md.
"""

import os
import re

import toml

# Keys the UI models directly in the [general] table.
GENERAL_KNOWN_KEYS = [
    "resolution",
    "caption_extension",
    "batch_size",
    "num_repeats",
    "enable_bucket",
    "bucket_no_upscale",
]

# Keys the UI models directly in each [[datasets]] entry. Includes the
# per-dataset overrides of GENERAL_KNOWN_KEYS plus dataset-only fields.
DATASET_KNOWN_KEYS = GENERAL_KNOWN_KEYS + [
    "image_directory",
    "image_jsonl_file",
    "video_directory",
    "video_jsonl_file",
    "cache_directory",
    "control_directory",
    "target_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
    "source_fps",
    "no_resize_control",
    "control_resolution",
]

SOURCE_KEYS = [
    "image_directory",
    "image_jsonl_file",
    "video_directory",
    "video_jsonl_file",
]

VIDEO_SOURCE_KEYS = ["video_directory", "video_jsonl_file"]
JSONL_SOURCE_KEYS = ["image_jsonl_file", "video_jsonl_file"]

FRAME_EXTRACTION_CHOICES = ["head", "chunk", "slide", "uniform", "full"]

# Mirrors musubi-tuner/src/musubi_tuner/dataset/media_utils.py's IMAGE_EXTENSIONS /
# VIDEO_EXTENSIONS (lowercased here since matching below is case-insensitive).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def detect_caption_extension(directory: str):
    """Guess the caption file extension used in a dataset directory.

    Looks for files that share a basename with an image/video file but have a
    different extension (e.g. `foo.jpg` + `foo.txt`), and returns the most
    common such extension. Returns None if the directory doesn't exist or no
    caption-like sibling files are found.
    """
    if not directory or not os.path.isdir(directory):
        return None

    try:
        entries = os.listdir(directory)
    except OSError:
        return None

    exts_by_stem = {}
    for name in entries:
        full_path = os.path.join(directory, name)
        if not os.path.isfile(full_path):
            continue
        stem, ext = os.path.splitext(name)
        exts_by_stem.setdefault(stem, set()).add(ext.lower())

    counts = {}
    for exts in exts_by_stem.values():
        if not exts & MEDIA_EXTENSIONS:
            continue
        for ext in exts - MEDIA_EXTENSIONS:
            if not ext:
                continue
            counts[ext] = counts.get(ext, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)


def load_dataset_config(path: str) -> dict:
    """Parse a dataset TOML file into {"general": {...}, "datasets": [...]}."""
    data = toml.load(path)
    general = dict(data.get("general", {}) or {})
    datasets = [dict(d) for d in (data.get("datasets", []) or [])]
    return {"general": general, "datasets": datasets}


def _coerce_int_pair(value, key):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = parse_int_pair(value)
    pair = list(value)
    if len(pair) != 2:
        raise ValueError(f"{key} must have exactly 2 values, got {pair}")
    return [int(pair[0]), int(pair[1])]


def _coerce_int_list(value, key):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = parse_int_list(value)
    return [int(v) for v in value]


def _coerce_value(key, value):
    """Coerce a single known key's value to the type dataset_config.md expects."""
    if value is None or value == "":
        return None
    if key in ("resolution", "control_resolution"):
        return _coerce_int_pair(value, key)
    if key == "target_frames":
        return _coerce_int_list(value, key)
    if key == "source_fps":
        return float(value)
    if key in (
        "batch_size",
        "num_repeats",
        "frame_stride",
        "frame_sample",
        "max_frames",
    ):
        return int(value)
    if key in ("enable_bucket", "bucket_no_upscale", "no_resize_control"):
        return bool(value)
    return value


def _build_entry(known_keys, source: dict) -> dict:
    """Build a TOML-ready dict: known keys coerced/dropped-if-empty, unknown pass through."""
    entry = {}
    for key, value in source.items():
        if key in known_keys:
            coerced = _coerce_value(key, value)
            if coerced is None:
                continue
            entry[key] = coerced
        else:
            # Unknown/advanced key: pass through untouched so it round-trips.
            entry[key] = value
    return entry


def save_dataset_config(path: str, general: dict, datasets: list) -> None:
    """Write [general] + [[datasets]] to path, dropping None/empty known fields."""
    general = general or {}
    datasets = datasets or []

    out = {}

    general_entry = _build_entry(GENERAL_KNOWN_KEYS, general)
    if general_entry:
        out["general"] = general_entry

    out["datasets"] = [_build_entry(DATASET_KNOWN_KEYS, d) for d in datasets]

    with open(path, "w", encoding="utf-8") as f:
        toml.dump(out, f)


def _n_times_4_plus_1(n: int) -> bool:
    return n >= 1 and (n - 1) % 4 == 0


def validate_dataset_config(general: dict, datasets: list) -> list:
    """Return human-readable ERROR:/WARNING: messages per dataset_config.md rules."""
    general = general or {}
    datasets = datasets or []
    messages = []

    cache_dirs_seen = {}

    for i, ds in enumerate(datasets):
        label = f"Dataset {i}"

        sources_present = [k for k in SOURCE_KEYS if ds.get(k)]
        if not sources_present:
            messages.append(
                f"ERROR: {label}: no source path set (image/video directory or jsonl file)."
            )
        elif len(sources_present) > 1:
            messages.append(
                f"ERROR: {label}: more than one source type set ({', '.join(sources_present)})."
            )

        is_jsonl = any(k in JSONL_SOURCE_KEYS for k in sources_present)
        is_video = any(k in VIDEO_SOURCE_KEYS for k in sources_present)

        if is_jsonl and not ds.get("cache_directory"):
            messages.append(f"ERROR: {label}: jsonl source requires cache_directory.")

        caption_extension = ds.get("caption_extension") or general.get(
            "caption_extension"
        )
        is_directory_source = any(
            k in ("image_directory", "video_directory") for k in sources_present
        )
        if is_directory_source and not caption_extension:
            messages.append(
                f"ERROR: {label}: directory source requires caption_extension (dataset or general)."
            )

        if is_video and not ds.get("target_frames"):
            messages.append(f"ERROR: {label}: video dataset requires target_frames.")

        target_frames = ds.get("target_frames") or []
        for n in target_frames:
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                continue
            if not _n_times_4_plus_1(n_int):
                messages.append(
                    f"WARNING: {label}: target_frames value {n_int} is not N*4+1; upstream will truncate it."
                )

        if ds.get("frame_extraction") == "chunk" and 1 in [
            int(n) for n in target_frames if str(n).lstrip("-").isdigit()
        ]:
            messages.append(
                f"WARNING: {label}: target_frames contains 1 with frame_extraction='chunk'."
            )

        cache_dir = ds.get("cache_directory")
        if cache_dir:
            if cache_dir in cache_dirs_seen:
                messages.append(
                    f"WARNING: {label}: cache_directory '{cache_dir}' duplicates {cache_dirs_seen[cache_dir]}."
                )
            else:
                cache_dirs_seen[cache_dir] = label

        source_fps = ds.get("source_fps")
        if source_fps is not None and source_fps != "":
            try:
                float(source_fps)
            except (TypeError, ValueError):
                messages.append(
                    f"ERROR: {label}: source_fps '{source_fps}' is not a valid number."
                )

        for key in (
            "image_directory",
            "video_directory",
            "cache_directory",
            "control_directory",
        ):
            value = ds.get(key)
            if value and not os.path.exists(value):
                messages.append(
                    f"WARNING: {label}: {key} '{value}' does not exist on disk."
                )
        for key in ("image_jsonl_file", "video_jsonl_file"):
            value = ds.get(key)
            if value and not os.path.isfile(value):
                messages.append(
                    f"WARNING: {label}: {key} '{value}' does not exist on disk."
                )

    return messages


def parse_int_pair(text: str):
    """Tolerant parser for '960,544' / '[960, 544]' style pairs. Raises ValueError."""
    values = _parse_number_list(text)
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got: {text!r}")
    return [int(v) for v in values]


def parse_int_list(text: str):
    """Tolerant parser for '1,25,45' / '[1, 25, 45]' style lists. Raises ValueError."""
    values = _parse_number_list(text)
    if not values:
        raise ValueError(f"Expected at least one comma-separated value, got: {text!r}")
    return [int(v) for v in values]


def _parse_number_list(text: str):
    if text is None:
        raise ValueError("Expected a value, got None")
    stripped = text.strip().strip("[]").strip()
    if stripped == "":
        raise ValueError(f"Expected comma-separated numbers, got: {text!r}")
    parts = [p.strip() for p in stripped.split(",")]
    values = []
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            raise ValueError(f"Expected an integer, got: {p!r} (from {text!r})")
        values.append(p)
    return values
