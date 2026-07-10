"""Headless tests for the dataset-config pure logic module (no Gradio).

Run with: uv run pytest test/test_dataset_toml.py -v
"""

import os
import sys

import pytest
import toml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.dataset_config_toml import (
    detect_caption_extension,
    load_dataset_config,
    parse_int_list,
    parse_int_pair,
    save_dataset_config,
    validate_dataset_config,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "test", "config", "dataset.toml")


def test_fixture_round_trip(tmp_path):
    data = load_dataset_config(FIXTURE)
    out_path = tmp_path / "roundtrip.toml"
    save_dataset_config(str(out_path), data["general"], data["datasets"])
    reloaded = load_dataset_config(str(out_path))

    assert reloaded["general"]["resolution"] == [960, 544]
    assert reloaded["general"]["caption_extension"] == ".txt"
    assert reloaded["general"]["batch_size"] == 1
    assert reloaded["general"]["enable_bucket"] is True
    assert reloaded["general"]["bucket_no_upscale"] is False

    assert len(reloaded["datasets"]) == 1
    ds = reloaded["datasets"][0]
    assert ds["image_directory"] == data["datasets"][0]["image_directory"]
    assert ds["cache_directory"] == data["datasets"][0]["cache_directory"]
    assert ds["num_repeats"] == 1


def test_unknown_key_preservation(tmp_path):
    src = tmp_path / "advanced.toml"
    src.write_text(
        """
[general]
resolution = [960, 544]
caption_extension = ".txt"

[[datasets]]
image_directory = "./images"
cache_directory = "./cache"
fp_1f_clean_indices = [0]
fp_1f_target_index = 9
multiple_target = true
""",
        encoding="utf-8",
    )

    data = load_dataset_config(str(src))
    ds = data["datasets"][0]
    ds["num_repeats"] = 5  # simulate editing a known field via the UI

    out_path = tmp_path / "advanced_out.toml"
    save_dataset_config(str(out_path), data["general"], data["datasets"])

    reloaded = toml.load(str(out_path))
    saved_ds = reloaded["datasets"][0]
    assert saved_ds["fp_1f_clean_indices"] == [0]
    assert saved_ds["fp_1f_target_index"] == 9
    assert saved_ds["multiple_target"] is True
    assert saved_ds["num_repeats"] == 5


def test_cleared_known_key_deletion(tmp_path):
    data = load_dataset_config(FIXTURE)
    ds = dict(data["datasets"][0])
    ds["cache_directory"] = ""  # cleared by the UI

    out_path = tmp_path / "cleared.toml"
    save_dataset_config(str(out_path), data["general"], [ds])

    reloaded = toml.load(str(out_path))
    assert "cache_directory" not in reloaded["datasets"][0]


@pytest.mark.parametrize(
    "general,datasets,expect_error_substring",
    [
        ({}, [{}], "no source path set"),
        (
            {},
            [
                {
                    "image_directory": "./a",
                    "video_directory": "./b",
                    "cache_directory": "./c",
                    "caption_extension": ".txt",
                }
            ],
            "more than one source type",
        ),
        (
            {},
            [{"image_jsonl_file": "./a.jsonl"}],
            "jsonl source requires cache_directory",
        ),
        (
            {},
            [{"image_directory": "./a", "cache_directory": "./c"}],
            "requires caption_extension",
        ),
        (
            {},
            [
                {
                    "video_directory": "./v",
                    "cache_directory": "./c",
                    "caption_extension": ".txt",
                }
            ],
            "requires target_frames",
        ),
        (
            {},
            [
                {
                    "image_directory": "./a",
                    "cache_directory": "./c",
                    "caption_extension": ".txt",
                    "source_fps": "abc",
                }
            ],
            "not a valid number",
        ),
    ],
)
def test_validation_errors_fire(general, datasets, expect_error_substring):
    messages = validate_dataset_config(general, datasets)
    assert any(
        expect_error_substring in m and m.startswith("ERROR:") for m in messages
    ), messages


def test_validation_warnings_fire():
    datasets = [
        {
            "video_directory": "./v",
            "cache_directory": "./c",
            "caption_extension": ".txt",
            "target_frames": [1, 25, 45],
            "frame_extraction": "chunk",
        },
        {
            "video_directory": "./v2",
            "cache_directory": "./c",  # duplicate cache dir
            "caption_extension": ".txt",
            "target_frames": [1, 25, 45],
            "frame_extraction": "chunk",
        },
    ]
    messages = validate_dataset_config({}, datasets)
    assert any("1 in target_frames" in m or "contains 1 with" in m for m in messages)
    assert any("duplicates" in m for m in messages)


def test_detect_caption_extension_on_real_fixture_dir():
    directory = os.path.join(REPO_ROOT, "test", "dataset", "darius kawasaki")
    assert detect_caption_extension(directory) == ".txt"


def test_detect_caption_extension_ignores_unrelated_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "a.txt").write_text("a caption", encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"")
    (tmp_path / "b.txt").write_text("b caption", encoding="utf-8")
    (tmp_path / "readme.md").write_text("not a caption", encoding="utf-8")
    assert detect_caption_extension(str(tmp_path)) == ".txt"


def test_detect_caption_extension_majority_wins(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "a.caption").write_text("a", encoding="utf-8")
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "b.caption").write_text("b", encoding="utf-8")
    (tmp_path / "c.mp4").write_bytes(b"")
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")
    assert detect_caption_extension(str(tmp_path)) == ".caption"


def test_detect_caption_extension_returns_none_when_no_captions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.jpg").write_bytes(b"")
    assert detect_caption_extension(str(tmp_path)) is None


def test_detect_caption_extension_returns_none_for_missing_dir():
    assert detect_caption_extension("./this/does/not/exist") is None


def test_validation_silent_on_good_fixture():
    data = load_dataset_config(FIXTURE)
    messages = validate_dataset_config(data["general"], data["datasets"])
    errors = [m for m in messages if m.startswith("ERROR:")]
    assert errors == []


def test_source_fps_written_as_float(tmp_path):
    general = {}
    datasets = [
        {
            "video_directory": "./v",
            "cache_directory": "./c",
            "caption_extension": ".txt",
            "target_frames": [25],
            "source_fps": 30,
        }
    ]
    out_path = tmp_path / "fps.toml"
    save_dataset_config(str(out_path), general, datasets)
    text = out_path.read_text(encoding="utf-8")
    assert "source_fps = 30.0" in text


@pytest.mark.parametrize(
    "text,expected", [("960,544", [960, 544]), ("[960, 544]", [960, 544])]
)
def test_parse_int_pair_accepts(text, expected):
    assert parse_int_pair(text) == expected


@pytest.mark.parametrize("text", ["960", "a,b"])
def test_parse_int_pair_rejects(text):
    with pytest.raises(ValueError):
        parse_int_pair(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,25,45", [1, 25, 45]),
        ("[1, 25, 45]", [1, 25, 45]),
        (" 1, 25 , 45 ", [1, 25, 45]),
    ],
)
def test_parse_int_list_accepts(text, expected):
    assert parse_int_list(text) == expected


@pytest.mark.parametrize("text", ["a,b", ""])
def test_parse_int_list_rejects(text):
    with pytest.raises(ValueError):
        parse_int_list(text)
