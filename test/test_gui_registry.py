"""Automated tests for the architecture registry and the Gradio wiring it drives.

These guard the two things Move 6/7/8 of the upstream-catchup wargame plan flagged
as the highest-risk parts of adding new architectures:
  1. FIELD_NAMES and settings_list silently drifting out of order/length.
  2. A registered architecture pointing at a script that doesn't exist, or an
     apply_architecture() mapping that doesn't match its declared field groups.

Run with: uv run pytest test/test_gui_registry.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.class_architecture import (
    REGISTRY,
    get_architecture,
    architecture_choices,
)
from musubi_tuner_gui.lora_gui import FIELD_NAMES, apply_architecture, gui_actions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSUBI_TUNER_DIR = os.path.join(REPO_ROOT, "musubi-tuner")

ALL_GROUP_NAMES = {
    "dit_vae",
    "hv_extras",
    "wan_extras",
    "qwen_image_extras",
    "perf",
    "flow_matching",
}


def test_field_names_has_no_duplicates():
    assert len(FIELD_NAMES) == len(set(FIELD_NAMES))


def test_field_names_is_non_empty():
    assert len(FIELD_NAMES) > 0


@pytest.mark.parametrize("key,spec", REGISTRY.items())
def test_registry_scripts_exist_on_disk(key, spec):
    for script_attr in ("train_script", "cache_latents_script", "cache_teo_script"):
        script_name = getattr(spec, script_attr)
        script_path = os.path.join(MUSUBI_TUNER_DIR, script_name)
        assert os.path.isfile(
            script_path
        ), f"{key}.{script_attr} = {script_name!r} does not exist at {script_path}"


@pytest.mark.parametrize("key,spec", REGISTRY.items())
def test_registry_model_field_groups_are_known(key, spec):
    unknown = set(spec.model_field_groups) - ALL_GROUP_NAMES
    assert not unknown, f"{key} references unknown field group(s): {unknown}"


@pytest.mark.parametrize("key,spec", REGISTRY.items())
def test_registry_extra_args_are_in_field_names(key, spec):
    missing = [name for name in spec.extra_args if name not in FIELD_NAMES]
    assert (
        not missing
    ), f"{key}.extra_args references field(s) not in FIELD_NAMES: {missing}"


def test_get_architecture_falls_back_to_default_for_unknown_key():
    spec = get_architecture("does-not-exist")
    assert spec.key == "hunyuanvideo"


def test_architecture_choices_cover_full_registry():
    labels_and_keys = architecture_choices()
    keys = {key for _, key in labels_and_keys}
    assert keys == set(REGISTRY.keys())


@pytest.mark.parametrize("key,spec", REGISTRY.items())
def test_apply_architecture_visibility_matches_model_field_groups(key, spec):
    dit_vae, hv_extras, wan_extras, qwen_image_extras, perf, flow_matching = (
        apply_architecture(key)
    )
    expected = {
        "dit_vae": "dit_vae" in spec.model_field_groups,
        "hv_extras": "hv_extras" in spec.model_field_groups,
        "wan_extras": "wan_extras" in spec.model_field_groups,
        "qwen_image_extras": "qwen_image_extras" in spec.model_field_groups,
        "perf": "perf" in spec.model_field_groups,
        "flow_matching": "flow_matching" in spec.model_field_groups,
    }
    actual = {
        "dit_vae": dit_vae.visible,
        "hv_extras": hv_extras.visible,
        "wan_extras": wan_extras.visible,
        "qwen_image_extras": qwen_image_extras.visible,
        "perf": perf.visible,
        "flow_matching": flow_matching.visible,
    }
    assert actual == expected


def _minimal_field_values(overrides):
    numeric_defaults = {
        "network_dim": 32,
        "max_train_steps": 10,
        "main_process_port": 0,
        "num_processes": 0,
        "num_machines": 0,
        "num_cpu_threads_per_process": 0,
    }
    return [overrides.get(name, numeric_defaults.get(name, "")) for name in FIELD_NAMES]


@pytest.mark.integration
@pytest.mark.parametrize("key,spec", REGISTRY.items())
def test_gui_actions_print_only_emits_correct_train_script(key, spec, capsys):
    """End-to-end: gui_actions(print_only=True) must resolve to this architecture's
    own train_script, proving the registry lookup (not a hardcoded path) drives the
    final accelerate launch command. Runs the real caching subprocesses against the
    repo's test dataset fixture; expected to fail past argument parsing on the fake
    model paths, not to succeed.
    """
    overrides = {
        "dataset_config": "test/config/dataset.toml",
        "dit": "fake/dit.safetensors",
        "vae": "fake/vae.safetensors",
        "text_encoder1": "fake/te1.safetensors",
        "text_encoder2": "fake/te2.safetensors",
        "caching_teo_text_encoder1": "fake/te1.safetensors",
        "caching_teo_text_encoder2": "fake/te2.safetensors",
        "t5": "fake/t5.pth",
        "clip": "fake/clip.pth",
        "task": "t2v-14B",
        "text_encoder": "fake/text_encoder.safetensors",
        "model_version": "original",
        "network_module": "networks.lora",
        "output_dir": "test/output",
        "output_name": "test_lora",
        "mixed_precision": "bf16",
        "architecture": key,
        "extra_accelerate_launch_args": "",
        "additional_parameters": "",
    }
    field_values = _minimal_field_values(overrides)
    gui_actions("train_model", False, "", True, True, *field_values)
    captured = capsys.readouterr()
    assert spec.train_script in captured.out
