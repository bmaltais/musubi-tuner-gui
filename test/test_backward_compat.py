"""Backward-compatibility tests for loading pre-refactor config.toml files.

wargame/legacy-config-baseline.toml is a real config.toml captured from the
`main` branch (git show main:config.toml) before the Move 8 architecture-
registry refactor. It has no "architecture" key and none of the Move 10/11
fields (task, save_precision, compile, etc.) -- exactly what any user's
saved config from before this mission looked like.

Run with: uv run pytest test/test_backward_compat.py -v
"""

import os
import sys

import toml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.class_architecture import DEFAULT_ARCHITECTURE
from musubi_tuner_gui.lora_gui import FIELD_NAMES, open_configuration

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_CONFIG_PATH = os.path.join(REPO_ROOT, "wargame", "legacy-config-baseline.toml")


def _fresh_gui_defaults():
    """Values a freshly-opened GUI would show before any config is loaded:
    every field falsy/empty except architecture, which defaults to
    DEFAULT_ARCHITECTURE -- matching how Model.__init__ actually
    constructs the architecture dropdown's default value."""
    defaults = {"architecture": DEFAULT_ARCHITECTURE}
    return [(name, defaults.get(name, "")) for name in FIELD_NAMES]


def test_legacy_config_file_exists():
    assert os.path.isfile(LEGACY_CONFIG_PATH), (
        "wargame/legacy-config-baseline.toml is missing -- it must be "
        "re-captured (git show main:config.toml) if ever removed"
    )


def test_legacy_config_loads_without_error():
    parameters = _fresh_gui_defaults()
    result = open_configuration(
        ask_for_file=False, file_path=LEGACY_CONFIG_PATH, parameters=parameters
    )
    assert result is not None
    assert result[0] == LEGACY_CONFIG_PATH


def test_legacy_config_defaults_to_hunyuanvideo_architecture():
    """The legacy config predates the "architecture" key entirely. Loading
    it must not crash and must select HunyuanVideo, since that was the
    GUI's only supported architecture at the time the file was saved."""
    parameters = _fresh_gui_defaults()
    result = open_configuration(
        ask_for_file=False, file_path=LEGACY_CONFIG_PATH, parameters=parameters
    )
    values_by_name = dict(zip(["file_path"] + FIELD_NAMES, result))
    assert values_by_name["architecture"] == DEFAULT_ARCHITECTURE == "hunyuanvideo"


def test_legacy_config_values_load_correctly():
    """Spot-check that real values from the legacy file survive the load
    unchanged -- this is what the whole test exists to protect."""
    with open(LEGACY_CONFIG_PATH, "r", encoding="utf-8") as f:
        legacy_data = toml.load(f)

    parameters = _fresh_gui_defaults()
    result = open_configuration(
        ask_for_file=False, file_path=LEGACY_CONFIG_PATH, parameters=parameters
    )
    values_by_name = dict(zip(["file_path"] + FIELD_NAMES, result))

    # Every key actually present in the legacy file must come through as-is.
    mismatches = {
        key: (legacy_data[key], values_by_name[key])
        for key in legacy_data
        if key in values_by_name and values_by_name[key] != legacy_data[key]
    }
    assert not mismatches, f"Legacy values changed on load: {mismatches}"


def test_legacy_config_missing_keys_default_safely():
    """Fields the legacy file never had (Move 10/11 additions: task,
    save_precision, compile, etc.) must fall back to the GUI's own
    defaults rather than raising or leaving None where a widget expects
    a concrete value."""
    with open(LEGACY_CONFIG_PATH, "r", encoding="utf-8") as f:
        legacy_data = toml.load(f)

    parameters = _fresh_gui_defaults()
    result = open_configuration(
        ask_for_file=False, file_path=LEGACY_CONFIG_PATH, parameters=parameters
    )
    values_by_name = dict(zip(["file_path"] + FIELD_NAMES, result))

    fields_not_in_legacy_file = [
        name for name in FIELD_NAMES if name not in legacy_data
    ]
    assert fields_not_in_legacy_file, "expected at least one Move 10/11-only field"
    for name in fields_not_in_legacy_file:
        # Falls back to the fresh-GUI default (never crashes, never KeyErrors).
        assert values_by_name[name] == dict(parameters)[name]
