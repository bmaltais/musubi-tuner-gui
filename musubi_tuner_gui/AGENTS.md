# musubi_tuner_gui

## Purpose

Python package that implements the Gradio UI for configuring and launching musubi-tuner LoRA training workflows (latent cache, text-encoder cache, accelerate train).

## Ownership

- Owns: all modules under this package (`class_*.py`, `lora_gui.py`, `common_gui.py`, `custom_logging.py`)
- Does not own: Gradio app entrypoint (`../gui.py`), CSS/JS assets (`../assets/`), training backend (`../musubi-tuner/`), default config file (`../config.toml`)

## Local Contracts

- Package import root is `musubi_tuner_gui`; entry from repo root via `gui.py` → `lora_gui.lora_tab`
- UI panels are one class per concern; `lora_gui` composes them and builds CLI invocations
- Training pipeline order: cache latents → cache text-encoder outputs → `accelerate launch` + `hv_train_network.py`
- Commands run via `uv run` and `CommandExecutor` / `subprocess` with `setup_environment()`
- Backend scripts are expected at `./musubi-tuner/` relative to repo root (submodule path)
- `GUIConfig` loads TOML defaults; missing config file is empty dict (no hard fail)
- `common_gui` holds shared path pickers, config save helpers, and validation utilities
- Logging goes through `custom_logging.setup_logging` (file: `musubi_tuner_gui.log` at repo root, gitignored)
- Dataset TOML schema follows `musubi-tuner/docs/dataset_config.md`; `dataset_config_toml.py`/`dataset_config_gui.py` model the documented fields only, and unknown/advanced keys (`fp_1f_*`, `multiple_target`, `qwen_image_edit_*`, etc.) round-trip untouched on open→edit→save
- `dataset_config_toml.py` is UI-free by design (no Gradio import) so it stays unit-testable; never import `lora_gui` from it or from `dataset_config_gui.py`

### Module map

| Module | Role |
|--------|------|
| `lora_gui.py` | Main tab, action wiring, train/cache command assembly |
| `class_gui_config.py` | Load/save/get TOML GUI defaults |
| `class_command_executor.py` | Start/stop training process (psutil) |
| `class_accelerate_launch.py` | Accelerate multi-GPU / dynamo launch options |
| `class_advanced_training.py` | Attention, DDP, sampling, logging options |
| `class_configuration_file.py` | Open/save GUI configuration files |
| `class_latent_caching.py` | Latent cache UI |
| `class_text_encoder_outputs_caching.py` | Text-encoder cache UI |
| `class_model.py` | DiT / VAE model paths and dtypes |
| `class_network.py` | LoRA network dim/alpha/module |
| `class_optimizer_and_scheduler.py` | Optimizer and LR scheduler |
| `class_training.py` | Epochs, steps, seed, dataloader |
| `class_save_load.py` | Output dir, save intervals, resume |
| `class_huggingface.py` | HF upload settings |
| `class_metadata.py` | LoRA metadata fields |
| `settings_gui.py` | Settings tab (GUI-wide preferences, e.g. info-tooltip toggle) persisted to `config.toml`'s `[settings]` table |
| `dataset_config_toml.py` | UI-free TOML load/save/validate/parse logic for musubi-tuner dataset config files |
| `dataset_config_gui.py` | Dataset Config tab: master/detail dataset editor, Open/Save round-trip, validation panel |
| `common_gui.py` | Shared Gradio helpers and path utilities |
| `custom_logging.py` | Logger setup |

## Work Guidance

- Prefer extending an existing `class_*.py` panel over bloating `lora_gui.py` further
- When adding a training flag: UI control → `gui_actions` / `train_model` parameter path → CLI arg mapping; keep names aligned with musubi-tuner CLI where possible
- Preserve headless behavior for buttons that hide file dialogs
- Do not hardcode machine-local model paths in Python; use config defaults or empty strings
- `PYTHONPATH` setup in `setup_environment` still references `sd-scripts`; treat as legacy from Kohya GUI lineage—verify before relying on it for musubi-tuner

## Verification

- Automated: `uv run pytest test/test_dataset_toml.py -v` covers `dataset_config_toml.py` (round-trip, unknown-key preservation, validation rules, parsing helpers). No automated tests yet for the rest of this package.
- Manual: `uv run gui.py` from repo root; exercise Open/Save config and dry-run / print command paths, and the Dataset Config tab's Open/Save/Add/Remove/Apply flow
- Manual backend smoke steps documented in `../test/test.MD` (requires checked-out submodule and model weights)

## Child DOX Index

None. Flat package; do not nest AGENTS.md per class module unless a subdirectory is introduced.
