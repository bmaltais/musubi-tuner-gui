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
| `common_gui.py` | Shared Gradio helpers and path utilities |
| `custom_logging.py` | Logger setup |

## Work Guidance

- Prefer extending an existing `class_*.py` panel over bloating `lora_gui.py` further
- When adding a training flag: UI control → `gui_actions` / `train_model` parameter path → CLI arg mapping; keep names aligned with musubi-tuner CLI where possible
- Preserve headless behavior for buttons that hide file dialogs
- Do not hardcode machine-local model paths in Python; use config defaults or empty strings
- `PYTHONPATH` setup in `setup_environment` still references `sd-scripts`; treat as legacy from Kohya GUI lineage—verify before relying on it for musubi-tuner

## Verification

- No automated unit tests for this package yet
- Manual: `uv run gui.py` from repo root; exercise Open/Save config and dry-run / print command paths
- Manual backend smoke steps documented in `../test/test.MD` (requires checked-out submodule and model weights)

## Child DOX Index

None. Flat package; do not nest AGENTS.md per class module unless a subdirectory is introduced.
