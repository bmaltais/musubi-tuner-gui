# musubi-tuner-gui

GUI for [musubi-tuner](https://github.com/kohya-ss/musubi-tuner), tracking upstream release v0.3.4.

Contributions to the GUI code are welcome. This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager to facilitate cross-platform use. The aim is to support Linux and Windows, with potential MacOS support pending contributions.

## Supported architectures

An "Architecture" dropdown at the top of Model Settings selects between the following, each with its own training/caching scripts and model fields:

| Architecture | Notes |
|---|---|
| HunyuanVideo | dual text encoder, VAE tiling |
| Wan 2.1/2.2 | task selector, T5/CLIP text encoders, dual DiT for 2.2 |
| Qwen-Image | VL text encoder, model version (original/layered/edit/edit-2509) |
| Z-Image | single text encoder |
| FLUX.2 | single text encoder, model version |
| FLUX.1 Kontext | dual text encoder |
| HunyuanVideo 1.5 | t2v/i2v task, ByT5 text encoder, image encoder |
| FramePack | image encoder, F1/one-frame sampling modes |
| Kandinsky 5 | CLIP+Qwen dual text encoder |
| HiDream-O1-Image | DINOv3 auxiliary loss (weight only; full tuning via Additional Parameters) |
| Ideogram4 | unconditional DiT, sampler presets, caption validation |
| Krea 2 | distilled Turbo DiT for sample generation |

Shared across every architecture: LoRA training, `save_precision`, torch.compile (`compile`/`compile_backend`/`compile_mode`), and block-swap performance options (pinned memory, H2D-only, ring size).

Some architecture-specific tuning flags with many rarely-changed sub-options (Kandinsky 5's nabla-attention params, HiDream-O1's DINOv3 loss internals, etc.) are intentionally left to the **Additional Parameters** free-text field rather than getting dedicated widgets — check the [musubi-tuner docs](https://github.com/kohya-ss/musubi-tuner/tree/main/docs) for the full flag reference per architecture.

## Settings

A **Settings** tab holds GUI-wide preferences, persisted to `config.toml` under a `[settings]` table:

| Setting | Default | Effect |
|---|---|---|
| Enable info tooltips on hover | On | Shows each field's description as a floating tooltip when you hover or focus its name, instead of always-on static hint text. Toggling it applies immediately in the browser, no restart needed. |

## Documentation about musubi-tuner

Have a read of the documentation posted on https://github.com/kohya-ss/musubi-tuner for details about dataset preparation and the accompanying toml file required for training. This only provide a GUI to configure the tuner parameter. Not the dataset configuration file.

## Requirements

- Python 3.10, 3.11, or 3.12
- An NVIDIA GPU with CUDA 12.4, 12.8, or 13.0 drivers (select the matching extra below)

## Installation (optional, you can skip this section if you prefer to use the provided uv code in the repo)

The installation process will be improved and automated in the future. For now, follow these steps:

1. Install uv (if not already present on your OS).

### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

To add `C:\Users\berna\.local\bin` to your PATH, either restart your system or run:

#### CMD

```cmd
set Path=C:\Users\berna\.local\bin;%Path%
```

#### Powershell

```powershell
$env:Path = "C:\Users\berna\.local\bin;$env:Path"
```

## Starting the GUI
### With uv installation

```shell
git clone --recursive https://github.com/bmaltais/musubi-tuner-gui.git
cd musubi-tuner-gui
uv run gui.py
```

### With repo based uv version
#### Windows

```shell
git clone --recursive https://github.com/bmaltais/musubi-tuner-gui.git
cd musubi-tuner-gui
.\gui.bat
```

#### Linux

```shell
git clone --recursive https://github.com/bmaltais/musubi-tuner-gui.git
cd musubi-tuner-gui
./gui.sh
```

### Selecting a CUDA version

`pyproject.toml` pins the `musubi-tuner` dependency to the `cu128` extra by default (CUDA 12.8, matching torch ≥2.7.1). If your driver supports a different CUDA version, edit the `musubi-tuner[cu128]` line to `cu124` (CUDA 12.4, torch ≥2.5.1) or `cu130` (CUDA 13.0, torch ≥2.9.1), then re-run `uv sync`.

## Running tests

```shell
uv run pytest test/ -v
```

This runs the registry regression suite (one architecture's fields/scripts/caching commands validated per test) and the backward-compatibility suite (pre-refactor `config.toml` files still load correctly). Some tests run the real musubi-tuner caching scripts against the bundled test dataset fixture and are expected to fail past argument parsing on placeholder model paths — that's the pass condition, not a bug.
