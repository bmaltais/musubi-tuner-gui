# musubi-tuner-gui

GUI for the new musubi-tuner.

Contributions to the GUI code are welcome. This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager to facilitate cross-platform use. The aim is to support Linux and Windows, with potential MacOS support pending contributions.

Work towards a Minimum Viable Product (MVP) will be done in the `dev` branch. The `main` branch will remain empty until there is a consensus on a viable first release.

## Installation

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

```shell
git clone --recursive https://github.com/bmaltais/musubi-tuner-gui.git
cd musubi-tuner-gui
uv run gui.py
```

## Caching generation

Until the GUI automastically ng, you have to do it manuallUse the followingtwo commands you ache images and txt prompts:e

```shell
uv run ./musubi-tuner/cache_latents.py --dataset_config "./test/config/dataset.toml" --vae "C:\Users\berna\Downloads\pytorch_model.pt" --vae_chunk_size 32 --vae_tiling

uv run ./musubi-tuner/cache_text_encoder_outputs.py --dataset_config "./test/config/dataset.toml" --text_encoder1 "C:\Users\berna\Downloads\llava_llama3_fp16.safetensors" --text_encoder2 "C:\Users\berna\Downloads\clip_l.safetensors" --batch_size 1
```
