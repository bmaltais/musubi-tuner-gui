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
