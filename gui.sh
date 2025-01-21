#!/usr/bin/env bash

export PATH=./uv/linux:$PATH

uv run gui.py "$@"