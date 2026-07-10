# test

## Purpose

Manual integration fixtures for musubi-tuner caching and training: sample image/caption dataset, dataset TOML, precomputed caches, and documented smoke commands.

## Ownership

- Owns: fixtures under `config/`, `dataset/`, `cache_directory/`, and `test.MD`
- Does not own: GUI code, production config (`../config.toml`), training outputs (`output/` is gitignored)

## Local Contracts

- `config/dataset.toml` points at `./test/dataset/darius kawasaki` and `./test/cache_directory` (paths relative to repo root when commands run from root)
- `config/dataset.toml` is also a round-trip fixture for `test_dataset_toml.py` (Dataset Config tab): don't casually reformat it — tests assert its loaded values (`resolution = [960, 544]`, `caption_extension = ".txt"`, batch_size 1, one dataset with `image_directory`/`cache_directory`) survive an open→save cycle
- Dataset pairs are image + matching `.txt` caption (`caption_extension = ".txt"`)
- `cache_directory/` may hold large `.safetensors` cache tensors; treat as binary fixtures, not source
- `test/output` is gitignored; created by training runs
- `test.MD` documents manual CLI smoke tests (cache latents, cache TE outputs, accelerate train)—not a pytest suite
- `test_dataset_toml.py` is an automated pytest suite for `musubi_tuner_gui/dataset_config_toml.py` (round-trip, unknown-key preservation, validation rules, parse helpers); run via `uv run pytest test/test_dataset_toml.py -v`

## Work Guidance

- Keep dataset paths in `dataset.toml` portable (relative paths under `test/`)
- Do not commit personal model weight paths into fixtures; pass weights on the command line or local config
- Prefer small sample sets for smoke tests; avoid bloating the repo with extra binary caches unless required for a documented scenario
- When GUI default paths reference this tree, keep them consistent with `config/dataset.toml`

## Verification

- Manual only (see `test.MD`):
  - `uv run ./musubi-tuner/cache_latents.py --dataset_config "./test/config/dataset.toml" ...`
  - `uv run ./musubi-tuner/cache_text_encoder_outputs.py ...`
  - `uv run accelerate launch ... ./musubi-tuner/hv_train_network.py ...`
- Requires: initialized `musubi-tuner` submodule, local VAE/DiT/text-encoder weights

## Child DOX Index

None. Subfolders are data only (`config/`, `dataset/`, `cache_directory/`).
