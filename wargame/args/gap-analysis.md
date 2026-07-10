# Gap Analysis: musubi-tuner-gui vs musubi-tuner v0.3.4

Generated 2026-07-10 during Move 4. Source of truth: `wargame/args/*.txt` (36 `--help` dumps
captured against upstream tag `v0.3.4`, commit `30c658c`), diffed against the current GUI's
field surface in `musubi_tuner_gui/lora_gui.py` (`gui_actions()` signature, lines 45-190).

## 1. HunyuanVideo (`hv_train_network.py`) — the GUI's existing baseline

**Result: zero renames, zero removals.** Every one of the ~95 fields the current GUI emits
(`dit`, `vae`, `text_encoder1/2`, `network_dim`, `optimizer_type`, `sample_prompts`, all
`save_*`/`huggingface_*`/`metadata_*` fields, etc.) still exists verbatim in
`hv_train_network.py --help` under v0.3.4. **Move 5 (restore baseline) requires no argument
fixes** — only needs a live-run verification (script paths are unchanged; scripts are now thin
wrappers around `musubi_tuner.<name>` but keep the same top-level filename and CLI surface).

### New HunyuanVideo trainer args not yet in the GUI (candidates for Move 11 shared-features / Move 5 bonus)
- `--save_precision {float,fp32,fp16,bf16}` — **highest priority**, default changed to fp32 in upstream v0.3.2
- `--compile`, `--compile_backend`, `--compile_mode`, `--compile_dynamic`, `--compile_fullgraph`, `--compile_cache_size_limit` — torch.compile support (trainer-scope, distinct from the GUI's existing `accelerate launch --dynamo_*` fields — see naming collision note below)
- `--dynamo_backend`, `--dynamo_mode`, `--dynamo_fullgraph`, `--dynamo_dynamic` — **trainer-scope** dynamo flags, separate from the GUI's `AccelerateLaunch`-scope fields of the same name. No runtime collision today (GUI never threads its accelerate-launch dynamo_* into the trainer TOML — confirmed via `SaveConfigFileToRun`'s exclusion list) but registry field names must be disambiguated (e.g. `compile_dynamo_backend`) to avoid a Python-level positional-arg collision in `gui_actions()`.
- `--cuda_allow_tf32`, `--cuda_cudnn_benchmark` — CUDA perf toggles
- `--block_swap_h2d_only`, `--block_swap_ring_size`, `--use_pinned_memory_for_block_swap` — block-swap perf features (June 2026 release notes)
- `--gradient_checkpointing_cpu_offload` — VRAM reduction (v0.2.12)
- `--flash3` — new attention backend alongside existing sdpa/flash_attn/sage_attn/xformers/split_attn
- `--disable_numpy_memmap`
- `--dit_in_channels`
- `--text_encoder_dtype`
- `--metadata_arch`, `--metadata_reso` — explicit metadata (v0.2.14)
- `--num_timestep_buckets`
- `--preserve_distribution_shape`

## 2. RECON outcomes confirmed

- **RECON #1 (competing GUI)** — CONFIRMED present but narrow: upstream ships `src/musubi_tuner/gui/gui.py` covering only Z-Image and Qwen-Image. User decision: continue standalone (recorded 2026-07-10).
- **RECON #2 (script layout)** — top-level scripts (`hv_train_network.py`, `wan_train_network.py`, etc.) are unchanged filenames, now thin wrappers (`from musubi_tuner.<name> import main`) around the installable `src/musubi_tuner/` package. **No invocation changes needed** — `uv run python musubi-tuner/hv_train_network.py ...` still works.
- **RECON #3 (Python/torch pin)** — upstream requires `>=3.10,<3.13`; local env is 3.11.9, resolved cleanly with `musubi-tuner[cu128]` extra → torch 2.11.0+cu128, CUDA available. Root `pyproject.toml` updated and committed (Move 3).
- **RECON #6 (sampling)** — CONFIRMED: `--sample_prompts`, `--sample_at_first`, `--sample_every_n_epochs`, `--sample_every_n_steps` already existed pre-mission and are already wired in the current GUI. No new work needed here.
- **RECON #7 (Gradio compat)** — root `pyproject.toml` already pinned `gradio>=5.0.0`; `uv sync` resolved `gradio==6.17.3`. Launch verification deferred to Move 5.

## 3. All 36 harvested scripts succeeded (no dep-blocked architectures)

Training scripts (12): `hv_train_network`, `hv_1_5_train_network`, `wan_train_network`,
`fpack_train_network`, `flux_kontext_train_network`, `flux_2_train_network`,
`qwen_image_train_network`, `zimage_train_network`, `hidream_o1_train_network`,
`kandinsky5_train_network`, `ideogram4_train_network`, `krea2_train_network`.

Caching scripts (24, 2 per architecture + the 2 legacy generic ones for HunyuanVideo):
`cache_latents`, `cache_text_encoder_outputs`, plus per-arch `*_cache_latents` /
`*_cache_text_encoder_outputs` (or `hidream_o1_cache_pixel` for HiDream).

`hidream_o1_train_network.py --help` printed a "Failed to import sageattention" warning but
exited 0 — non-fatal, sageattention is an optional attention backend, not a hard dependency.

## 4. Cross-architecture shared-arg summary (for Move 11 design)

Present across most/all trainer `--help` dumps (verify per-arch before wiring globally, per the
wargame plan's Move 11 counter-move): `--sdpa/--flash_attn/--sage_attn/--xformers/--flash3/--split_attn`,
`--compile*` family, `--blocks_to_swap` + block-swap variants, `--save_precision`,
`--network_module/--network_args/--network_dim/--network_alpha/--network_dropout` (LoRA today;
LoHa/LoKr module names to confirm per-arch), `--optimizer_type/--optimizer_args`,
`--lr_scheduler*` family (incl. REX, Adafactor per release notes — confirm exact `--optimizer_type`
choices per script), sampling (`--sample_*`), metadata (`--metadata_*`), HuggingFace upload
(`--huggingface_*`), DDP (`--ddp_*`).

Architecture-unique surface observed at a glance (full detail in the per-script `.txt` dumps):
Wan has task/model-variant selectors; FLUX Kontext/Edit-style archs and Qwen-Image-Edit have
control-image concepts; HiDream-O1 has DINOv3 auxiliary-loss flags. Full per-arch field
extraction is deferred to each architecture's own Move 10 instance — this document only
establishes the shared baseline.
