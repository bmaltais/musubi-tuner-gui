# Design: Architecture Registry for musubi-tuner-gui

Written during Move 7 of `wargame/2026-07-10-musubi-tuner-upstream-catchup.md`. This is the
design-only deliverable for that move; Move 8 implements it. No code changes in this document.

## Problem being solved

Two fragility points block adding architectures 2 through 12:

1. **The command-construction is hardcoded to HunyuanVideo.** `lora_gui.py:475` (`hv_train_network.py`)
   and the two cache commands at `lora_gui.py:344`/`:410` (`cache_latents.py`, `cache_text_encoder_outputs.py`)
   are literal strings. Every new architecture needs a different script triple plus a different
   set of model-path fields (Wan needs 4+ model paths and a task selector; Qwen-Image-Edit needs
   control-image fields; HiDream needs DINOv3 flags).
2. **The Gradio wiring is a flat, hand-synchronized positional pipeline.** `lora_tab()` builds
   `settings_list` (a Python list of ~140 Gradio component objects, `lora_gui.py:627-772`) and
   passes it as `inputs=[...] + settings_list` to three different `.click()` handlers
   (`lora_gui.py:783-846`). `gui_actions()` (`lora_gui.py:45-190`) receives those same values as
   ~140 *positional* function parameters and immediately does
   `parameters = [(k, v) for k, v in locals().items() ...]` — it relies on the Python parameter
   **names**, in **order**, matching `settings_list`'s **order** exactly, with no reconciliation
   check. Insert or reorder one field in `settings_list` without touching the signature (or vice
   versa) and every field after that point silently receives the wrong value. There is currently
   no test or assertion that would catch this.

The registry design below fixes the first problem structurally and the second problem by
construction (not by adding a check on top of the existing pattern, but by removing the dual
list entirely).

## Part A — the field-name/order fragility fix (applies regardless of how many architectures exist)

Replace the implicit "locals() must match settings_list order" contract with an explicit,
single source of truth.

**New: `FIELD_SPECS`, a single ordered list of `(name: str, component_getter)` pairs**, built
once in `lora_tab()` alongside the widget construction (replacing today's `settings_list`, which
becomes derived from it: `settings_list = [c for _, c in FIELD_SPECS]`).

`gui_actions()` stops taking ~140 positional parameters. Its new signature:

```python
def gui_actions(action_type, bool_value, file_path, headless, print_only, *field_values):
    parameters = list(zip(FIELD_NAMES, field_values))
    ...
```

where `FIELD_NAMES = [name for name, _ in FIELD_SPECS]` is imported/shared from the same place
`lora_tab()` builds `FIELD_SPECS`. There is now exactly **one** ordered list of names, and
`field_values` is positionally zipped against it at the call site, not reconstructed from `locals()`
in a hand-maintained function signature 140 lines long. Adding, removing, or reordering a field is
a one-place edit. `save_configuration`, `open_configuration`, and `train_model` already accept
`parameters` as a list of `(name, value)` tuples — **no change needed there**, since they already
consume the tuple-list form, not positional args.

This alone (independent of multi-architecture support) removes the single highest-risk pattern
flagged in Move 6 / the wargame plan's executor notes, and should ship even if multi-arch work
stalled here.

## Part B — the architecture registry

### `ArchitectureSpec` (new file: `musubi_tuner_gui/class_architecture.py`)

```python
@dataclass
class ArchitectureSpec:
    key: str                        # stable id stored in config TOML, e.g. "hunyuanvideo", "wan"
    label: str                      # display name in the dropdown, e.g. "HunyuanVideo", "Wan 2.1/2.2"
    train_script: str               # relative to musubi-tuner/, e.g. "hv_train_network.py"
    cache_latents_script: str       # e.g. "wan_cache_latents.py"
    cache_teo_script: str           # e.g. "wan_cache_text_encoder_outputs.py" (or cache_pixel for HiDream)
    model_field_group: str          # name of the model-fields group this arch shows (see Part C)
    unsupported_shared_args: set[str]  # shared-feature arg names (Part D) this arch's trainer does NOT accept
    extra_args: list[str]           # arch-unique arg names beyond the model-field group (e.g. Wan's --task)

REGISTRY: dict[str, ArchitectureSpec] = {
    "hunyuanvideo": ArchitectureSpec(
        key="hunyuanvideo", label="HunyuanVideo",
        train_script="hv_train_network.py",
        cache_latents_script="cache_latents.py",
        cache_teo_script="cache_text_encoder_outputs.py",
        model_field_group="hunyuanvideo",
        unsupported_shared_args=set(),   # baseline; refine per Move 11 per-arch help dumps
        extra_args=[],
    ),
    # Move 9 adds "wan", Move 10 instances add the rest, one entry each.
}
```

The registry is a plain dict, not a Gradio object — it's imported by both `lora_gui.py` (to
build the dropdown and drive visibility) and `class_model.py` (to know which model-field group
to render/show).

### Config-key prefix decision: **none — flat namespace, disambiguate by grouping instead**

Considered prefixing every architecture's fields (e.g. `wan_dit` vs `hv_dit`) to avoid TOML key
collisions. Rejected: it would mean rewriting every existing HunyuanVideo TOML key (breaking
backward compatibility, which Move 12 explicitly requires preserving) and would multiply the
number of fields threaded through `FIELD_SPECS` by ~12x even though most architectures share the
*same* conceptual fields (a DiT path, a VAE path, text encoder paths). Instead: **one field name
means one thing across all architectures that have it** (`dit`, `vae`, `text_encoder1`, etc. stay
the same Python/TOML identifiers everywhere), and architectures that need an extra field not in
the shared set (Wan's `--task`, HiDream's DINOv3 flags) add uniquely-named fields
(`wan_task`, `hidream_dinov3_*`) that are simply hidden/excluded when not applicable. This
mirrors how `SaveConfigFileToRun` already excludes fields by name pattern
(`caching_latent_*`, `caching_teo_*`) — extending that mechanism to `arch_only_*` patterns is a
small, well-precedented change, not a new concept.

### Dropdown placement and behavior

A new `gr.Dropdown` ("Architecture") is added as the **first** control inside the existing
"Model Settings" accordion (`lora_gui.py:576-577`, where `Model(...)` is constructed today) —
not a new top-level accordion, since the architecture choice belongs conceptually with "what
model am I training." Its `.change()` handler calls one function,
`apply_architecture(arch_key) -> list[gr.update]`, that returns a `gr.update(visible=...)` for
every field group (Part C). This is the **only** place visibility logic lives — both the dropdown
change event and config-load (Move 8/12) call through this one function, per the wargame plan's
Move 8 counter-move ("route both dropdown changes and config loads through one shared apply
function").

## Part C — model-field grouping (the `class_model.py` refactor)

`class_model.py` currently renders one fixed set of ~25 fields
(`dit`, `dit_dtype`, `vae`, ..., `show_timesteps` — see `class_model.py:18-215`) unconditionally.
Refactor `Model.__init__` to render each field inside a `gr.Group(visible=...)` keyed by which
architecture(s) use it, using a small number of **shared groups** rather than one group per
architecture (12 groups would mean most fields are duplicated 12 times):

- **`group_dit_vae_te`** — `dit`, `dit_dtype`, `vae`, `vae_dtype`, `vae_tiling`, `vae_chunk_size`,
  `vae_spatial_tile_sample_min_size`, `text_encoder1`, `text_encoder2`, `text_encoder_dtype`,
  `fp8_llm`, `fp8_base` — visible for every architecture that follows the DiT+VAE+text-encoder
  shape (HunyuanVideo, HunyuanVideo 1.5, Wan, FramePack, FLUX Kontext/2, Qwen-Image, Z-Image,
  Kandinsky 5, Krea 2 — confirm exact per-arch model surface from each arch's `wargame/args/*.txt`
  dump and `docs/<arch>.md` at Move 9/10 time, since names may differ, e.g. Wan uses separate CLIP
  and T5 paths instead of `text_encoder1`/`text_encoder2`).
- **`group_flow_matching`** — `timestep_sampling`, `discrete_flow_shift`, `sigmoid_scale`,
  `weighting_scheme`, `logit_mean`, `logit_std`, `mode_scale`, `min_timestep`, `max_timestep`,
  `show_timesteps` — visible for flow-matching architectures (most of them; verify per-arch).
- **`group_perf`** — `blocks_to_swap`, `img_in_txt_in_offloading`, `guidance_scale` — visible per
  arch capability (confirmed present/absent via the `wargame/args/*.txt` dumps already harvested).
- Architecture-unique groups added only when Move 9/10 hits a field with no home in the above
  (e.g. `group_wan_task` for Wan's `--task`, `group_hidream_dinov3` for HiDream's auxiliary loss).

`ArchitectureSpec.model_field_group` is replaced by a list of group names to show
(`model_field_groups: list[str]`), since most architectures need 2-3 of the shared groups plus
zero or one unique group, not a single monolithic per-arch group.

## Part D — shared new features (Move 11 features, gated per architecture)

Each shared feature (torch.compile family, `save_precision`, block-swap pinned-memory/H2D
variants, LoHa/LoKr network modules, `flash3`, new optimizers) is implemented **once** as its own
field group, always present in the DOM, with per-field `interactive`/visible state driven by
`ArchitectureSpec.unsupported_shared_args`. Do not duplicate these fields per architecture — the
wargame plan's Move 11 counter-move already establishes that presence must be checked per-arch
against the harvested `--help` dumps before wiring a shared flag globally.

## Part E — config save/load migration (feeds Move 12)

- **New key `architecture`**, written by `SaveConfigFile`/`SaveConfigFileToRun` like any other
  field, defaulting to `"hunyuanvideo"` when absent (covers every config saved before this
  refactor, since `open_configuration`'s existing fallback logic — `toml_value if toml_value is
  not None else value` at `lora_gui.py:327` pre-refactor — already returns the field's Python
  default when a TOML key is missing; the dropdown's default value is set to `"hunyuanvideo"`).
- On config load, `open_configuration` calls `apply_architecture(loaded_arch_key)` (Part B) so the
  visible field set matches the loaded architecture, not whatever the dropdown happened to show
  before the file was opened.

## Test-on-paper check (per the wargame plan's Move 7 acceptance criterion)

*"Adding architecture #13 touches only the registry file plus at most one arch-specific class
file."* Walk-through for a hypothetical 13th architecture that reuses all existing model-field
groups and all existing shared features:

1. Add one `ArchitectureSpec` entry to `REGISTRY` in `class_architecture.py`. **(1 file)**
2. If it needs zero new fields (reuses existing groups exactly): done. **(1 file total)**
3. If it needs one arch-unique field (like Wan's `--task`): add that field + its `gr.Group` to
   `class_model.py`, add its name to `FIELD_SPECS` in `lora_gui.py`. **(2 files)**
4. If it needs a new caching script pattern not matching the existing
   `cache_latents_script`/`cache_teo_script` shape (e.g. HiDream's `cache_pixel` naming): the
   `ArchitectureSpec` fields already model this as arbitrary script-name strings — no code change
   needed, only data. **(0 extra files)**

Worst case is 2 files touched (registry + one field addition), well inside the ≤5-file phase
budget the user's global CLAUDE.md invariant requires, and the passing case (reusing existing
groups) touches exactly 1. **Design passes the test.**

## What Move 8 implements from this design

1. `musubi_tuner_gui/class_architecture.py` — `ArchitectureSpec` dataclass + `REGISTRY` dict
   (HunyuanVideo entry only).
2. `class_model.py` — wrap existing fields in the `group_dit_vae_te` / `group_flow_matching` /
   `group_perf` `gr.Group`s (all `visible=True` initially, since only HunyuanVideo exists yet).
3. `lora_gui.py` — add the architecture `gr.Dropdown`; introduce `FIELD_SPECS`/`FIELD_NAMES` and
   rewrite `gui_actions()`'s signature per Part A; add `apply_architecture()`; wire script-name
   lookups in `train_model()` (currently hardcoded at lines 344, 410, 475) to read from
   `REGISTRY[current_arch].train_script` etc. instead of literal strings.
4. Verification (per the wargame plan's Move 8 acceptance criterion): the print-only command for
   HunyuanVideo must be textually identical (modulo whitespace/ordering) to the pre-refactor
   baseline captured in Move 5, and the live `hv_train_network.py --config_file ...` smoke test
   (already used in Moves 5 and 6) must still reach the same `FileNotFoundError` on the fake DiT
   path.
