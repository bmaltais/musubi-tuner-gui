# Wargame: Musubi-Tuner-GUI Upstream Catch-Up (Jan 2025 → June 2026)

> Generated 2026-07-10 · branch `main` · HEAD `ccf4d0b` · musubi-tuner submodule pinned at `dd5e030` (uninitialized locally)

## Mission Summary

The GUI repo has been dormant since 2025-01-20. It is a Gradio frontend hardcoded to a single upstream script pair (`hv_train_network.py` for HunyuanVideo training, `cache_latents.py` / `cache_text_encoder_outputs.py` for caching) against a musubi-tuner commit from January 2025. Upstream (kohya-ss/musubi-tuner) has since shipped ~18 releases (latest v0.3.4, June 2026) and now supports 12 architectures — HunyuanVideo, HunyuanVideo 1.5, Wan 2.1/2.2, FramePack, FLUX.1 Kontext, FLUX.2, Qwen-Image (+Edit/+Layered), Z-Image, HiDream-O1-Image, Kandinsky 5, Ideogram4, Krea 2 — plus a pip-installable package layout, LoHa/LoKr network modules, torch.compile, FP8 block-wise scaling, improved block swap (pinned memory, H2D-only), `--save_precision`, new optimizers/schedulers (Adafactor, REX), and a large v0.3.0 internal refactor ("NetworkTrainer extension architecture"). The mission: update the submodule, restore the existing HunyuanVideo path as a working baseline, refactor the GUI into an architecture-registry design, then add each new architecture and each new shared feature in small verified phases.

**Hard project constraints (from user's global directives):** use `uv` exclusively (never pip); ≤5 files changed per phase, verify and get explicit approval between phases; run `black` before every commit; re-read files before editing; commit dead-code removal separately before any >300 LOC refactor.

---

## Recon Requirements

### Materials already read at planning time
- `pyproject.toml` — Python pinned `>=3.10,<3.11`; deps: `musubi-tuner` (local path source), `gradio>=5.0.0`.
- `musubi_tuner_gui/lora_gui.py` (801 LOC) — single-architecture flow; command built at lines 340–531: `uv run accelerate launch … {scriptdir}/musubi-tuner/hv_train_network.py`, caching via `uv run ./musubi-tuner/cache_latents.py`. `gui_actions()` takes every widget value as a flat positional arg list (~100+ params) — fragile, central to the refactor.
- `musubi_tuner_gui/class_*.py` — one class per accordion section (model, network, optimizer, training, latent caching, TE caching, accelerate, save/load, HF, metadata). All HunyuanVideo-shaped.
- `README.md` — clone `--recursive`, launch via `uv run gui.py` / `gui.bat` / `gui.sh`.
- Upstream README (fetched 2026-07-10) — 12 architectures with per-arch `*_train_network.py` + per-arch caching scripts; Python 3.10+; PyTorch ≥2.5.1.
- Upstream releases (fetched 2026-07-10) — v0.2.11 → v0.3.4 feature list summarized in Mission Summary above. NOTE: a release note for v0.2.15 mentions an "experimental GUI" **in upstream itself** — see RECON NEEDED #1.

### Materials the executor must read before/while executing
- `.gitmodules` and the submodule URL (confirm it still points at kohya-ss/musubi-tuner).
- After submodule update: upstream `pyproject.toml`, `README.md`, and every file under `musubi-tuner/docs/` (per-architecture docs: dataset config requirements, model file requirements, recommended settings).
- Upstream `CHANGELOG` / release notes for arg renames or removals affecting existing GUI fields.
- `test/` and `test/config/dataset.toml` — the existing test fixture; reuse for smoke tests.
- `config.toml` at repo root — the GUI's default/saved config format that must stay loadable.

### RECON NEEDED (unresolved — each with the exact check that settles it)
1. **Upstream now ships its own experimental GUI — does it make this project redundant or a merge target?** — Check: after submodule update, `ls musubi-tuner/` for a gui/ or webui/ directory and grep upstream README for "GUI". If upstream's GUI is substantial (multi-arch, maintained), STOP and surface to the user: continue standalone vs. adopt/wrap upstream's — this is a strategy fork only the user can decide.
2. **Exact new upstream layout and invocation style** — Check: `ls musubi-tuner/src/musubi_tuner/` and read `[project.scripts]` in `musubi-tuner/pyproject.toml`. Settles whether GUI commands become `uv run --directory . accelerate launch src/musubi_tuner/<arch>_train_network.py` or entry-point form (`uv run wan_train_network`). All command-construction code depends on this answer.
3. **Does the current `requires-python >=3.10,<3.11` pin still satisfy upstream?** — Check: `requires-python` in upstream `pyproject.toml` and whether `uv sync` resolves with torch ≥2.5.1 on Windows. Settles whether `pyproject.toml`, `gui.bat`, `gui.sh`, and `uv/` pinned binaries need a Python bump.
4. **Which existing HunyuanVideo args were renamed/removed by the v0.3.0 refactor?** — Check: `uv run python musubi-tuner/src/musubi_tuner/hv_train_network.py --help` (adjust path per #2) diffed against the arg names currently emitted by `lora_gui.py`. Every mismatch is a GUI field to fix in Phase 1.
5. **Full per-architecture argument sets** — Check: run `--help` on all 12 training scripts and all caching scripts; save each output to `wargame/args/<script>.txt`. This is the source of truth for per-arch GUI fields; docs may lag.
6. **Sample/preview generation during training** — Check: `--help` output for `--sample_prompts` / `--sample_every_n_*` on the training scripts. If present, it's a new GUI section (the GUI has none today).
7. **Is `gradio>=5.0.0` still the right pin, and does the 18-month-old GUI code run on current Gradio?** — Check: `uv sync && uv run gui.py` and exercise one accordion. Gradio 5.x minor releases have broken component kwargs before; any traceback at launch settles it.

---

## Execution Plan

**Phasing note:** Moves 1–6 are the foundation (submodule + baseline restore). Moves 7–9 are the multi-architecture refactor. Move 10 is a repeatable per-architecture template executed once per new architecture, in the priority order given. Moves 11–13 are shared features, compat, and docs. Respect the ≤5-files-per-phase invariant: each Move below is one phase or less; Move 10 instances are one phase each. Await user approval between phases.

### Move 1: Snapshot and safety branch
**Action:** Create branch `feat/upstream-catchup` from `main`. Record current submodule pointer: `git submodule status > wargame/baseline-submodule.txt`. Do not touch `main`.
**Expected Observation:** `git branch --show-current` prints `feat/upstream-catchup`; baseline file contains `dd5e030…`.
**Most Likely Failure:** Untracked `AGENTS.md` files at root/subdirs cause confusion about what belongs in the commit. **Counter-Move:** Leave them untracked; never `git add -A`, always add explicit paths.

### Move 2: Initialize and update the submodule to the latest release tag
**Action:** `git submodule update --init`, then in `musubi-tuner/`: `git fetch --tags origin` and `git checkout <latest release tag>` (v0.3.4 at planning time — check `git tag --sort=-creatordate | head -5` for anything newer). Commit the submodule pointer bump by itself.
**Expected Observation:** `git -C musubi-tuner log -1` shows a June-2026-or-later commit; `ls musubi-tuner/` shows the new package layout (a `src/musubi_tuner/` directory is expected — RECON #2 settles specifics).
**Most Likely Failure:** Submodule URL stale or shallow clone lacks tags. **Counter-Move:** Check `.gitmodules`; if URL moved, `git submodule set-url musubi-tuner <new-url> && git submodule sync`. If tags missing, `git fetch --unshallow --tags`.
**Fork:** If you observe a `gui/` or web-UI directory in upstream (RECON #1) → STOP, report to user with a one-paragraph comparison, await decision.

### Move 3: Resolve the environment (RECON #3, #7)
**Action:** Read upstream `pyproject.toml`. Update root `pyproject.toml`: relax/bump `requires-python` to match upstream, keep `musubi-tuner = { path = "musubi-tuner" }` source if upstream is now a proper package (RECON #2), otherwise adapt. Run `uv sync`. Then `uv run python -c "import musubi_tuner"` (or equivalent import check per actual package name).
**Expected Observation:** `uv sync` resolves without error; the import check exits 0; `uv run python -c "import torch; print(torch.__version__)"` prints ≥2.5.1.
**Most Likely Failure:** Torch/CUDA wheel resolution on Windows fails or pulls CPU-only torch. **Counter-Move:** Add the explicit PyTorch index config to `pyproject.toml` (`[tool.uv.sources]`/`[[tool.uv.index]]` per upstream's documented install) matching the user's CUDA version; verify with `torch.cuda.is_available()` — but do NOT abort on `False` if no local GPU check is expected; the executor should report GPU availability rather than assume.
**Fork:** If Python must move past 3.10 and the repo's vendored `uv/` launcher or `gui.bat`/`gui.sh` hardcode a version, include those (≤5 files) in this phase.

### Move 4: Harvest ground truth for all scripts (RECON #4, #5, #6)
**Action:** For every `*_train_network.py` and every caching script upstream ships, run `uv run python <script> --help` and save output to `wargame/args/<script-name>.txt`. Build `wargame/args/gap-analysis.md`: (a) HunyuanVideo args currently emitted by the GUI that no longer exist or were renamed; (b) new shared args present across most trainers (network module choices, compile, fp8 variants, block swap, save_precision, sampling); (c) per-arch unique args (model paths, task flags like Wan's `--task`, control-image options for Edit/Kontext models).
**Expected Observation:** ≥12 training-script help dumps plus caching dumps; gap-analysis lists concrete arg names in all three buckets. This is read-only; no GUI files change.
**Most Likely Failure:** Some scripts crash on `--help` due to optional deps (e.g., arch-specific libs not in base install). **Counter-Move:** Note the missing dep in gap-analysis, install the documented extra via `uv add`/extras if upstream defines one, and re-run; if an architecture genuinely needs deps that won't resolve on Windows, mark that arch "deferred — dep blocked" rather than aborting the mission.

### Move 5: Restore the HunyuanVideo baseline
**Action:** Fix `lora_gui.py` (and only the classes it forces, ≤5 files) so the existing HunyuanVideo path works against the new submodule: correct script paths to the new layout, apply arg renames/removals from gap-analysis bucket (a). Do NOT add features in this move.
**Expected Observation:** `uv run gui.py` launches; configure a training run with `test/config/dataset.toml`, use the GUI's "print command" path (`print_only`) — the emitted command's script path exists on disk, and running the emitted command with `--help` appended exits 0. Caching commands likewise.
**Most Likely Failure:** Silent arg drift — command runs `--help` fine but a renamed arg would be rejected at real parse time. **Counter-Move:** Instead of `--help`, validate by launching the emitted training command against a deliberately empty/missing dataset and confirm it fails **past argparse** (an error about data/model files, not "unrecognized arguments").
**Fork:** If Gradio 5.current breaks GUI launch (RECON #7), fix launch blockers first as their own commit within this phase; if breakage exceeds ~3 files of fixes, report scope to user before continuing.

### Move 6: Dead-code pass (required pre-refactor step)
**Action:** Per the user's `dead_code_first` invariant: before the >300 LOC refactor in Move 7, remove dead props/params/imports/debug logs from `lora_gui.py` and `common_gui.py` (args deleted upstream, commented-out code, unused imports). Run `black`. Commit separately.
**Expected Observation:** GUI still launches and print-only still emits an argparse-clean command (same check as Move 5); the commit touches no behavior.
**Most Likely Failure:** A "dead" parameter is actually threaded through the flat positional `gui_actions()` arg list, and removing it desynchronizes the Gradio inputs list from the function signature — every field after it silently shifts. **Counter-Move:** After any parameter removal, verify count and order: the `inputs=[…]` list length must equal the `gui_actions` signature length; add a startup `assert` for this while refactoring.

### Move 7: Design the architecture registry (design doc only, no code)
**Action:** Write `wargame/design-arch-registry.md` proposing: a per-architecture spec (display name, training script, caching scripts, model-path fields, arch-specific args, hidden/shown shared fields, config-file key prefix), a top-of-GUI architecture dropdown (or tab set — pick dropdown: preserves one shared settings pane and matches how upstream varies mostly by script+model paths), dynamic show/hide of arch-specific sections via Gradio `visible=` updates, and replacement of the flat 100+-positional-arg `gui_actions()` with a dict/`dataclass` keyed by field name. Include the config save/load migration rule (old HunyuanVideo configs load as architecture="hunyuanvideo" with new fields defaulted).
**Expected Observation:** Doc exists and maps every bucket-(b) and bucket-(c) arg from gap-analysis to a GUI section; user has approved it (this is a phase boundary — explicit approval required).
**Most Likely Failure:** Underestimating how entangled the positional-args pattern is, producing a design that still requires touching every widget for every new arch. **Counter-Move:** The design must pass this test on paper: "adding architecture #13 touches only the registry file plus at most one arch-specific class file." If it doesn't, redesign before coding.

### Move 8: Implement the registry core + wire HunyuanVideo through it
**Action:** Implement per approved design: new `class_architecture.py` (registry) + rework `lora_gui.py` command construction + adapt `class_model.py` for arch-driven model fields. HunyuanVideo is the only registered architecture at the end of this move. ≤5 files.
**Expected Observation:** GUI behaves identically to Move 5's baseline: same print-only command (diff the emitted command text against a saved Move-5 copy — allow only ordering/formatting differences), config save→load round-trips to identical TOML.
**Most Likely Failure:** Gradio dynamic visibility + config load interact badly (loading a config doesn't re-fire visibility updates, leaving wrong fields shown). **Counter-Move:** Route both dropdown changes and config loads through one shared "apply architecture" function; test config load with the GUI open by observing the visible field set change.

### Move 9: Add the first new architecture — Wan 2.1/2.2 (template validation)
**Action:** Register Wan: `wan_train_network.py`, `wan_cache_latents.py`, `wan_cache_text_encoder_outputs.py`, Wan model-path fields (DiT, VAE, T5, CLIP per upstream `docs/wan.md`), `--task` selector (t2v/i2v/t2i variants incl. 2.2), Wan-specific args from gap-analysis. Update `test/config/` with a Wan sample GUI config.
**Expected Observation:** Architecture dropdown shows Wan; print-only emits a command whose script exists and which fails past-argparse (Move 5's validation technique); switching back to HunyuanVideo still emits the baseline command; both configs save/load round-trip.
**Most Likely Failure:** Wan 2.2 needs settings the registry didn't anticipate (dual-model high/low-noise training, timestep boundary args), forcing a registry schema change. **Counter-Move:** Expected — amend the registry schema NOW while there's one extra arch, not after ten; re-verify HunyuanVideo after the schema change.

### Move 10 (template, repeat per architecture): Add architecture X
Execute once per remaining architecture, one phase each, in this priority order (community demand, then release order): **Qwen-Image (incl. Edit/Edit-2509/2511 control images, Layered)**, **FLUX.2**, **FLUX.1 Kontext**, **HunyuanVideo 1.5**, **Z-Image**, **FramePack**, **Kandinsky 5**, **HiDream-O1-Image**, **Ideogram4**, **Krea 2**.
**Action:** Read upstream `docs/<arch>.md` + the saved `--help` dump; add registry entry, arch-specific fields, caching commands; add a sample GUI config under `test/config/`.
**Expected Observation:** Same four checks as Move 9 (command emitted, past-argparse failure, no regression on previously added arch — spot-check HunyuanVideo + the most recent prior arch, config round-trip).
**Most Likely Failure:** Edit/Kontext-style architectures need dataset-config concepts the GUI doesn't model (control images, reference masking) and the arch appears supported but produces untrainable configs. **Counter-Move:** The GUI declaredly does not manage dataset TOMLs (per README) — surface arch-specific dataset requirements as an inline Gradio Markdown help block linking the upstream doc, rather than silently omitting them.
**Fork:** If an architecture's script was marked "dep blocked" in Move 4, skip it, list it in the final report, and continue.

### Move 11: Shared new features
**Action:** In ≤5-file phases as needed: (a) network module dropdown — `networks.lora` / LoHa / LoKr with their `network_args`; (b) fp8 options (block-wise scaling flags); (c) block swap controls (`--blocks_to_swap`, pinned-memory/H2D variants per help dumps); (d) `--save_precision` dropdown (default fp32 per v0.3.2); (e) torch.compile toggle + settings; (f) new optimizer/scheduler entries (Adafactor, REX) in `class_optimizer_and_scheduler.py`; (g) sample generation section if RECON #6 confirmed.
**Expected Observation:** Each feature appears in the emitted command only when set; defaults emit nothing (no arg spam); past-argparse validation passes on at least HunyuanVideo and Wan.
**Most Likely Failure:** A shared flag isn't actually shared (exists on some trainers only) and emits an unrecognized arg on other archs. **Counter-Move:** For each shared flag, grep the saved help dumps: if absent from any arch's dump, gate it per-arch in the registry, not globally.

### Move 12: Backward compatibility + tests
**Action:** Load a pre-mission `config.toml` (grab one from `main` via `git show main:config.toml`) into the new GUI; fix loader so unknown-old/missing-new keys default sanely. Add/extend `test/` with: config round-trip test, registry completeness test (every registered arch's scripts exist on disk), and the inputs-list/signature-length assertion from Move 6 as a real test.
**Expected Observation:** Old config loads without traceback and selects HunyuanVideo; `uv run pytest test/` (or the repo's test invocation) passes.
**Most Likely Failure:** Old config contains since-removed args that now crash the loader. **Counter-Move:** Loader ignores unknown keys with a logged warning, never raises.

### Move 13: Docs, launchers, and ship
**Action:** Update `README.md` (supported architectures table, new Python/torch requirements, screenshots optional), `gui.bat`/`gui.sh` if the Python bump (Move 3 fork) touched them, bump `pyproject.toml` version to 0.3.0. Run `black` over `musubi_tuner_gui/`. Commit; open PR to `main` per repo convention only when the user asks.
**Expected Observation:** Fresh-clone simulation passes: `git clone --recursive` instructions in README, followed literally in a scratch directory, reach a launched GUI.
**Most Likely Failure:** The vendored `uv/` directory contains a stale uv binary that can't resolve the new lockfile. **Counter-Move:** Test `gui.bat`'s uv path explicitly; if stale, update the vendored binary or change launchers to prefer system uv with vendored fallback.

---

## Decision Forks & Branches

- **Upstream ships a real GUI (RECON #1)** → user decision: continue standalone / wrap upstream's / archive this repo. Do not proceed past Move 2 without it.
- **Python bump required (RECON #3)** → fold launcher/venv changes into Move 3's phase.
- **Architecture dep-blocked on Windows (Move 4)** → defer that arch, continue, report.
- **Gradio breakage > ~3 files (Move 5)** → report scope before continuing.
- **Registry schema insufficient (Move 9)** → amend schema at arch #2, re-verify arch #1, then continue.

## Abort Conditions

Stop immediately and report — do not improvise past any of these:
1. Upstream musubi-tuner repo is archived, relicensed non-permissively, or has moved with no successor — the mission premise is void.
2. Move 5 cannot restore a working HunyuanVideo baseline after applying gap-analysis fixes — never build 11 architectures on a broken foundation.
3. `uv sync` cannot produce a torch ≥2.5.1 environment on this Windows machine under any documented index configuration.
4. Any step would require force-pushing, rewriting `main` history, or deleting user files outside the repo.
5. The flat-args → dict refactor (Move 8) cannot reproduce the baseline command byte-for-byte-equivalent and the discrepancy can't be explained — silent config corruption risk.
6. User approval is pending at a phase boundary — waiting is mandatory, not optional.

## Verification Protocol

Run after all moves complete, on a fresh `uv sync`:
1. **GUI launch** — `uv run gui.py` on Windows. Pass looks like: Gradio serves, page renders, zero tracebacks in console.
2. **Launcher parity** — `gui.bat` (and `gui.sh` if WSL/Linux available). Pass looks like: same result as item 1.
3. **Per-architecture command emission** — for every registered architecture, load its `test/config/` sample, use print-only. Pass looks like: emitted script path exists; command with the sample dataset fails past argparse (model/data error, never "unrecognized arguments").
4. **Per-architecture caching commands** — same technique for latent + text-encoder caching per arch. Pass looks like: past-argparse failure or clean run.
5. **Config round-trip** — save config, reload, save again. Pass looks like: second TOML identical to first, for every architecture.
6. **Legacy config load** — `git show main:config.toml` loaded in new GUI. Pass looks like: no traceback, HunyuanVideo selected, warning logs (not errors) for removed keys.
7. **Architecture switching** — cycle the dropdown through all architectures. Pass looks like: field sections show/hide correctly each time, no stale fields from the previous arch visible.
8. **Shared-feature gating** — enable LoHa, fp8, block swap, save_precision, compile on two archs; check one arch where a flag is unsupported. Pass looks like: flag present in command where supported, absent where not.
9. **Test suite** — repo test invocation. Pass looks like: all tests pass, including registry-completeness and inputs-signature-length tests.
10. **Formatting** — `uv run black --check musubi_tuner_gui/`. Pass looks like: no reformatting needed.
11. **One real training smoke test (if GPU available)** — smallest arch (image model, e.g. Qwen-Image or HunyuanVideo image mode) with tiny dataset, `max_train_steps=10`. Pass looks like: training loop starts, loss prints, checkpoint file appears. If no GPU: state explicitly in the final report that this item was skipped and why.
12. **README accuracy** — follow README install/launch instructions literally in a scratch clone. Pass looks like: reaches a launched GUI with no undocumented steps.

## Notes for the Executor

- The single highest-risk artifact is `lora_gui.py`'s flat positional argument passing between Gradio `inputs=[…]` and `gui_actions(…)` — any insertion/removal that desynchronizes order corrupts every downstream field silently. Guard it with the length assertion from Move 6 before touching anything else.
- Validate emitted commands by "fails past argparse," never by `--help` alone — `--help` exits before argument validation and hides renamed-arg drift.
- All help dumps and the gap analysis live under `wargame/args/`; they are the source of truth over possibly-stale upstream docs.
- Upstream pins and docs were fetched 2026-07-10; re-check for newer releases at execution time (Move 2 does this).
