# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## Project Scope (root-owned)

Gradio GUI for [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Cross-platform (Windows/Linux; MacOS welcome). Package manager: **uv**. Python `>=3.10,<3.11`. Version source: `pyproject.toml` → `project.version`.

### Root-owned paths

| Path | Role |
|------|------|
| `gui.py` | Gradio app entry: CLI flags, config load, Blocks launch |
| `gui.bat` / `gui.sh` | Platform launchers; prepend vendored `uv/` then `uv run gui.py` |
| `pyproject.toml` | Project metadata; path dep on `musubi-tuner` |
| `config.toml` | Default GUI field values (often machine-local paths—do not treat as portable secrets store) |
| `README.md` | Install/start docs for humans |
| `uv/windows`, `uv/linux` | Vendored uv binaries for offline-friendly launch |
| `.gitmodules` | Submodule declaration for backend |
| `LICENSE` | License text |

### Cross-cutting rules

- Run from repo root so relative paths (`./assets`, `./musubi-tuner`, `./config.toml`, `./test/...`) resolve
- GUI authors dataset TOML files via the Dataset Config tab, but the *schema* is owned upstream: field names, types, and validation rules must track `musubi-tuner/docs/dataset_config.md` (see `musubi_tuner_gui/AGENTS.md`)
- Prefer `uv run` over bare `pip`/`python` for app and tool invocation
- Do not commit secrets (`huggingface_token`, `wandb_api_key`) or large personal weight paths as if they were shared defaults
- Gitignored runtime: `musubi_tuner_gui.log`, `test/output`, `uv.lock`, `*.npz`
- MVP work historically targeted `dev`; treat `main` release posture per README until product policy changes

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

| Child | Path | Owns |
|-------|------|------|
| GUI package | [`musubi_tuner_gui/AGENTS.md`](musubi_tuner_gui/AGENTS.md) | Gradio panels, command assembly, config helpers, logging |
| Front-end assets | [`assets/AGENTS.md`](assets/AGENTS.md) | CSS and browser JS for the Gradio shell |
| Manual test fixtures | [`test/AGENTS.md`](test/AGENTS.md) | Sample dataset, dataset TOML, cache fixtures, smoke-test notes |
| Training backend submodule | [`musubi-tuner/AGENTS.md`](musubi-tuner/AGENTS.md) | Submodule boundary and invocation contract for kohya-ss/musubi-tuner |

### Intentionally root-only (no child AGENTS.md)

- `uv/` — vendored binaries only; no product logic
- Top-level launchers and packaging files listed under Project Scope
