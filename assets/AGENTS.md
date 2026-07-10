# assets

## Purpose

Static front-end assets for the Gradio shell: CSS theme overrides and browser JS helpers (tab hooks, localization).

## Ownership

- Owns: `style.css`, `js/script.js`, `js/localization.js`, `js/info_tooltip.js`
- Does not own: Gradio layout/components (built in `musubi_tuner_gui/`), app launch CSS injection (`gui.py` reads `./assets/style.css`)

## Local Contracts

- `gui.py` loads `./assets/style.css` into `gr.Blocks(css=...)` at startup; missing file is treated as empty string
- `js/script.js` provides Gradio DOM helpers (`gradioApp`, tab callbacks)—pattern inherited from AUTOMATIC1111 / Kohya-style UIs
- `js/localization.js` applies `window.localization` string maps to text nodes when present
- `js/info_tooltip.js` is inlined into `gr.Blocks(head=...)` by `gui.py` (not `<script src>`-loaded) alongside a bootstrap line setting `window.MUSUBI_INFO_TOOLTIP_ENABLED` from `config.toml`; it globally hooks `mouseover`/`focusin` to reveal Gradio's `.info-text` divs as fixed-position tooltips. The Settings tab checkbox (`musubi_tuner_gui/settings_gui.py`) live-toggles the same `window` flag via its `js=` handler and persists the choice—see that file and the matching `.info-text` CSS block below for the full contract.
- Paths are repo-relative from process CWD (expected: repo root)

## Work Guidance

- Keep CSS selectors compatible with Gradio 5+ DOM where practical
- Prefer small, dependency-free browser JS; no bundler in this repo
- Version or feature-detect if Gradio markup changes break selectors

## Verification

- Visual: launch GUI and confirm custom styles apply (e.g. version div `.ver-class` if styled)
- No automated front-end tests

## Child DOX Index

None. `js/` is too small to warrant a separate DOX node.
