"""Dataset Config tab: create/open/edit/validate/save the musubi-tuner dataset TOML.

Deliberately avoids @gr.render dynamic components (master/detail pattern with
fixed widgets + an explicit "Apply changes" button instead of per-widget
change-sync) to sidestep re-render focus loss and index-vs-state races.

Must not import lora_gui (dependency arrow points the other way / nowhere).
"""

import os

import gradio as gr

from .class_gui_config import GUIConfig
from .common_gui import get_file_path, get_folder_path, get_saveasfile_path
from .custom_logging import setup_logging
from .dataset_config_toml import (
    DATASET_KNOWN_KEYS,
    FRAME_EXTRACTION_CHOICES,
    SOURCE_KEYS,
    VIDEO_SOURCE_KEYS,
    load_dataset_config,
    parse_int_list,
    parse_int_pair,
    save_dataset_config,
    validate_dataset_config,
)

log = setup_logging()

DATASET_TYPE_CHOICES = [
    ("Image directory", "image_directory"),
    ("Image JSONL file", "image_jsonl_file"),
    ("Video directory", "video_directory"),
    ("Video JSONL file", "video_jsonl_file"),
]

# Number of values in the detail-editor tuple (see _dataset_to_editor_values /
# _empty_editor_values). Asserted at module import time so an accidental
# widget/output-list drift fails loudly instead of silently.
DETAIL_EDITOR_FIELD_COUNT = 19


# ---------------------------------------------------------------------------
# Pure(-ish) helpers: dict <-> editor widget values. No Gradio state mutation.
# ---------------------------------------------------------------------------


def _dataset_type_from_dict(ds: dict) -> str:
    for k in SOURCE_KEYS:
        if ds.get(k):
            return k
    for k in SOURCE_KEYS:
        if k in ds:
            return k
    return "image_directory"


def _unknown_keys_note(ds: dict) -> str:
    unknown = sorted(k for k in ds.keys() if k not in DATASET_KNOWN_KEYS)
    if not unknown:
        return ""
    return "Advanced keys present (preserved on save): " + ", ".join(unknown)


def _pair_to_text(value) -> str:
    if not value:
        return ""
    return f"{value[0]},{value[1]}"


def _list_to_text(value) -> str:
    if not value:
        return ""
    return ",".join(str(v) for v in value)


def _dataset_to_editor_values(ds: dict) -> tuple:
    dtype = _dataset_type_from_dict(ds)
    source_fps = ds.get("source_fps")
    values = (
        dtype,
        ds.get(dtype, ""),
        ds.get("cache_directory", ""),
        ds.get("control_directory", ""),
        ds.get("caption_extension", ""),
        _pair_to_text(ds.get("resolution")),
        ds.get("batch_size"),
        ds.get("num_repeats"),
        bool(ds.get("enable_bucket", False)),
        bool(ds.get("bucket_no_upscale", False)),
        bool(ds.get("no_resize_control", False)),
        _pair_to_text(ds.get("control_resolution")),
        _list_to_text(ds.get("target_frames")),
        ds.get("frame_extraction", FRAME_EXTRACTION_CHOICES[0]),
        ds.get("frame_stride"),
        ds.get("frame_sample"),
        ds.get("max_frames"),
        "" if source_fps is None else str(source_fps),
        _unknown_keys_note(ds),
    )
    assert len(values) == DETAIL_EDITOR_FIELD_COUNT
    return values


def _empty_editor_values() -> tuple:
    values = (
        "image_directory",
        "",
        "",
        "",
        "",
        "",
        None,
        None,
        False,
        False,
        False,
        "",
        "",
        FRAME_EXTRACTION_CHOICES[0],
        None,
        None,
        None,
        "",
        "",
    )
    assert len(values) == DETAIL_EDITOR_FIELD_COUNT
    return values


def _apply_editor_to_dataset(
    raw_ds: dict,
    dtype,
    source_path,
    cache_directory,
    control_directory,
    caption_extension,
    resolution_text,
    batch_size,
    num_repeats,
    enable_bucket,
    bucket_no_upscale,
    no_resize_control,
    control_resolution_text,
    target_frames_text,
    frame_extraction,
    frame_stride,
    frame_sample,
    max_frames,
    source_fps_text,
) -> dict:
    """Merge editor widget values onto raw_ds. Unknown keys pass through untouched;
    known keys are authoritative from the widgets (blank/None deletes the key)."""
    merged = dict(raw_ds)

    for k in SOURCE_KEYS:
        merged.pop(k, None)
    if source_path:
        merged[dtype] = source_path

    def set_or_delete(key, value):
        if value in (None, ""):
            merged.pop(key, None)
        else:
            merged[key] = value

    set_or_delete("cache_directory", cache_directory)
    set_or_delete("control_directory", control_directory)
    set_or_delete("caption_extension", caption_extension)

    if resolution_text:
        merged["resolution"] = parse_int_pair(resolution_text)
    else:
        merged.pop("resolution", None)

    set_or_delete(
        "batch_size", int(batch_size) if batch_size not in (None, "") else None
    )
    set_or_delete(
        "num_repeats", int(num_repeats) if num_repeats not in (None, "") else None
    )
    merged["enable_bucket"] = bool(enable_bucket)
    merged["bucket_no_upscale"] = bool(bucket_no_upscale)
    merged["no_resize_control"] = bool(no_resize_control)

    if control_resolution_text:
        merged["control_resolution"] = parse_int_pair(control_resolution_text)
    else:
        merged.pop("control_resolution", None)

    if dtype in VIDEO_SOURCE_KEYS:
        if target_frames_text:
            merged["target_frames"] = parse_int_list(target_frames_text)
        else:
            merged.pop("target_frames", None)
        set_or_delete("frame_extraction", frame_extraction)
        set_or_delete(
            "frame_stride",
            int(frame_stride) if frame_stride not in (None, "") else None,
        )
        set_or_delete(
            "frame_sample",
            int(frame_sample) if frame_sample not in (None, "") else None,
        )
        set_or_delete(
            "max_frames", int(max_frames) if max_frames not in (None, "") else None
        )
        if source_fps_text not in (None, ""):
            merged["source_fps"] = float(source_fps_text)
        else:
            merged.pop("source_fps", None)
    else:
        for k in (
            "target_frames",
            "frame_extraction",
            "frame_stride",
            "frame_sample",
            "max_frames",
            "source_fps",
        ):
            merged.pop(k, None)

    return merged


def _general_to_widgets(general: dict) -> tuple:
    return (
        _pair_to_text(general.get("resolution")),
        general.get("caption_extension", ""),
        general.get("batch_size"),
        general.get("num_repeats"),
        bool(general.get("enable_bucket", False)),
        bool(general.get("bucket_no_upscale", False)),
    )


def _general_from_widgets(
    resolution_text,
    caption_extension,
    batch_size,
    num_repeats,
    enable_bucket,
    bucket_no_upscale,
) -> dict:
    general = {}
    if resolution_text:
        general["resolution"] = parse_int_pair(resolution_text)
    if caption_extension:
        general["caption_extension"] = caption_extension
    if batch_size not in (None, ""):
        general["batch_size"] = int(batch_size)
    if num_repeats not in (None, ""):
        general["num_repeats"] = int(num_repeats)
    general["enable_bucket"] = bool(enable_bucket)
    general["bucket_no_upscale"] = bool(bucket_no_upscale)
    return general


def _dataset_summary_rows(datasets: list) -> list:
    rows = []
    for i, ds in enumerate(datasets):
        dtype = next((k for k in SOURCE_KEYS if ds.get(k)), None)
        source = ds.get(dtype, "") if dtype else ""
        rows.append(
            [
                i,
                dtype or "(no source)",
                source,
                ds.get("cache_directory", ""),
                ds.get("num_repeats", ""),
            ]
        )
    return rows


def _format_validation(messages: list) -> str:
    if not messages:
        return "No validation issues found."
    return "\n".join(messages)


def _status_text(selected_index, datasets: list) -> str:
    if selected_index is None or not (0 <= selected_index < len(datasets)):
        return (
            "**No dataset selected.** Pick a *Dataset type* below and click **Browse** next to "
            "*Source path* to start one from an existing folder/file, or click **Add image dataset** "
            "/ **Add video dataset** for a blank row."
        )
    return (
        f"**Editing dataset #{selected_index}** of {len(datasets)}. Fill in the fields below, then "
        "click **Apply changes** to save them into this row (Save/Save as writes the whole list to disk)."
    )


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------


def dataset_config_tab(
    headless=False,
    config: GUIConfig = {},
    config_file_path: str = "./config.toml",
    training_dataset_config_component=None,
):
    gr.Markdown(
        "Create, open, edit, validate, and save a musubi-tuner dataset TOML file. "
        "Comments in hand-written files are not preserved on save; unknown/advanced "
        "keys are preserved untouched.\n\n"
        "**How to build a dataset from scratch:**\n"
        "1. Set defaults in **General** below (they apply to every dataset unless overridden per-dataset).\n"
        "2. Pick a *Dataset type* under **Selected dataset**, then click **Browse** next to *Source path* "
        "and choose your image/video folder (or jsonl file) — this creates a new row in **Datasets** for you.\n"
        "3. Fill in the rest of the fields for that dataset (cache directory, caption extension, etc.), then click "
        "**Apply changes** to save them into the row.\n"
        "4. Repeat step 2-3 for more datasets, or click a row in the **Datasets** table to switch which one you're editing.\n"
        "5. Check **Validation** — ERRORs block saving, WARNINGs don't — then click **Save** or **Save as**."
    )

    datasets_state = gr.State([])
    selected_index_state = gr.State(None)

    with gr.Row():
        dataset_path = gr.Textbox(
            label="Dataset Config File",
            placeholder="Path to the dataset TOML file",
            value=str(config.get("settings.dataset_config_edit_path", "")),
            scale=4,
        )
        button_open = gr.Button("📂 Open", visible=(not headless))
        button_save = gr.Button("💾 Save")
        button_save_as = gr.Button("💾 Save as", visible=(not headless))

    with gr.Accordion("General", open=True):
        with gr.Row():
            general_resolution = gr.Textbox(
                label="Resolution (W,H)", placeholder="960,544"
            )
            general_caption_extension = gr.Textbox(
                label="Caption Extension", placeholder=".txt"
            )
            general_batch_size = gr.Number(label="Batch Size", precision=0)
            general_num_repeats = gr.Number(label="Num Repeats", precision=0)
        with gr.Row():
            general_enable_bucket = gr.Checkbox(label="Enable Bucket", value=True)
            general_bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale")

    general_widgets = [
        general_resolution,
        general_caption_extension,
        general_batch_size,
        general_num_repeats,
        general_enable_bucket,
        general_bucket_no_upscale,
    ]

    gr.Markdown("### Datasets")
    with gr.Row():
        button_add_image = gr.Button("Add image dataset")
        button_add_video = gr.Button("Add video dataset")
        button_duplicate = gr.Button("Duplicate selected")
        button_remove = gr.Button("Remove selected")

    datasets_table = gr.Dataframe(
        headers=["#", "Type", "Source", "Cache dir", "Repeats"],
        datatype=["number", "str", "str", "str", "number"],
        interactive=False,
        row_count=(0, "dynamic"),
    )

    gr.Markdown("### Selected dataset")
    status_markdown = gr.Markdown(_status_text(None, []))
    dtype_radio = gr.Radio(
        label="Dataset type",
        choices=DATASET_TYPE_CHOICES,
        value="image_directory",
    )
    with gr.Row():
        source_path = gr.Textbox(label="Source path", scale=4)
        button_browse_source = gr.Button("Browse")
    with gr.Row():
        cache_directory = gr.Textbox(label="Cache directory", scale=4)
        button_browse_cache = gr.Button("Browse")
    with gr.Row():
        control_directory = gr.Textbox(label="Control directory", scale=4)
        button_browse_control = gr.Button("Browse")

    with gr.Row():
        caption_extension = gr.Textbox(
            label="Caption Extension (override)", placeholder=".txt"
        )
        resolution = gr.Textbox(
            label="Resolution override (W,H)", placeholder="960,544"
        )
        batch_size = gr.Number(label="Batch Size override", precision=0)
        num_repeats = gr.Number(label="Num Repeats", precision=0)

    with gr.Row():
        enable_bucket = gr.Checkbox(label="Enable Bucket override")
        bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale override")
        no_resize_control = gr.Checkbox(label="No Resize Control")

    control_resolution = gr.Textbox(
        label="Control Resolution (W,H)", placeholder="960,544"
    )

    with gr.Column(visible=False) as video_group:
        gr.Markdown("Video-only fields")
        target_frames = gr.Textbox(label="Target Frames", placeholder="1,25,45")
        with gr.Row():
            frame_extraction = gr.Dropdown(
                label="Frame Extraction",
                choices=FRAME_EXTRACTION_CHOICES,
                value=FRAME_EXTRACTION_CHOICES[0],
            )
            frame_stride = gr.Number(label="Frame Stride", precision=0)
            frame_sample = gr.Number(label="Frame Sample", precision=0)
            max_frames = gr.Number(label="Max Frames", precision=0)
        source_fps = gr.Textbox(label="Source FPS", placeholder="30")

    unknown_keys_note = gr.Markdown("")

    button_apply = gr.Button("Apply changes to selected dataset", variant="primary")

    validation_panel = gr.Textbox(label="Validation", interactive=False, lines=6)

    detail_editor_widgets = [
        dtype_radio,
        source_path,
        cache_directory,
        control_directory,
        caption_extension,
        resolution,
        batch_size,
        num_repeats,
        enable_bucket,
        bucket_no_upscale,
        no_resize_control,
        control_resolution,
        target_frames,
        frame_extraction,
        frame_stride,
        frame_sample,
        max_frames,
        source_fps,
        unknown_keys_note,
    ]
    assert len(detail_editor_widgets) == DETAIL_EDITOR_FIELD_COUNT

    dtype_radio.change(
        fn=lambda dtype: gr.Column(visible=(dtype in VIDEO_SOURCE_KEYS)),
        inputs=[dtype_radio],
        outputs=[video_group],
    )

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    def on_select_row(datasets, evt: gr.SelectData):
        idx = evt.index[0] if evt.index is not None else None
        if idx is None or not (0 <= idx < len(datasets)):
            return (None, _status_text(None, datasets)) + _empty_editor_values()
        return (idx, _status_text(idx, datasets)) + _dataset_to_editor_values(
            datasets[idx]
        )

    datasets_table.select(
        fn=on_select_row,
        inputs=[datasets_state],
        outputs=[selected_index_state, status_markdown] + detail_editor_widgets,
    )

    def add_dataset(datasets, is_video):
        ds = {"video_directory": ""} if is_video else {"image_directory": ""}
        datasets = list(datasets) + [ds]
        idx = len(datasets) - 1
        return (
            datasets,
            _dataset_summary_rows(datasets),
            idx,
            _status_text(idx, datasets),
        ) + _dataset_to_editor_values(ds)

    button_add_image.click(
        fn=lambda datasets: add_dataset(datasets, False),
        inputs=[datasets_state],
        outputs=[datasets_state, datasets_table, selected_index_state, status_markdown]
        + detail_editor_widgets,
    )
    button_add_video.click(
        fn=lambda datasets: add_dataset(datasets, True),
        inputs=[datasets_state],
        outputs=[datasets_state, datasets_table, selected_index_state, status_markdown]
        + detail_editor_widgets,
    )

    def duplicate_dataset(datasets, selected_index):
        if selected_index is None or not (0 <= selected_index < len(datasets)):
            editor_vals = (
                _empty_editor_values()
                if not datasets
                else _dataset_to_editor_values(datasets[0])
            )
            fallback_idx = None if not datasets else 0
            return (
                datasets,
                _dataset_summary_rows(datasets),
                fallback_idx,
                _status_text(fallback_idx, datasets),
            ) + editor_vals
        ds = dict(datasets[selected_index])
        datasets = list(datasets) + [ds]
        idx = len(datasets) - 1
        return (
            datasets,
            _dataset_summary_rows(datasets),
            idx,
            _status_text(idx, datasets),
        ) + _dataset_to_editor_values(ds)

    button_duplicate.click(
        fn=duplicate_dataset,
        inputs=[datasets_state, selected_index_state],
        outputs=[datasets_state, datasets_table, selected_index_state, status_markdown]
        + detail_editor_widgets,
    )

    def remove_dataset(datasets, selected_index):
        if selected_index is None or not (0 <= selected_index < len(datasets)):
            return (
                datasets,
                _dataset_summary_rows(datasets),
                selected_index,
                _status_text(selected_index, datasets),
            ) + _empty_editor_values()
        datasets = list(datasets)
        del datasets[selected_index]
        if not datasets:
            return (
                datasets,
                _dataset_summary_rows(datasets),
                None,
                _status_text(None, datasets),
            ) + _empty_editor_values()
        new_idx = min(selected_index, len(datasets) - 1)
        return (
            datasets,
            _dataset_summary_rows(datasets),
            new_idx,
            _status_text(new_idx, datasets),
        ) + _dataset_to_editor_values(datasets[new_idx])

    button_remove.click(
        fn=remove_dataset,
        inputs=[datasets_state, selected_index_state],
        outputs=[datasets_state, datasets_table, selected_index_state, status_markdown]
        + detail_editor_widgets,
    )

    def browse_source(dtype, current, datasets, selected_index):
        if dtype in ("image_directory", "video_directory"):
            new_path = get_folder_path(current)
        else:
            new_path = get_file_path(
                current,
                default_extension=".jsonl",
                extension_name="JSONL files (*.jsonl)",
            )

        if not new_path or new_path == current:
            # Dialog cancelled or unchanged: leave dataset state untouched.
            return (
                datasets,
                _dataset_summary_rows(datasets),
                selected_index,
                _status_text(selected_index, datasets),
                new_path,
            )

        if selected_index is None or not (0 <= selected_index < len(datasets)):
            # Nothing selected yet: picking a source folder/file starts a new dataset row.
            ds = {dtype: new_path}
            datasets = list(datasets) + [ds]
            idx = len(datasets) - 1
            return (
                datasets,
                _dataset_summary_rows(datasets),
                idx,
                _status_text(idx, datasets),
                new_path,
            )

        # A dataset row is already selected: just update the field; Apply changes commits it.
        return (
            datasets,
            _dataset_summary_rows(datasets),
            selected_index,
            _status_text(selected_index, datasets),
            new_path,
        )

    button_browse_source.click(
        fn=browse_source,
        inputs=[dtype_radio, source_path, datasets_state, selected_index_state],
        outputs=[
            datasets_state,
            datasets_table,
            selected_index_state,
            status_markdown,
            source_path,
        ],
    )
    button_browse_cache.click(
        fn=get_folder_path, inputs=[cache_directory], outputs=[cache_directory]
    )
    button_browse_control.click(
        fn=get_folder_path, inputs=[control_directory], outputs=[control_directory]
    )

    def apply_changes(datasets, selected_index, *values):
        editor_values = values[
            : DETAIL_EDITOR_FIELD_COUNT - 1
        ]  # exclude unknown_keys_note (read-only)
        general_values = values[DETAIL_EDITOR_FIELD_COUNT - 1 :]

        try:
            general = _general_from_widgets(*general_values)
        except ValueError as e:
            return (
                datasets,
                _dataset_summary_rows(datasets),
                "",
                f"ERROR: General: {e}",
            )

        if selected_index is None or not (0 <= selected_index < len(datasets)):
            validation = validate_dataset_config(general, datasets)
            return (
                datasets,
                _dataset_summary_rows(datasets),
                "",
                _format_validation(validation),
            )

        try:
            merged = _apply_editor_to_dataset(datasets[selected_index], *editor_values)
        except ValueError as e:
            return (
                datasets,
                _dataset_summary_rows(datasets),
                f"ERROR: {e}",
                f"ERROR: {e}",
            )

        datasets = list(datasets)
        datasets[selected_index] = merged
        validation = validate_dataset_config(general, datasets)
        return (
            datasets,
            _dataset_summary_rows(datasets),
            _unknown_keys_note(merged),
            _format_validation(validation),
        )

    button_apply.click(
        fn=apply_changes,
        inputs=[datasets_state, selected_index_state]
        + detail_editor_widgets[:-1]
        + general_widgets,
        outputs=[datasets_state, datasets_table, unknown_keys_note, validation_panel],
    )

    def open_dataset_config(ask_for_file, path):
        original = path
        if ask_for_file:
            path = get_file_path(
                path, default_extension=".toml", extension_name="TOML files (*.toml)"
            )
        if not path:
            path = original
        if not path or not os.path.isfile(path):
            return (
                (path, [], _dataset_summary_rows([]), None, _status_text(None, []))
                + _general_to_widgets({})
                + _empty_editor_values()
                + ("No file loaded.",)
            )

        data = load_dataset_config(path)
        general = data["general"]
        datasets = data["datasets"]
        idx = 0 if datasets else None
        editor_vals = (
            _dataset_to_editor_values(datasets[0])
            if datasets
            else _empty_editor_values()
        )
        validation = validate_dataset_config(general, datasets)
        return (
            (
                path,
                datasets,
                _dataset_summary_rows(datasets),
                idx,
                _status_text(idx, datasets),
            )
            + _general_to_widgets(general)
            + editor_vals
            + (_format_validation(validation),)
        )

    button_open.click(
        fn=lambda path: open_dataset_config(True, path),
        inputs=[dataset_path],
        outputs=[
            dataset_path,
            datasets_state,
            datasets_table,
            selected_index_state,
            status_markdown,
        ]
        + general_widgets
        + detail_editor_widgets
        + [validation_panel],
    )

    def do_save(path, datasets, save_as, *general_values):
        try:
            general = _general_from_widgets(*general_values)
        except ValueError as e:
            return path, f"ERROR: General: {e}"
        validation = validate_dataset_config(general, datasets)
        errors = [m for m in validation if m.startswith("ERROR:")]
        if errors:
            return path, _format_validation(validation)

        if save_as or not path:
            new_path = get_saveasfile_path(
                path, defaultextension=".toml", extension_name="TOML files (*.toml)"
            )
            if not new_path:
                return path, _format_validation(validation)
            path = new_path

        save_dataset_config(path, general, datasets)
        config.config.setdefault("settings", {})["dataset_config_edit_path"] = path
        config.save_config(config.config, config_file_path)
        log.info(f"Dataset config saved to {path}")
        return path, _format_validation(validation)

    button_save.click(
        fn=lambda path, datasets, *g: do_save(path, datasets, False, *g),
        inputs=[dataset_path, datasets_state] + general_widgets,
        outputs=[dataset_path, validation_panel],
    )
    button_save_as.click(
        fn=lambda path, datasets, *g: do_save(path, datasets, True, *g),
        inputs=[dataset_path, datasets_state] + general_widgets,
        outputs=[dataset_path, validation_panel],
    )

    if training_dataset_config_component is not None:
        button_use_in_training = gr.Button("Use this file in training tab")
        button_use_in_training.click(
            fn=lambda path: path,
            inputs=[dataset_path],
            outputs=[training_dataset_config_component],
        )
