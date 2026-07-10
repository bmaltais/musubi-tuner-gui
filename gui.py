import os
import sys
import argparse
import contextlib
import toml
import gradio as gr

from musubi_tuner_gui.lora_gui import lora_tab
from musubi_tuner_gui.custom_logging import setup_logging
from musubi_tuner_gui.class_gui_config import GUIConfig
from musubi_tuner_gui.settings_gui import settings_tab
from musubi_tuner_gui.dataset_config_gui import dataset_config_tab
import toml

# Constants
PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYPROJECT_FILE_PATH = "./pyproject.toml"


# Function to read file content, suppressing any FileNotFoundError
def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""


# Function to initialize the Gradio UI interface
def initialize_ui_interface(
    config, config_file_path, headless, release_info, readme_content
):
    # Create the main Gradio Blocks interface
    ui_interface = gr.Blocks(title=f"Musubi Tuner GUI {release_info}")
    with ui_interface:
        with gr.Tab("Musubi Tuner"):
            training_dataset_config_component = lora_tab(
                headless=headless, config=config
            )

        with gr.Tab("Dataset Config"):
            dataset_config_tab(
                headless=headless,
                config=config,
                config_file_path=config_file_path,
                training_dataset_config_component=training_dataset_config_component,
            )

        with gr.Tab("Settings"):
            settings_tab(config=config, config_file_path=config_file_path)

        with gr.Tab("About"):
            gr.Markdown(f"Musubi Tuner GUI {release_info}")
            with gr.Tab("README"):
                gr.Markdown(readme_content)
        
        gr.Markdown(f"<div class='ver-class'>{release_info}</div>")
    
    return ui_interface


# Function to configure and launch the UI
def UI(**kwargs):
    """Configure and launch the UI."""
    log.info(f"headless: {kwargs.get('headless', False)}")

    release_info = "Unknown version"
    try:
        with open(PYPROJECT_FILE_PATH, "r", encoding="utf-8") as f:
            pyproject_data = toml.load(f)
            release_info = pyproject_data.get("project", {}).get(
                "version", release_info
            )
    except (FileNotFoundError, toml.TomlDecodeError, KeyError) as e:
        log.error(f"Error loading release information: {e}")

    readme_content = read_file_content("./README.md")
    css = read_file_content("./assets/style.css")

    # Load configuration from the specified file
    config_file_path = kwargs.get("config")
    config = GUIConfig(config_file_path=config_file_path)
    if config.is_config_loaded():
        log.info(f"Loaded default GUI values from '{kwargs.get('config')}'...")

    # Positions the hover-revealed `info=` tooltip (see assets/style.css).
    # The initial enabled/disabled state comes from the Settings tab's
    # persisted config.toml value; the checkbox there live-updates
    # window.MUSUBI_INFO_TOOLTIP_ENABLED via its own js= handler.
    enable_info_tooltip = config.get("settings.enable_info_tooltip", True)
    info_tooltip_js = read_file_content("./assets/js/info_tooltip.js")
    head = (
        f'<script type="text/javascript">window.MUSUBI_INFO_TOOLTIP_ENABLED = {str(enable_info_tooltip).lower()};</script>'
        f'<script type="text/javascript">{info_tooltip_js}</script>'
    )

    # Initialize the Gradio UI interface
    ui_interface = initialize_ui_interface(
        config,
        config_file_path,
        kwargs.get("headless", False),
        release_info,
        readme_content,
    )

    launch_params = {
        "server_name": kwargs.get("listen"),
        "auth": (
            (kwargs["username"], kwargs["password"])
            if kwargs.get("username") and kwargs.get("password")
            else None
        ),
        "server_port": (
            kwargs.get("server_port", 0) if kwargs.get("server_port", 0) > 0 else None
        ),
        "inbrowser": kwargs.get("inbrowser", False),
        "share": (
            False if kwargs.get("do_not_share", False) else kwargs.get("share", False)
        ),
        "root_path": kwargs.get("root_path", None),
        "debug": kwargs.get("debug", False),
        "css": css,
        "head": head,
        "theme": gr.themes.Default(),
    }

    # This line filters out any key-value pairs from `launch_params` where the value is `None`, ensuring only valid parameters are passed to the `launch` function.
    launch_params = {k: v for k, v in launch_params.items() if v is not None}
    ui_interface.launch(**launch_params)


# Function to initialize argument parser for command-line arguments
def initialize_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.toml",
        help="Path to the toml config file for interface defaults",
    )
    parser.add_argument("--debug", action="store_true", help="Debug on")
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port", type=int, default=0, help="Port to run the server listener on"
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument(
        "--headless", action="store_true", help="Is the server headless"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Set custom language"
    )
    parser.add_argument("--use-ipex", action="store_true", help="Use IPEX environment")
    parser.add_argument("--use-rocm", action="store_true", help="Use ROCm environment")
    parser.add_argument(
        "--do_not_use_shell",
        action="store_true",
        help="Enforce not to use shell=True when running external commands",
    )
    parser.add_argument(
        "--do_not_share", action="store_true", help="Do not share the gradio UI"
    )
    parser.add_argument(
        "--requirements",
        type=str,
        default=None,
        help="requirements file to use for validation",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=None,
        help="`root_path` for Gradio to enable reverse proxy support. e.g. /kohya_ss",
    )
    parser.add_argument(
        "--noverify", action="store_true", help="Disable requirements verification"
    )
    return parser


if __name__ == "__main__":
    parser = initialize_arg_parser()
    args = parser.parse_args()
    log = setup_logging(debug=args.debug)

    # Launch the UI with the provided arguments
    UI(**vars(args))
