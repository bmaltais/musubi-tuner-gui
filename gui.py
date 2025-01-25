import os
import sys
import argparse
import contextlib
import logging
import toml
import gradio as gr

from musubi_tuner_gui.lora_gui import lora_tab
from musubi_tuner_gui.custom_logging import setup_logging
from musubi_tuner_gui.class_gui_config import GUIConfig

# Constants
PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_FILE_PATH = "./assets/style.css"
PYPROJECT_FILE_PATH = "./pyproject.toml"
README_FILE_PATH = "./README.md"

def read_file_content(file_path: str) -> str:
    """Read file content, suppressing any FileNotFoundError."""
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""

def initialize_ui_interface(config: GUIConfig, headless: bool, release_info: str, readme_content: str) -> gr.Blocks:
    """Initialize the Gradio UI interface."""
    css = read_file_content(CSS_FILE_PATH)
    ui_interface = gr.Blocks(css=css, title=f"Musubi Tuner GUI {release_info}", theme=gr.themes.Default())
    
    with ui_interface:
        with gr.Tab("Musubi Tuner"):
            lora_tab(headless=headless, config=config)
        
        with gr.Tab("About"):
            gr.Markdown(f"Musubi Tuner GUI {release_info}")
            with gr.Tab("README"):
                gr.Markdown(readme_content)
        
        gr.Markdown(f"<div class='ver-class'>{release_info}</div>")
    
    return ui_interface

def UI(**kwargs):
    """Configure and launch the UI."""
    log.info(f"headless: {kwargs.get('headless', False)}")

    release_info = "Unknown version"
    try:
        with open(PYPROJECT_FILE_PATH, "r", encoding="utf-8") as f:
            pyproject_data = toml.load(f)
            release_info = pyproject_data.get("project", {}).get("version", release_info)
    except (FileNotFoundError, toml.TomlDecodeError, KeyError) as e:
        log.error(f"Error loading release information: {e}")
    
    readme_content = read_file_content(README_FILE_PATH)
    
    config = GUIConfig(config_file_path=kwargs.get("config"))
    if config.is_config_loaded():
        log.info(f"Loaded default GUI values from '{kwargs.get('config')}'...")

    ui_interface = initialize_ui_interface(config, kwargs.get("headless", False), release_info, readme_content)

    launch_params = {
        "server_name": kwargs.get("listen"),
        "auth": (kwargs["username"], kwargs["password"]) if kwargs.get("username") and kwargs.get("password") else None,
        "server_port": kwargs.get("server_port", 0) if kwargs.get("server_port", 0) > 0 else None,
        "inbrowser": kwargs.get("inbrowser", False),
        "share": False if kwargs.get("do_not_share", False) else kwargs.get("share", False),
        "root_path": kwargs.get("root_path", None),
        "debug": kwargs.get("debug", False),
    }
    
    launch_params = {k: v for k, v in launch_params.items() if v is not None}
    ui_interface.launch(**launch_params)

def initialize_arg_parser() -> argparse.ArgumentParser:
    """Initialize argument parser for command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.toml", help="Path to the toml config file for interface defaults")
    parser.add_argument("--debug", action="store_true", help="Debug on")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP to listen on for connections to Gradio")
    parser.add_argument("--username", type=str, default="", help="Username for authentication")
    parser.add_argument("--password", type=str, default="", help="Password for authentication")
    parser.add_argument("--server_port", type=int, default=0, help="Port to run the server listener on")
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument("--headless", action="store_true", help="Is the server headless")
    parser.add_argument("--do_not_share", action="store_true", help="Do not share the gradio UI")
    parser.add_argument("--root_path", type=str, default=None, help="`root_path` for Gradio to enable reverse proxy support. e.g. /kohya_ss")
    return parser

if __name__ == "__main__":
    parser = initialize_arg_parser()
    args = parser.parse_args()
    log = setup_logging(debug=args.debug)
    UI(**vars(args))
