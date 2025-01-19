import gradio as gr
import toml
import time
from .class_gui_config import GUIConfig
from .class_command_executor import CommandExecutor

train_state_value = time.time()

class LatentCaching:
    def __init__(
        self,
        headless: bool,
        config: GUIConfig,
    ) -> None:
        self.config = config
        self.headless = headless

        # Initialize the UI components
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.caching_latent_device = gr.Textbox(
                label="Device",
                placeholder="Device to use (default is CUDA if available)",
                value=self.config.get("caching_latent_device", "cuda"),
                interactive=True,
            )
            self.caching_latent_batch_size = gr.Number(
                label="Batch Size",
                value=self.config.get("caching_latent_batch_size", None),
                step=1,
                interactive=True,
                info="Override dataset config if dataset batch size > this"
            )
            self.caching_latent_num_workers = gr.Number(
                label="Number of Workers",
                value=self.config.get("caching_latent_num_workers", None),
                step=1,
                interactive=True,
                info="Default is CPU count - 1"
            )

        with gr.Row():
            self.caching_latent_skip_existing = gr.Checkbox(
                label="Skip Existing",
                value=self.config.get("caching_latent_skip_existing", False),
                interactive=True,
                info="Skip existing cache files"
            )
            self.caching_latent_keep_cache = gr.Checkbox(
                label="Keep Cache",
                value=self.config.get("caching_latent_keep_cache", False),
                interactive=True,
                info="Keep cache files not in dataset"
            )

        with gr.Row():
            self.caching_latent_debug_mode = gr.Dropdown(
                label="Debug Mode",
                choices=["image", "console"],
                allow_custom_value=True,
                value=self.config.get("caching_latent_debug_mode", None),
                interactive=True,
            )
            self.caching_latent_console_width = gr.Number(
                label="Console Width",
                value=self.config.get("caching_latent_console_width", 80),
                step=1,
                interactive=True,
                info="Console width for debug mode"
            )
            self.caching_latent_console_back = gr.Textbox(
                label="Console Background Color",
                placeholder="Background color for debug console",
                value=self.config.get("caching_latent_console_back", None),
                interactive=True,
            )

        with gr.Row():
            self.caching_latent_console_num_images = gr.Number(
                label="Number of Images",
                value=self.config.get("caching_latent_console_num_images", None),
                step=1,
                interactive=True,
                info="Number of images to show for each dataset in debug mode"
            )