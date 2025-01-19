import gradio as gr
import toml
from .class_gui_config import GUIConfig

class TextEncoderOutputsCaching:
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
            self.caching_teo_text_encoder1 = gr.Textbox(
                label="Text Encoder 1 Directory",
                placeholder="Path to Text Encoder 1 directory",
                value=self.config.get("caching_teo_text_encoder1", ""),
                interactive=True,
            )
            self.caching_teo_text_encoder2 = gr.Textbox(
                label="Text Encoder 2 Directory",
                placeholder="Path to Text Encoder 2 directory",
                value=self.config.get("caching_teo_text_encoder2", ""),
                interactive=True,
            )
            self.caching_teo_text_encoder_dtype = gr.Dropdown(
                label="Text Encoder Data Type",
                choices=["float16", "bfloat16"],
                value=self.config.get("caching_teo_text_encoder_dtype", "float16"),
                interactive=True,
                info="Default is float16"
            )

        with gr.Row():
            self.caching_teo_device = gr.Textbox(
                label="Device",
                placeholder="Device to use (default is CUDA if available)",
                value=self.config.get("caching_teo_device", "cuda"),
                interactive=True,
            )
            self.caching_teo_fp8_llm = gr.Checkbox(
                label="Use FP8 for LLM",
                value=self.config.get("caching_teo_fp8_llm", False),
                interactive=True,
                info="Enable FP8 for Text Encoder 1"
            )
            self.caching_teo_batch_size = gr.Number(
                label="Batch Size",
                value=self.config.get("caching_teo_batch_size", None),
                step=1,
                interactive=True,
                info="Override dataset config if dataset batch size > this"
            )
            self.caching_teo_num_workers = gr.Number(
                label="Number of Workers",
                value=self.config.get("caching_teo_num_workers", None),
                step=1,
                interactive=True,
                info="Default is CPU count - 1"
            )

        with gr.Row():
            self.caching_teo_skip_existing = gr.Checkbox(
                label="Skip Existing",
                value=self.config.get("caching_teo_skip_existing", False),
                interactive=True,
                info="Skip existing cache files"
            )
            self.caching_teo_keep_cache = gr.Checkbox(
                label="Keep Cache",
                value=self.config.get("caching_teo_keep_cache", False),
                interactive=True,
                info="Keep cache files not in dataset"
            )
