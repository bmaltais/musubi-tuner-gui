import gradio as gr
from .class_gui_config import GUIConfig
from .class_architecture import architecture_choices, DEFAULT_ARCHITECTURE


class Model:
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
            self.architecture = gr.Dropdown(
                label="Architecture",
                info="Model architecture to train",
                choices=architecture_choices(),
                value=self.config.get("architecture", DEFAULT_ARCHITECTURE),
                interactive=True,
            )

        with gr.Row():
            self.dataset_config = gr.Textbox(
                label="Dataset Config",
                placeholder="Path to the dataset config file",
                value=str(self.config.get("dataset_config", "")),
            )

        self.group_dit_vae_te = gr.Group(visible=True)
        with self.group_dit_vae_te:
            self._initialize_dit_vae_te_fields()

        self.group_perf = gr.Group(visible=True)
        with self.group_perf:
            self._initialize_perf_fields()

        self.group_flow_matching = gr.Group(visible=True)
        with self.group_flow_matching:
            self._initialize_flow_matching_fields()

    def _initialize_dit_vae_te_fields(self) -> None:
        with gr.Row():
            self.dit = gr.Textbox(
                label="DiT Checkpoint Path",
                placeholder="Path to DiT checkpoint",
                value=self.config.get("dit", ""),
            )

            self.dit_dtype = gr.Dropdown(
                label="DiT Data Type",
                info="Select the data type for DiT",
                choices=["float16", "bfloat16"],
                value=self.config.get("dit_dtype", "bfloat16"),
                interactive=True,
            )

        with gr.Row():
            self.vae = gr.Textbox(
                label="VAE Checkpoint Path",
                placeholder="Path to VAE checkpoint",
                value=self.config.get("vae", ""),
            )
            self.vae_dtype = gr.Dropdown(
                label="VAE Data Type",
                info="Select the data type for VAE",
                choices=["float16", "bfloat16"],
                value=self.config.get("vae_dtype", "float16"),
                interactive=True,
            )

        with gr.Row():
            self.vae_tiling = gr.Checkbox(
                label="Enable VAE Spatial Tiling",
                value=self.config.get("vae_tiling", False),
                interactive=True,
            )

            self.vae_chunk_size = gr.Number(
                label="VAE Chunk Size",
                info="Chunk size for CausalConv3d in VAE",
                value=self.config.get("vae_chunk_size", None),
                step=1,
                interactive=True,
            )

            self.vae_spatial_tile_sample_min_size = gr.Number(
                label="VAE Spatial Tile Sample Min Size",
                info="Spatial tile sample min size for VAE (default: 256)",
                value=self.config.get("vae_spatial_tile_sample_min_size", 256),
                interactive=True,
            )

        with gr.Row():
            self.text_encoder1 = gr.Textbox(
                label="Text Encoder 1 Directory/file",
                placeholder="Path to Text Encoder 1 directory or file",
                value=self.config.get("text_encoder1", ""),
            )

            self.text_encoder2 = gr.Textbox(
                label="Text Encoder 2 Directory/file",
                placeholder="Path to Text Encoder 2 directory or file",
                value=self.config.get("text_encoder2", ""),
            )

            self.text_encoder_dtype = gr.Dropdown(
                label="Text Encoder Data Type",
                info="Select the data type for Text Encoder",
                choices=["float16", "bfloat16"],
                value=self.config.get("text_encoder_dtype", "float16"),
                interactive=True,
            )

        with gr.Row():
            self.fp8_llm = gr.Checkbox(
                label="Use FP8 for LLM",
                value=self.config.get("fp8_llm", False),
            )

            self.fp8_base = gr.Checkbox(
                label="Use FP8 for Base Model",
                value=self.config.get("fp8_base", False),
            )

    def _initialize_perf_fields(self) -> None:
        with gr.Row():
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap",
                info="Number of blocks to swap in the model (max XXX)",
                value=self.config.get("blocks_to_swap", None),
                step=1,
                interactive=True,
            )

            self.img_in_txt_in_offloading = gr.Checkbox(
                label="Offload img_in and txt_in to CPU",
                value=self.config.get("img_in_txt_in_offloading", False),
            )

            self.guidance_scale = gr.Number(
                label="Guidance Scale",
                info="Embedded classifier-free guidance scale",
                value=self.config.get("guidance_scale", 1.0),
                step=0.001,
                interactive=True,
            )

    def _initialize_flow_matching_fields(self) -> None:
        with gr.Row():
            self.timestep_sampling = gr.Dropdown(
                label="Timestep Sampling Method",
                choices=["sigma", "uniform", "sigmoid", "shift"],
                value=self.config.get("timestep_sampling", "sigma"),
                interactive=True,
            )

        with gr.Row():
            self.discrete_flow_shift = gr.Number(
                label="Discrete Flow Shift",
                info="Discrete flow shift for the Euler Discrete Scheduler (default: 1.0)",
                value=self.config.get("discrete_flow_shift", 1.0),
                step=0.001,
                interactive=True,
            )

            self.sigmoid_scale = gr.Number(
                label="Sigmoid Scale",
                info="Scale factor for sigmoid timestep sampling",
                value=self.config.get("sigmoid_scale", 1.0),
                step=0.001,
                interactive=True,
            )

            self.weighting_scheme = gr.Dropdown(
                label="Weighting Scheme",
                choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
                value=self.config.get("weighting_scheme", "none"),
                interactive=True,
            )

        with gr.Row():
            self.logit_mean = gr.Number(
                label="Logit Mean",
                info="Mean for 'logit_normal' weighting scheme",
                value=self.config.get("logit_mean", 0.0),
                step=0.001,
                interactive=True,
            )

            self.logit_std = gr.Number(
                label="Logit Std",
                info="Standard deviation for 'logit_normal' weighting scheme",
                value=self.config.get("logit_std", 1.0),
                step=0.001,
                interactive=True,
            )

            self.mode_scale = gr.Number(
                label="Mode Scale",
                info="Scale of mode weighting scheme",
                value=self.config.get("mode_scale", 1.29),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
            self.min_timestep = gr.Number(
                label="Min Timestep",
                info="Minimum timestep for training (0-999)",
                value=self.config.get("min_timestep", 0),
                step=1,
                minimum=0,
                maximum=999,
                interactive=True,
            )

            self.max_timestep = gr.Number(
                label="Max Timestep",
                info="Maximum timestep for training (1-1000)",
                value=self.config.get("max_timestep", 1000),
                minimum=1,
                maximum=1000,
                step=1,
                interactive=True,
            )

            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
                choices=["image", "console"],
                allow_custom_value=True,
                value=self.config.get("show_timesteps", None),
                interactive=True,
            )
