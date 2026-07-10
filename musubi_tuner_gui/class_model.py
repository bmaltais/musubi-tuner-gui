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

        self.group_dit_vae = gr.Group(visible=True)
        with self.group_dit_vae:
            self._initialize_dit_vae_fields()

        self.group_hv_extras = gr.Group(visible=True)
        with self.group_hv_extras:
            self._initialize_hv_extras_fields()

        self.group_dual_text_encoder = gr.Group(visible=True)
        with self.group_dual_text_encoder:
            self._initialize_dual_text_encoder_fields()

        self.group_fp8_common = gr.Group(visible=True)
        with self.group_fp8_common:
            self._initialize_fp8_common_fields()

        self.group_wan_extras = gr.Group(visible=True)
        with self.group_wan_extras:
            self._initialize_wan_extras_fields()

        self.group_single_text_encoder = gr.Group(visible=True)
        with self.group_single_text_encoder:
            self._initialize_single_text_encoder_fields()

        self.group_model_version = gr.Group(visible=True)
        with self.group_model_version:
            self._initialize_model_version_fields()

        self.group_qwen_image_extras = gr.Group(visible=True)
        with self.group_qwen_image_extras:
            self._initialize_qwen_image_extras_fields()

        self.group_flux_2_extras = gr.Group(visible=True)
        with self.group_flux_2_extras:
            self._initialize_flux_2_extras_fields()

        self.group_perf = gr.Group(visible=True)
        with self.group_perf:
            self._initialize_perf_fields()

        self.group_flow_matching = gr.Group(visible=True)
        with self.group_flow_matching:
            self._initialize_flow_matching_fields()

    def _initialize_dit_vae_fields(self) -> None:
        """Fields shared by every architecture that follows the DiT+VAE shape."""
        with gr.Row():
            self.dit = gr.Textbox(
                label="DiT Checkpoint Path",
                placeholder="Path to DiT checkpoint",
                value=self.config.get("dit", ""),
            )

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

    def _initialize_hv_extras_fields(self) -> None:
        """HunyuanVideo-only model fields (DiT dtype, VAE tiling, text encoder dtype/fp8)."""
        with gr.Row():
            self.dit_dtype = gr.Dropdown(
                label="DiT Data Type",
                info="Select the data type for DiT",
                choices=["float16", "bfloat16"],
                value=self.config.get("dit_dtype", "bfloat16"),
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
            self.text_encoder_dtype = gr.Dropdown(
                label="Text Encoder Data Type",
                info="Select the data type for Text Encoder",
                choices=["float16", "bfloat16"],
                value=self.config.get("text_encoder_dtype", "float16"),
                interactive=True,
            )

            self.fp8_llm = gr.Checkbox(
                label="Use FP8 for LLM",
                value=self.config.get("fp8_llm", False),
            )

    def _initialize_dual_text_encoder_fields(self) -> None:
        """Shared by architectures with two text encoder paths (HunyuanVideo, FLUX Kontext)."""
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

    def _initialize_fp8_common_fields(self) -> None:
        """fp8_base and fp8_scaled are supported by every architecture seen so
        far; fp8_t5 is shared by Wan and FLUX Kontext specifically."""
        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="Use FP8 for Base Model",
                value=self.config.get("fp8_base", False),
            )

            self.fp8_scaled = gr.Checkbox(
                label="Use scaled FP8 for DiT",
                value=self.config.get("fp8_scaled", False),
            )

            self.fp8_t5 = gr.Checkbox(
                label="Use FP8 for T5",
                value=self.config.get("fp8_t5", False),
            )

    def _initialize_wan_extras_fields(self) -> None:
        """Wan 2.1/2.2-only model fields (task selector, T5/CLIP, dual DiT)."""
        with gr.Row():
            self.task = gr.Dropdown(
                label="Wan Task",
                info="The Wan task to run",
                choices=[
                    "t2v-14B",
                    "t2v-1.3B",
                    "i2v-14B",
                    "t2i-14B",
                    "flf2v-14B",
                    "t2v-1.3B-FC",
                    "t2v-14B-FC",
                    "i2v-14B-FC",
                    "i2v-A14B",
                    "t2v-A14B",
                ],
                value=self.config.get("task", "t2v-14B"),
                interactive=True,
            )

            self.dit_high_noise = gr.Textbox(
                label="DiT High Noise Checkpoint Path (Wan2.2)",
                placeholder="Path to the high-noise DiT checkpoint (Wan2.2 only)",
                value=self.config.get("dit_high_noise", ""),
            )

            self.timestep_boundary = gr.Number(
                label="Timestep Boundary",
                info="Timestep boundary for switching between high and low noise models (Wan2.2)",
                value=self.config.get("timestep_boundary", None),
                interactive=True,
            )

        with gr.Row():
            self.t5 = gr.Textbox(
                label="T5 Checkpoint Path",
                placeholder="Path to the T5 text encoder checkpoint",
                value=self.config.get("t5", ""),
            )

            self.clip = gr.Textbox(
                label="CLIP Checkpoint Path (Wan2.1 I2V only)",
                placeholder="Path to the CLIP text encoder checkpoint, required for Wan2.1 I2V",
                value=self.config.get("clip", ""),
            )

        with gr.Row():
            self.vae_cache_cpu = gr.Checkbox(
                label="Cache VAE features on CPU",
                value=self.config.get("vae_cache_cpu", False),
            )

    def _initialize_single_text_encoder_fields(self) -> None:
        """Shared by every architecture with exactly one text encoder path
        (Qwen-Image, Z-Image, FLUX.2, and likely most remaining image archs)."""
        with gr.Row():
            self.text_encoder = gr.Textbox(
                label="Text Encoder Path",
                placeholder="Path to the text encoder checkpoint",
                value=self.config.get("text_encoder", ""),
            )

    def _initialize_model_version_fields(self) -> None:
        """Shared by architectures with a model-version selector (Qwen-Image, FLUX.2)."""
        with gr.Row():
            self.model_version = gr.Dropdown(
                label="Model Version",
                info="Model variant to train",
                choices=["original", "layered", "edit", "edit-2509"],
                value=self.config.get("model_version", "original"),
                interactive=True,
                allow_custom_value=True,
            )

    def _initialize_qwen_image_extras_fields(self) -> None:
        """Qwen-Image-only model fields (VL fp8, layered mode)."""
        with gr.Row():
            self.fp8_vl = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                value=self.config.get("fp8_vl", False),
            )

            self.num_layers = gr.Number(
                label="Number of DiT Layers",
                info="Default is None (60)",
                value=self.config.get("num_layers", None),
                step=1,
                interactive=True,
            )

            self.remove_first_image_from_target = gr.Checkbox(
                label="Remove First Image From Target (layered model)",
                value=self.config.get("remove_first_image_from_target", False),
            )

    def _initialize_flux_2_extras_fields(self) -> None:
        """FLUX.2-only model fields."""
        with gr.Row():
            self.fp8_text_encoder = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                value=self.config.get("fp8_text_encoder", False),
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
