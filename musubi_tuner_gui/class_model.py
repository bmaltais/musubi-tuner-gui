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

        self.group_dit_dtype = gr.Group(visible=True)
        with self.group_dit_dtype:
            self._initialize_dit_dtype_fields()

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

        self.group_fp8_vl = gr.Group(visible=True)
        with self.group_fp8_vl:
            self._initialize_fp8_vl_fields()

        self.group_qwen_image_extras = gr.Group(visible=True)
        with self.group_qwen_image_extras:
            self._initialize_qwen_image_extras_fields()

        self.group_flux_2_extras = gr.Group(visible=True)
        with self.group_flux_2_extras:
            self._initialize_flux_2_extras_fields()

        self.group_image_encoder = gr.Group(visible=True)
        with self.group_image_encoder:
            self._initialize_image_encoder_fields()

        self.group_hv_1_5_extras = gr.Group(visible=True)
        with self.group_hv_1_5_extras:
            self._initialize_hv_1_5_extras_fields()

        self.group_framepack_extras = gr.Group(visible=True)
        with self.group_framepack_extras:
            self._initialize_framepack_extras_fields()

        self.group_kandinsky5_extras = gr.Group(visible=True)
        with self.group_kandinsky5_extras:
            self._initialize_kandinsky5_extras_fields()

        self.group_hidream_o1_extras = gr.Group(visible=True)
        with self.group_hidream_o1_extras:
            self._initialize_hidream_o1_extras_fields()

        self.group_ideogram4_extras = gr.Group(visible=True)
        with self.group_ideogram4_extras:
            self._initialize_ideogram4_extras_fields()

        self.group_krea2_extras = gr.Group(visible=True)
        with self.group_krea2_extras:
            self._initialize_krea2_extras_fields()

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

    def _initialize_dit_dtype_fields(self) -> None:
        """Shared by architectures with a DiT dtype selector (HunyuanVideo, HunyuanVideo 1.5)."""
        with gr.Row():
            self.dit_dtype = gr.Dropdown(
                label="DiT Data Type",
                info="Select the data type for DiT",
                choices=["float16", "bfloat16"],
                value=self.config.get("dit_dtype", "bfloat16"),
                interactive=True,
            )

    def _initialize_hv_extras_fields(self) -> None:
        """HunyuanVideo-only model fields (VAE tiling, text encoder dtype/fp8)."""
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

    def _initialize_fp8_vl_fields(self) -> None:
        """Shared by architectures with a VL/vision-language text encoder fp8
        toggle (Qwen-Image, HunyuanVideo 1.5)."""
        with gr.Row():
            self.fp8_vl = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                value=self.config.get("fp8_vl", False),
            )

    def _initialize_qwen_image_extras_fields(self) -> None:
        """Qwen-Image-only model fields (layered mode)."""
        with gr.Row():
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

    def _initialize_image_encoder_fields(self) -> None:
        """Shared by architectures with an image encoder for i2v
        (HunyuanVideo 1.5, FramePack)."""
        with gr.Row():
            self.image_encoder = gr.Textbox(
                label="Image Encoder Path (i2v)",
                placeholder="Path to the image encoder checkpoint, required for i2v",
                value=self.config.get("image_encoder", ""),
            )

    def _initialize_hv_1_5_extras_fields(self) -> None:
        """HunyuanVideo 1.5-only model fields (t2v/i2v task, ByT5, VAE patch conv)."""
        with gr.Row():
            self.hv15_task = gr.Dropdown(
                label="Task",
                info="Text-to-video (t2v) or image-to-video (i2v)",
                choices=["t2v", "i2v"],
                value=self.config.get("hv15_task", "t2v"),
                interactive=True,
            )

            self.byt5 = gr.Textbox(
                label="ByT5 Checkpoint Path",
                placeholder="Path to the ByT5 text encoder checkpoint",
                value=self.config.get("byt5", ""),
            )

        with gr.Row():
            self.vae_enable_patch_conv = gr.Checkbox(
                label="Enable VAE Patch Conv",
                value=self.config.get("vae_enable_patch_conv", False),
            )

            self.vae_sample_size = gr.Number(
                label="VAE Sample Size",
                value=self.config.get("vae_sample_size", None),
                step=1,
                interactive=True,
            )

    def _initialize_framepack_extras_fields(self) -> None:
        """FramePack-only model fields."""
        with gr.Row():
            self.latent_window_size = gr.Number(
                label="Latent Window Size",
                value=self.config.get("latent_window_size", None),
                step=1,
                interactive=True,
            )

            self.f1 = gr.Checkbox(
                label="Use F1 Sampling",
                value=self.config.get("f1", False),
            )

            self.bulk_decode = gr.Checkbox(
                label="Bulk Decode",
                value=self.config.get("bulk_decode", False),
            )

            self.one_frame = gr.Checkbox(
                label="One Frame Training",
                value=self.config.get("one_frame", False),
            )

    def _initialize_kandinsky5_extras_fields(self) -> None:
        """Kandinsky 5-only model fields. --task is a required free-form
        string upstream (no fixed choices), unlike Wan/HunyuanVideo 1.5's
        --task enums, so it keeps its own field name rather than sharing
        theirs; more esoteric nabla-attention tuning flags (nabla_P,
        nabla_wH/wT/wW, nabla_method, etc.) are left to the existing
        Additional Parameters passthrough rather than getting dedicated
        widgets."""
        with gr.Row():
            self.kandinsky5_task = gr.Textbox(
                label="Task",
                placeholder="Required task identifier, see musubi-tuner docs",
                value=self.config.get("kandinsky5_task", ""),
            )

            self.text_encoder_clip = gr.Textbox(
                label="CLIP Text Encoder Path",
                placeholder="Path to the CLIP text encoder checkpoint",
                value=self.config.get("text_encoder_clip", ""),
            )

            self.text_encoder_qwen = gr.Textbox(
                label="Qwen Text Encoder Path",
                placeholder="Path to the Qwen text encoder checkpoint",
                value=self.config.get("text_encoder_qwen", ""),
            )

    def _initialize_hidream_o1_extras_fields(self) -> None:
        """HiDream-O1-only model fields. Has no separate text-encoder
        checkpoint path -- its text encoder is derived from --dit itself
        during caching. Only dino_loss_weight (0 disables the optional
        DINOv3 auxiliary loss) gets a dedicated widget; the remaining
        DINOv3 tuning flags (layer, feature_mode, model_type, backend,
        etc.) are left to Additional Parameters."""
        with gr.Row():
            self.hidream_task = gr.Dropdown(
                label="Task",
                info="Text-to-image (t2i) or image-to-image (i2i)",
                choices=["t2i", "i2i"],
                value=self.config.get("hidream_task", "t2i"),
                interactive=True,
            )

            self.hidream_model_type = gr.Dropdown(
                label="Model Type",
                choices=["full", "dev"],
                value=self.config.get("hidream_model_type", "full"),
                interactive=True,
            )

            self.fp8_te = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                value=self.config.get("fp8_te", False),
            )

        with gr.Row():
            self.dino_loss_weight = gr.Number(
                label="DINOv3 Auxiliary Loss Weight",
                info="0 disables the DINOv3 auxiliary loss",
                value=self.config.get("dino_loss_weight", 0),
                step=0.001,
                interactive=True,
            )

    def _initialize_ideogram4_extras_fields(self) -> None:
        """Ideogram4-only model fields. log_loss_stats (a debug diagnostics
        flag) is left to Additional Parameters rather than getting a
        dedicated widget."""
        with gr.Row():
            self.unconditional_dit = gr.Textbox(
                label="Unconditional DiT Path",
                placeholder="Path to the unconditional Ideogram 4 DiT checkpoint",
                value=self.config.get("unconditional_dit", ""),
            )

            self.sampler_preset = gr.Dropdown(
                label="Sampler Preset",
                choices=["V4_DEFAULT_20", "V4_QUALITY_48", "V4_TURBO_12"],
                value=self.config.get("sampler_preset", None),
                interactive=True,
            )

            self.initial_sigma = gr.Number(
                label="Initial Sigma",
                info="Override the first denoising sigma for sampling",
                value=self.config.get("initial_sigma", None),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
            self.use_unconditional_dit_for_lora_sampling = gr.Checkbox(
                label="Use Unconditional DiT for LoRA Sampling",
                value=self.config.get("use_unconditional_dit_for_lora_sampling", False),
            )

            self.validate_caption_structure = gr.Checkbox(
                label="Validate Caption Structure",
                value=self.config.get("validate_caption_structure", False),
            )

            self.warn_on_caption_issues = gr.Checkbox(
                label="Warn on Caption Issues",
                value=self.config.get("warn_on_caption_issues", False),
            )

    def _initialize_krea2_extras_fields(self) -> None:
        """Krea 2-only model fields."""
        with gr.Row():
            self.turbo_dit = gr.Textbox(
                label="Turbo DiT Path",
                placeholder="Distilled Turbo DiT checkpoint path (for sample generation)",
                value=self.config.get("turbo_dit", ""),
            )

            self.turbo_dit_cache = gr.Checkbox(
                label="Cache Turbo DiT",
                value=self.config.get("turbo_dit_cache", False),
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

        with gr.Row():
            self.use_pinned_memory_for_block_swap = gr.Checkbox(
                label="Use Pinned Memory for Block Swap",
                info="Speeds up block swap at the cost of extra host RAM",
                value=self.config.get("use_pinned_memory_for_block_swap", False),
            )

            self.block_swap_h2d_only = gr.Checkbox(
                label="Block Swap H2D Only",
                info="Only swap host-to-device, skipping the device-to-host copy",
                value=self.config.get("block_swap_h2d_only", False),
            )

            self.block_swap_ring_size = gr.Number(
                label="Block Swap Ring Size",
                value=self.config.get("block_swap_ring_size", None),
                step=1,
                interactive=True,
            )

    def _initialize_flow_matching_fields(self) -> None:
        with gr.Row():
            self.timestep_sampling = gr.Dropdown(
                label="Timestep Sampling Method",
                choices=["sigma", "uniform", "sigmoid", "shift"],
                value=self.config.get("timestep_sampling", "sigma"),
                interactive=True,
                allow_custom_value=True,
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
