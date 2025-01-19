import gradio as gr
from typing import Tuple
from .common_gui import (
    get_folder_path,
    get_any_file_path,
    list_files,
    list_dirs,
    create_refresh_button,
    document_symbol,
)


class AdvancedTraining:
    """
    This class configures and initializes the advanced training settings for a machine learning model,
    including options for headless operation, fine-tuning, training type selection, and default directory paths.

    Attributes:
        headless (bool): If True, run without the Gradio interface.
        finetuning (bool): If True, enables fine-tuning of the model.
        training_type (str): Specifies the type of training to perform.
        no_token_padding (gr.Checkbox): Checkbox to disable token padding.
        gradient_accumulation_steps (gr.Slider): Slider to set the number of gradient accumulation steps.
        weighted_captions (gr.Checkbox): Checkbox to enable weighted captions.
    """

    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
    ) -> None:
        """
        Initializes the AdvancedTraining class with given settings.

        Parameters:
            headless (bool): Run in headless mode without GUI.
            finetuning (bool): Enable model fine-tuning.
            training_type (str): The type of training to be performed.
            config (dict): Configuration options for the training process.
        """
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config

        # Determine the current directories for VAE and output, falling back to defaults if not specified.
        self.current_vae_dir = self.config.get("advanced.vae_dir", "./models/vae")
        self.current_state_dir = self.config.get("advanced.state_dir", "./outputs")
        self.current_log_tracker_config_dir = self.config.get(
            "advanced.log_tracker_config_dir", "./logs"
        )

        # # Define the behavior for changing noise offset type.
        # def noise_offset_type_change(
        #     noise_offset_type: str,
        # ) -> Tuple[gr.Group, gr.Group]:
        #     """
        #     Returns a tuple of Gradio Groups with visibility set based on the noise offset type.

        #     Parameters:
        #         noise_offset_type (str): The selected noise offset type.

        #     Returns:
        #         Tuple[gr.Group, gr.Group]: A tuple containing two Gradio Group elements with their visibility set.
        #     """
        #     if noise_offset_type == "Original":
        #         return (gr.Group(visible=True), gr.Group(visible=False))
        #     else:
        #         return (gr.Group(visible=False), gr.Group(visible=True))

        # # GUI elements are only visible when not fine-tuning.
        # with gr.Row(visible=not finetuning):
        #     # Exclude token padding option for LoRA training type.
        #     if training_type != "lora":
        #         self.no_token_padding = gr.Checkbox(
        #             label="No token padding",
        #             value=self.config.get("advanced.no_token_padding", False),
        #         )
        #     self.gradient_accumulation_steps = gr.Slider(
        #         label="Gradient accumulate steps",
        #         info="Number of updates steps to accumulate before performing a backward/update pass",
        #         value=self.config.get("advanced.gradient_accumulation_steps", 1),
        #         minimum=1,
        #         maximum=120,
        #         step=1,
        #     )
        #     self.weighted_captions = gr.Checkbox(
        #         label="Weighted captions",
        #         value=self.config.get("advanced.weighted_captions", False),
        #     )
        # with gr.Group(), gr.Row(visible=not finetuning):
        #     self.prior_loss_weight = gr.Number(
        #         label="Prior loss weight",
        #         value=self.config.get("advanced.prior_loss_weight", 1.0),
        #     )

        #     def list_vae_files(path):
        #         self.current_vae_dir = path if not path == "" else "."
        #         return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

        #     self.vae = gr.Dropdown(
        #         label="VAE (Optional: Path to checkpoint of vae for training)",
        #         interactive=True,
        #         choices=[self.config.get("advanced.vae_dir", "")]
        #         + list_vae_files(self.current_vae_dir),
        #         value=self.config.get("advanced.vae_dir", ""),
        #         allow_custom_value=True,
        #     )
        #     create_refresh_button(
        #         self.vae,
        #         lambda: None,
        #         lambda: {
        #             "choices": [self.config.get("advanced.vae_dir", "")]
        #             + list_vae_files(self.current_vae_dir)
        #         },
        #         "open_folder_small",
        #     )
        #     self.vae_button = gr.Button(
        #         "📂", elem_id="open_folder_small", visible=(not headless)
        #     )
        #     self.vae_button.click(
        #         get_any_file_path,
        #         outputs=self.vae,
        #         show_progress=False,
        #     )

        #     self.vae.change(
        #         fn=lambda path: gr.Dropdown(
        #             choices=[self.config.get("advanced.vae_dir", "")]
        #             + list_vae_files(path)
        #         ),
        #         inputs=self.vae,
        #         outputs=self.vae,
        #         show_progress=False,
        #     )

        with gr.Row():
            self.additional_parameters = gr.Textbox(
                label="Additional parameters",
                placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
                value=self.config.get("additional_parameters", ""),
            )
            
        with gr.Row():
            self.dataset_config = gr.Textbox(
                label="Dataset Config",
                placeholder='Path to the dataset config file',
                value=str(self.config.get("dataset_config", "")),
            )

        with gr.Row():
            self.sdpa = gr.Checkbox(
                label="Use SDPA for CrossAttention",
                value=self.config.get("sdpa", False),
            )

        with gr.Row():
            self.flash_attn = gr.Checkbox(
                label="FlashAttention",
                info="Use FlashAttention for CrossAttention",
                value=self.config.get("flash_attn", False),
            )

        with gr.Row():
            self.sage_attn = gr.Checkbox(
                label="SageAttention",
                info="Use SageAttention for CrossAttention",
                value=self.config.get("sage_attn", False),
            )

        with gr.Row():
            self.xformers = gr.Checkbox(
                label="xformers",
                info="Use xformers for CrossAttention",
                value=self.config.get("xformers", False),
            )

        with gr.Row():
            self.split_attn = gr.Checkbox(
                label="Split Attention",
                info="Use Split Attention for CrossAttention",
                value=self.config.get("split_attn", False),
            )

        with gr.Row():
            self.max_train_steps = gr.Number(
                label="Max Training Steps",
                info="Maximum number of training steps",
                value=self.config.get("max_train_steps", 1600),
                interactive=True,
                
            )

        with gr.Row():
            self.max_train_epochs = gr.Number(
                label="Max Training Epochs",
                info='Overrides max_train_steps',
                value=self.config.get("max_train_epochs", None),
            )

        with gr.Row():
            self.max_data_loader_n_workers = gr.Number(
                label="Max DataLoader Workers",
                info='Lower values reduce RAM usage and speed up epoch start',
                value=self.config.get("max_data_loader_n_workers", 8),
                interactive=True,
            )

        with gr.Row():
            self.persistent_data_loader_workers = gr.Checkbox(
                label="Persistent DataLoader Workers",
                info='Keep DataLoader workers alive between epochs',
                value=self.config.get("persistent_data_loader_workers", False),
            )

        with gr.Row():
            self.seed = gr.Number(
                label="Random Seed for Training",
                info="Optional: set a fixed seed for reproducibility",
                value=self.config.get("seed", None),
            )

        with gr.Row():
            self.gradient_checkpointing = gr.Checkbox(
                label="Enable Gradient Checkpointing",
                info="Enable gradient checkpointing for memory savings",
                value=self.config.get("gradient_checkpointing", False),
            )

        with gr.Row():
            self.gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                info="Number of steps to accumulate gradients before backward pass",
                value=self.config.get("gradient_accumulation_steps", 1),
                interactive=True,
            )

        # Already set via accelerate
        # with gr.Row():
        #     self.mixed_precision = gr.Dropdown(
        #         label="Mixed Precision",
        #         choices=["no", "fp16", "bf16"],
        #         value=self.config.get("mixed_precision", "no"),
        #         interactive=True,
        #     )

        with gr.Row():
            self.logging_dir = gr.Textbox(
                label="Logging Directory",
                placeholder="Directory for TensorBoard logs",
                value=self.config.get("logging_dir", ""),
            )

        with gr.Row():
            self.log_with = gr.Dropdown(
                label="Logging Tool",
                info="Select the logging tool to use",
                choices=["tensorboard", "wandb", "all"],
                allow_custom_value=True,
                value=self.config.get("log_with", ""),
                interactive=True,
            )

        with gr.Row():
            self.log_prefix = gr.Textbox(
                label="Log Directory Prefix",
                placeholder="Prefix for each log directory",
                value=self.config.get("log_prefix", ""),
            )

        with gr.Row():
            self.log_tracker_name = gr.Textbox(
                label="Log Tracker Name",
                placeholder="Name of the tracker used for logging",
                value=self.config.get("log_tracker_name", ""),
            )

        with gr.Row():
            self.wandb_run_name = gr.Textbox(
                label="WandB Run Name",
                placeholder="Name of the specific WandB session",
                value=self.config.get("wandb_run_name", ""),
            )

        with gr.Row():
            self.log_tracker_config = gr.Textbox(
                label="Log Tracker Config",
                placeholder="Path to the tracker config file for logging",
                value=self.config.get("log_tracker_config", ""),
            )

        with gr.Row():
            self.wandb_api_key = gr.Textbox(
                label="WandB API Key",
                placeholder="Optional: Specify WandB API key to log in before training",
                value=self.config.get("wandb_api_key", ""),
            )

        with gr.Row():
            self.log_config = gr.Checkbox(
                label="Log Training Configuration",
                info="Log the training configuration to the logging directory",
                value=self.config.get("log_config", False),
            )

        with gr.Row():
            self.ddp_timeout = gr.Number(
                label="DDP Timeout (minutes)",
                info="Set DDP timeout in minutes (None for default)",
                value=self.config.get("ddp_timeout", None),
            )

        with gr.Row():
            self.ddp_gradient_as_bucket_view = gr.Checkbox(
                label="Enable Gradient as Bucket View for DDP",
                value=self.config.get("ddp_gradient_as_bucket_view", False),
            )

        with gr.Row():
            self.ddp_static_graph = gr.Checkbox(
                label="Enable Static Graph for DDP",
                value=self.config.get("ddp_static_graph", False),
            )

        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                info="Generate sample images every N steps",
                value=self.config.get("sample_every_n_steps", None),
            )

        with gr.Row():
            self.sample_at_first = gr.Checkbox(
                label="Sample Before Training",
                value=self.config.get("sample_at_first", False),
            )

        with gr.Row():
            self.sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs",
                info="Generate sample images every N epochs (overrides N steps)",
                value=self.config.get("sample_every_n_epochs", None),
            )

        with gr.Row():
            self.sample_prompts = gr.Textbox(
                label="Sample Prompts",
                placeholder="File containing prompts to generate sample images",
                value=self.config.get("sample_prompts", ""),
            )

        with gr.Row():
            self.optimizer_type = gr.Dropdown(
                label="Optimizer Type",
                info="Select the optimizer to use",
                choices=["AdamW", "AdamW8bit", "AdaFactor"],
                allow_custom_value=True,
                value=self.config.get("optimizer_type", "AdamW"),
                interactive=True,
            )

        with gr.Row():
            self.optimizer_args = gr.Textbox(
                label="Optimizer Arguments",
                placeholder='Additional arguments for optimizer (e.g., "weight_decay=0.01 betas=0.9,0.999")',
                value=self.config.get("optimizer_args", ""),
            )

        with gr.Row():
            self.learning_rate = gr.Number(
                label="Learning Rate",
                info="Specify the learning rate (e.g., 2.0e-6)",
                value=self.config.get("learning_rate", 2.0e-6),
                interactive=True,
                step=1e-6,
            )

        with gr.Row():
            self.max_grad_norm = gr.Number(
                label="Max Gradient Norm",
                info="Maximum gradient norm (0 for no clipping)",
                value=self.config.get("max_grad_norm", 1.0),
                interactive=True,
                step=0.0001,
            )

        with gr.Row():
            self.lr_scheduler = gr.Dropdown(
                label="Learning Rate Scheduler",
                info="Select the learning rate scheduler to use",
                choices=["constant", "linear", "cosine", "constant_with_warmup"],
                allow_custom_value=False,
                value=self.config.get("lr_scheduler", "constant"),
                interactive=True,
            )   

        with gr.Row():
            self.lr_warmup_steps = gr.Number(
                label="LR Warmup Steps",
                info="Number of warmup steps or ratio of train steps (e.g., 0.1 for 10%)",
                value=self.config.get("lr_warmup_steps", 0),
                interactive=True,
                step=0.01,
                maximum=1,
            )

        with gr.Row():
            self.lr_decay_steps = gr.Number(
                label="LR Decay Steps",
                info="Number of decay steps or ratio of train steps (e.g., 0.1 for 10%)",
                value=self.config.get("lr_decay_steps", 0),
                interactive=True,
                step=0.01,
                maximum=1,
            )

        with gr.Row():
            self.lr_scheduler_num_cycles = gr.Number(
                label="LR Scheduler Num Cycles",
                info="Number of restarts for cosine scheduler with restarts",
                value=self.config.get("lr_scheduler_num_cycles", 1),
                interactive=True,
                minimum=1,
            )

        with gr.Row():
            self.lr_scheduler_power = gr.Number(
                label="LR Scheduler Polynomial Power",
                info="Polynomial power for polynomial scheduler",
                value=self.config.get("lr_scheduler_power", 1),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_timescale = gr.Number(
                label="LR Scheduler Timescale",
                info="Timescale for inverse sqrt scheduler (defaults to num_warmup_steps)",
                value=self.config.get("lr_scheduler_timescale", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_min_lr_ratio = gr.Number(
                label="LR Scheduler Min LR Ratio",
                info="Minimum LR as a ratio of initial LR for cosine with min LR scheduler",
                value=self.config.get("lr_scheduler_min_lr_ratio", None),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_type = gr.Textbox(
                label="LR Scheduler Type",
                placeholder="Specify custom scheduler module",
                value=self.config.get("lr_scheduler_type", ""),
            )

        with gr.Row():
            self.lr_scheduler_args = gr.Textbox(
                label="LR Scheduler Arguments",
                placeholder='Additional arguments for scheduler (e.g., "T_max=100")',
                value=" ".join(self.config.get("lr_scheduler_args", []) or []),
            )

        with gr.Row():
            self.dit = gr.Textbox(
                label="DiT Checkpoint Path",
                placeholder="Path to DiT checkpoint",
                value=self.config.get("dit", ""),
            )

        with gr.Row():
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

        with gr.Row():
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

        with gr.Row():
            self.vae_chunk_size = gr.Number(
                label="VAE Chunk Size",
                info="Chunk size for CausalConv3d in VAE",
                value=self.config.get("vae_chunk_size", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
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

        with gr.Row():
            self.text_encoder2 = gr.Textbox(
                label="Text Encoder 2 Directory/file",
                placeholder="Path to Text Encoder 2 directory or file",
                value=self.config.get("text_encoder2", ""),
            )

        with gr.Row():
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

        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="Use FP8 for Base Model",
                value=self.config.get("fp8_base", False),
            )

        with gr.Row():
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap",
                info="Number of blocks to swap in the model (max XXX)",
                value=self.config.get("blocks_to_swap", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.img_in_txt_in_offloading = gr.Checkbox(
                label="Offload img_in and txt_in to CPU",
                value=self.config.get("img_in_txt_in_offloading", False),
            )

        with gr.Row():
            self.guidance_scale = gr.Number(
                label="Guidance Scale",
                info="Embedded classifier-free guidance scale",
                value=self.config.get("guidance_scale", 1.0),
                step=0.001,
                interactive=True,
            )

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

        with gr.Row():
            self.sigmoid_scale = gr.Number(
                label="Sigmoid Scale",
                info="Scale factor for sigmoid timestep sampling",
                value=self.config.get("sigmoid_scale", 1.0),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
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

        with gr.Row():
            self.logit_std = gr.Number(
                label="Logit Std",
                info="Standard deviation for 'logit_normal' weighting scheme",
                value=self.config.get("logit_std", 1.0),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
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

        with gr.Row():
            self.max_timestep = gr.Number(
                label="Max Timestep",
                info="Maximum timestep for training (1-1000)",
                value=self.config.get("max_timestep", 1000),
                minimum=1,
                maximum=1000,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
                choices=["image", "console"],
                allow_custom_value=True,
                value=self.config.get("show_timesteps", None),
                interactive=True,
            )

        with gr.Row():
            self.no_metadata = gr.Checkbox(
                label="Do Not Save Metadata",
                value=self.config.get("no_metadata", False),
            )

        with gr.Row():
            self.network_weights = gr.Textbox(
                label="Network Weights",
                placeholder="Path to pretrained weights for network",
                value=self.config.get("network_weights", None),
            )

        with gr.Row():
            self.network_module = gr.Textbox(
                label="Network Module",
                placeholder="Module of the network to train",
                value=self.config.get("network_module", None),
            )

        with gr.Row():
            self.network_dim = gr.Number(
                label="Network Dimensions",
                info="Specify dimensions for the network (depends on the module)",
                value=self.config.get("network_dim", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.network_alpha = gr.Number(
                label="Network Alpha",
                info="Alpha value for LoRA weight scaling (default: 1)",
                value=self.config.get("network_alpha", 1),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.network_dropout = gr.Number(
                label="Network Dropout",
                info="Dropout rate (0 or None for no dropout, 1 drops all neurons)",
                value=self.config.get("network_dropout", 0),
                step=0.01,
                minimum=0,
                maximum=1,
                interactive=True,
            )

        with gr.Row():
            self.network_args = gr.Textbox(
                label="Network Arguments",
                placeholder="Additional network arguments (key=value)",
                value=self.config.get("network_args", ""),
                interactive=True,
            )

        with gr.Row():
            self.training_comment = gr.Textbox(
                label="Training Comment",
                placeholder="Arbitrary comment string to store in metadata",
                value=self.config.get("training_comment", None),
            )

        with gr.Row():
            self.dim_from_weights = gr.Checkbox(
                label="Determine Dimensions from Network Weights",
                value=self.config.get("dim_from_weights", False),
            )

        with gr.Row():
            self.scale_weight_norms = gr.Number(
                label="Scale Weight Norms",
                info="Scaling factor for weights (1 is a good starting point)",
                value=self.config.get("scale_weight_norms", None),
                step=0.001,
                interactive=True,
                minimum=0,
            )

        with gr.Row():
            self.base_weights = gr.Textbox(
                label="Base Weights",
                placeholder="Paths to network weights to merge into the model before training",
                value=self.config.get("base_weights", ""),
            )

        with gr.Row():
            self.base_weights_multiplier = gr.Textbox(
                label="Base Weights Multiplier",
                placeholder="Multipliers for network weights to merge into the model before training",
                value=self.config.get("base_weights_multiplier", ""),
            )

        with gr.Row():
            self.output_dir = gr.Textbox(
                label="Output Directory",
                placeholder="Directory to save the trained model",
                value=self.config.get("output_dir", None),
                interactive=True,
            )

        with gr.Row():
            self.output_name = gr.Textbox(
                label="Output Name",
                placeholder="Base name of the trained model file (excluding extension)",
                value=self.config.get("output_name", "lora"),
                interactive=True,
            )

        with gr.Row():
            self.resume = gr.Textbox(
                label="Resume Training State",
                placeholder="Path to saved state to resume training",
                value=self.config.get("resume", None),
                interactive=True,
            )

        with gr.Row():
            self.save_every_n_epochs = gr.Number(
                label="Save Every N Epochs",
                info="Save a checkpoint every N epochs",
                value=self.config.get("save_every_n_epochs", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_every_n_steps = gr.Number(
                label="Save Every N Steps",
                info="Save a checkpoint every N steps",
                value=self.config.get("save_every_n_steps", None),
                interactive=True,
                step=1,
            )

        with gr.Row():
            self.save_last_n_epochs = gr.Number(
                label="Save Last N Epochs",
                info="Save only the last N checkpoints when saving every N epochs",
                value=self.config.get("save_last_n_epochs", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_epochs_state = gr.Number(
                label="Save Last N Epochs State",
                info="Save states of the last N epochs (overrides save_last_n_epochs)",
                value=self.config.get("save_last_n_epochs_state", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_steps = gr.Number(
                label="Save Last N Steps",
                info="Save checkpoints until N steps elapsed (remove older ones afterward)",
                value=self.config.get("save_last_n_steps", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_steps_state = gr.Number(
                label="Save Last N Steps State",
                info="Save states until N steps elapsed (overrides save_last_n_steps)",
                value=self.config.get("save_last_n_steps_state", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_state = gr.Checkbox(
                label="Save Training State",
                value=self.config.get("save_state", False),
            )

        with gr.Row():
            self.save_state_on_train_end = gr.Checkbox(
                label="Save State on Train End",
                value=self.config.get("save_state_on_train_end", False),
                interactive=True,
            )

        # with gr.Accordion("Scheduled Huber Loss", open=False):
        #     with gr.Row():
        #         self.loss_type = gr.Dropdown(
        #             label="Loss type",
        #             choices=["huber", "smooth_l1", "l1", "l2"],
        #             value=self.config.get("advanced.loss_type", "l2"),
        #             info="The type of loss to use and whether it's scheduled based on the timestep",
        #         )
        #         self.huber_schedule = gr.Dropdown(
        #             label="Huber schedule",
        #             choices=[
        #                 "constant",
        #                 "exponential",
        #                 "snr",
        #             ],
        #             value=self.config.get("advanced.huber_schedule", "snr"),
        #             info="The type of loss to use and whether it's scheduled based on the timestep",
        #         )
        #         self.huber_c = gr.Number(
        #             label="Huber C",
        #             value=self.config.get("advanced.huber_c", 0.1),
        #             minimum=0.0,
        #             maximum=1.0,
        #             step=0.01,
        #             info="The huber loss parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type",
        #         )
        #         self.huber_scale = gr.Number(
        #             label="Huber scale",
        #             value=self.config.get("advanced.huber_scale", 1.0),
        #             minimum=0.0,
        #             maximum=10.0,
        #             step=0.01,
        #             info="The Huber loss scale parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type.",
        #         )

        # with gr.Row():
        #     self.save_every_n_steps = gr.Number(
        #         label="Save every N steps",
        #         value=self.config.get("advanced.save_every_n_steps", 0),
        #         precision=0,
        #         info="(Optional) The model is saved every specified steps",
        #     )
        #     self.save_last_n_steps = gr.Number(
        #         label="Save last N steps",
        #         value=self.config.get("advanced.save_last_n_steps", 0),
        #         precision=0,
        #         info="(Optional) Save only the specified number of models (old models will be deleted)",
        #     )
        #     self.save_last_n_steps_state = gr.Number(
        #         label="Save last N steps state",
        #         value=self.config.get("advanced.save_last_n_steps_state", 0),
        #         precision=0,
        #         info="(Optional) Save only the specified number of states (old models will be deleted)",
        #     )
        #     self.save_last_n_epochs = gr.Number(
        #         label="Save last N epochs",
        #         value=self.config.get("advanced.save_last_n_epochs", 0),
        #         precision=0,
        #         info="(Optional) Save only the specified number of epochs (old epochs will be deleted)",
        #     )
        #     self.save_last_n_epochs_state = gr.Number(
        #         label="Save last N epochs state",
        #         value=self.config.get("advanced.save_last_n_epochs_state", 0),
        #         precision=0,
        #         info="(Optional) Save only the specified number of epochs states (old models will be deleted)",
        #     )
        # with gr.Row():

        #     def full_options_update(full_fp16, full_bf16):
        #         full_fp16_active = True
        #         full_bf16_active = True

        #         if full_fp16:
        #             full_bf16_active = False
        #         if full_bf16:
        #             full_fp16_active = False
        #         return gr.Checkbox(
        #             interactive=full_fp16_active,
        #         ), gr.Checkbox(interactive=full_bf16_active)

        #     self.keep_tokens = gr.Slider(
        #         label="Keep n tokens",
        #         value=self.config.get("advanced.keep_tokens", 0),
        #         minimum=0,
        #         maximum=32,
        #         step=1,
        #     )
        #     self.clip_skip = gr.Slider(
        #         label="Clip skip",
        #         value=self.config.get("advanced.clip_skip", 1),
        #         minimum=0,
        #         maximum=12,
        #         step=1,
        #     )
        #     self.max_token_length = gr.Dropdown(
        #         label="Max Token Length",
        #         choices=[
        #             75,
        #             150,
        #             225,
        #         ],
        #         info="max token length of text encoder",
        #         value=self.config.get("advanced.max_token_length", 75),
        #     )

        # with gr.Row():
        #     self.fp8_base = gr.Checkbox(
        #         label="fp8 base",
        #         info="Use fp8 for base model",
        #         value=self.config.get("advanced.fp8_base", False),
        #     )
        #     self.fp8_base_unet  = gr.Checkbox(
        #         label="fp8 base unet",
        #         info="Flux can be trained with fp8, and CLIP-L can be trained with bf16/fp16.",
        #         value=self.config.get("advanced.fp8_base_unet", False),
        #     )
        #     self.full_fp16 = gr.Checkbox(
        #         label="Full fp16 training (experimental)",
        #         value=self.config.get("advanced.full_fp16", False),
        #     )
        #     self.full_bf16 = gr.Checkbox(
        #         label="Full bf16 training (experimental)",
        #         value=self.config.get("advanced.full_bf16", False),
        #         info="Required bitsandbytes >= 0.36.0",
        #     )

        #     self.full_fp16.change(
        #         full_options_update,
        #         inputs=[self.full_fp16, self.full_bf16],
        #         outputs=[self.full_fp16, self.full_bf16],
        #     )
        #     self.full_bf16.change(
        #         full_options_update,
        #         inputs=[self.full_fp16, self.full_bf16],
        #         outputs=[self.full_fp16, self.full_bf16],
        #     )
            
        # with gr.Row():
        #     self.highvram = gr.Checkbox(
        #         label="highvram",
        #         value=self.config.get("advanced.highvram", False),
        #         info="Disable low VRAM optimization. e.g. do not clear CUDA cache after each latent caching (for machines which have bigger VRAM)",
        #         interactive=True,
        #     )
        #     self.lowvram = gr.Checkbox(
        #         label="lowvram",
        #         value=self.config.get("advanced.lowvram", False),
        #         info="Enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)",
        #         interactive=True,
        #     )
        #     self.skip_cache_check = gr.Checkbox(
        #         label="Skip cache check",
        #         value=self.config.get("advanced.skip_cache_check", False),
        #         info="Skip cache check for faster training start",
        #     )

        # with gr.Row():
        #     self.gradient_checkpointing = gr.Checkbox(
        #         label="Gradient checkpointing",
        #         value=self.config.get("advanced.gradient_checkpointing", False),
        #     )
        #     self.shuffle_caption = gr.Checkbox(
        #         label="Shuffle caption",
        #         value=self.config.get("advanced.shuffle_caption", False),
        #     )
        #     self.persistent_data_loader_workers = gr.Checkbox(
        #         label="Persistent data loader",
        #         value=self.config.get("advanced.persistent_data_loader_workers", False),
        #     )
        #     self.mem_eff_attn = gr.Checkbox(
        #         label="Memory efficient attention",
        #         value=self.config.get("advanced.mem_eff_attn", False),
        #     )
        # with gr.Row():
        #     self.xformers = gr.Dropdown(
        #         label="CrossAttention",
        #         choices=["none", "sdpa", "xformers"],
        #         value=self.config.get("advanced.xformers", "xformers"),
        #     )
        #     self.color_aug = gr.Checkbox(
        #         label="Color augmentation",
        #         value=self.config.get("advanced.color_aug", False),
        #         info="Enable weak color augmentation",
        #     )
        #     self.flip_aug = gr.Checkbox(
        #         label="Flip augmentation",
        #         value=getattr(self.config, "advanced.flip_aug", False),
        #         info="Enable horizontal flip augmentation",
        #     )
        #     self.masked_loss = gr.Checkbox(
        #         label="Masked loss",
        #         value=self.config.get("advanced.masked_loss", False),
        #         info="Apply mask for calculating loss. conditioning_data_dir is required for dataset",
        #     )
        # with gr.Row():
        #     self.scale_v_pred_loss_like_noise_pred = gr.Checkbox(
        #         label="Scale v prediction loss",
        #         value=self.config.get(
        #             "advanced.scale_v_pred_loss_like_noise_pred", False
        #         ),
        #         info="Only for SD v2 models. By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.",
        #     )
        #     self.min_snr_gamma = gr.Slider(
        #         label="Min SNR gamma",
        #         value=self.config.get("advanced.min_snr_gamma", 0),
        #         minimum=0,
        #         maximum=20,
        #         step=1,
        #         info="Recommended value of 5 when used",
        #     )
        #     self.debiased_estimation_loss = gr.Checkbox(
        #         label="Debiased Estimation loss",
        #         value=self.config.get("advanced.debiased_estimation_loss", False),
        #         info="Automates the processing of noise, allowing for faster model fitting, as well as balancing out color issues. Do not use if Min SNR gamma is specified.",
        #     )
        # with gr.Row():
        #     # self.sdpa = gr.Checkbox(label='Use sdpa', value=False, info='Use sdpa for CrossAttention')
        #     self.bucket_no_upscale = gr.Checkbox(
        #         label="Don't upscale bucket resolution",
        #         value=self.config.get("advanced.bucket_no_upscale", True),
        #     )
        #     self.bucket_reso_steps = gr.Slider(
        #         label="Bucket resolution steps",
        #         value=self.config.get("advanced.bucket_reso_steps", 64),
        #         minimum=1,
        #         maximum=128,
        #     )
        #     self.random_crop = gr.Checkbox(
        #         label="Random crop instead of center crop",
        #         value=self.config.get("advanced.random_crop", False),
        #     )
        #     self.v_pred_like_loss = gr.Slider(
        #         label="V Pred like loss",
        #         value=self.config.get("advanced.v_pred_like_loss", 0),
        #         minimum=0,
        #         maximum=1,
        #         step=0.01,
        #         info="Recommended value of 0.5 when used",
        #     )

        # with gr.Row():
        #     self.min_timestep = gr.Slider(
        #         label="Min Timestep",
        #         value=self.config.get("advanced.min_timestep", 0),
        #         step=1,
        #         minimum=0,
        #         maximum=1000,
        #         info="Values greater than 0 will make the model more img2img focussed. 0 = image only",
        #     )
        #     self.max_timestep = gr.Slider(
        #         label="Max Timestep",
        #         value=self.config.get("advanced.max_timestep", 1000),
        #         step=1,
        #         minimum=0,
        #         maximum=1000,
        #         info="Values lower than 1000 will make the model more img2img focussed. 1000 = noise only",
        #     )

        # with gr.Row():
        #     self.noise_offset_type = gr.Dropdown(
        #         label="Noise offset type",
        #         choices=[
        #             "Original",
        #             "Multires",
        #         ],
        #         value=self.config.get("advanced.noise_offset_type", "Original"),
        #         scale=1,
        #     )
        #     with gr.Row(visible=True) as self.noise_offset_original:
        #         self.noise_offset = gr.Slider(
        #             label="Noise offset",
        #             value=self.config.get("advanced.noise_offset", 0),
        #             minimum=0,
        #             maximum=1,
        #             step=0.01,
        #             info="Recommended values are 0.05 - 0.15",
        #         )
        #         self.noise_offset_random_strength = gr.Checkbox(
        #             label="Noise offset random strength",
        #             value=self.config.get(
        #                 "advanced.noise_offset_random_strength", False
        #             ),
        #             info="Use random strength between 0~noise_offset for noise offset",
        #         )
        #         self.adaptive_noise_scale = gr.Slider(
        #             label="Adaptive noise scale",
        #             value=self.config.get("advanced.adaptive_noise_scale", 0),
        #             minimum=-1,
        #             maximum=1,
        #             step=0.001,
        #             info="Add `latent mean absolute value * this value` to noise_offset",
        #         )
        #     with gr.Row(visible=False) as self.noise_offset_multires:
        #         self.multires_noise_iterations = gr.Slider(
        #             label="Multires noise iterations",
        #             value=self.config.get("advanced.multires_noise_iterations", 0),
        #             minimum=0,
        #             maximum=64,
        #             step=1,
        #             info="Enable multires noise (recommended values are 6-10)",
        #         )
        #         self.multires_noise_discount = gr.Slider(
        #             label="Multires noise discount",
        #             value=self.config.get("advanced.multires_noise_discount", 0.3),
        #             minimum=0,
        #             maximum=1,
        #             step=0.01,
        #             info="Recommended values are 0.8. For LoRAs with small datasets, 0.1-0.3",
        #         )
        #     with gr.Row(visible=True):
        #         self.ip_noise_gamma = gr.Slider(
        #             label="IP noise gamma",
        #             value=self.config.get("advanced.ip_noise_gamma", 0),
        #             minimum=0,
        #             maximum=1,
        #             step=0.01,
        #             info="enable input perturbation noise. used for regularization. recommended value: around 0.1",
        #         )
        #         self.ip_noise_gamma_random_strength = gr.Checkbox(
        #             label="IP noise gamma random strength",
        #             value=self.config.get(
        #                 "advanced.ip_noise_gamma_random_strength", False
        #             ),
        #             info="Use random strength between 0~ip_noise_gamma for input perturbation noise",
        #         )
        #     self.noise_offset_type.change(
        #         noise_offset_type_change,
        #         inputs=[self.noise_offset_type],
        #         outputs=[
        #             self.noise_offset_original,
        #             self.noise_offset_multires,
        #         ],
        #     )
        # with gr.Row():
        #     self.caption_dropout_every_n_epochs = gr.Number(
        #         label="Dropout caption every n epochs",
        #         value=self.config.get("advanced.caption_dropout_every_n_epochs", 0),
        #     )
        #     self.caption_dropout_rate = gr.Slider(
        #         label="Rate of caption dropout",
        #         value=self.config.get("advanced.caption_dropout_rate", 0),
        #         minimum=0,
        #         maximum=1,
        #     )
        #     self.vae_batch_size = gr.Slider(
        #         label="VAE batch size",
        #         minimum=0,
        #         maximum=32,
        #         value=self.config.get("advanced.vae_batch_size", 0),
        #         step=1,
        #     )
        #     self.blocks_to_swap = gr.Slider(
        #         label="Blocks to swap",
        #         value=self.config.get("advanced.blocks_to_swap", 0),
        #         info="The number of blocks to swap. The default is None (no swap). These options must be combined with --fused_backward_pass or --blockwise_fused_optimizers. The recommended maximum value is 36.",
        #         minimum=0,
        #         maximum=57,
        #         step=1,
        #         interactive=True,
        #     )
        # with gr.Group(), gr.Row():
        #     self.save_state = gr.Checkbox(
        #         label="Save training state",
        #         value=self.config.get("advanced.save_state", False),
        #         info="Save training state (including optimizer states etc.) when saving models"
        #     )

        #     self.save_state_on_train_end = gr.Checkbox(
        #         label="Save training state at end of training",
        #         value=self.config.get("advanced.save_state_on_train_end", False),
        #         info="Save training state (including optimizer states etc.) on train end"
        #     )

        #     def list_state_dirs(path):
        #         self.current_state_dir = path if not path == "" else "."
        #         return list(list_dirs(path))

        #     self.resume = gr.Dropdown(
        #         label='Resume from saved training state (path to "last-state" state folder)',
        #         choices=[self.config.get("advanced.state_dir", "")]
        #         + list_state_dirs(self.current_state_dir),
        #         value=self.config.get("advanced.state_dir", ""),
        #         interactive=True,
        #         allow_custom_value=True,
        #         info="Saved state to resume training from"
        #     )
        #     create_refresh_button(
        #         self.resume,
        #         lambda: None,
        #         lambda: {
        #             "choices": [self.config.get("advanced.state_dir", "")]
        #             + list_state_dirs(self.current_state_dir)
        #         },
        #         "open_folder_small",
        #     )
        #     self.resume_button = gr.Button(
        #         "📂", elem_id="open_folder_small", visible=(not headless)
        #     )
        #     self.resume_button.click(
        #         get_folder_path,
        #         outputs=self.resume,
        #         show_progress=False,
        #     )
        #     self.resume.change(
        #         fn=lambda path: gr.Dropdown(
        #             choices=[self.config.get("advanced.state_dir", "")]
        #             + list_state_dirs(path)
        #         ),
        #         inputs=self.resume,
        #         outputs=self.resume,
        #         show_progress=False,
        #     )
        #     self.max_data_loader_n_workers = gr.Number(
        #         label="Max num workers for DataLoader",
        #         info="Override number of epoch. Default: 0",
        #         step=1,
        #         minimum=0,
        #         value=self.config.get("advanced.max_data_loader_n_workers", 0),
        #     )
        # with gr.Row():
        #     self.log_with = gr.Dropdown(
        #         label="Logging",
        #         choices=["","wandb", "tensorboard","all"],
        #         value="",
        #         info="Loggers to use, tensorboard will be used as the default.",
        #     )
        #     self.wandb_api_key = gr.Textbox(
        #         label="WANDB API Key",
        #         value=self.config.get("advanced.wandb_api_key", ""),
        #         placeholder="(Optional)",
        #         info="Users can obtain and/or generate an api key in the their user settings on the website: https://wandb.ai/login",
        #     )
        #     self.wandb_run_name = gr.Textbox(
        #         label="WANDB run name",
        #         value=self.config.get("advanced.wandb_run_name", ""),
        #         placeholder="(Optional)",
        #         info="The name of the specific wandb session",
        #     )
        # with gr.Group(), gr.Row():

        #     def list_log_tracker_config_files(path):
        #         self.current_log_tracker_config_dir = path if not path == "" else "."
        #         return list(list_files(path, exts=[".json"], all=True))

        #     self.log_config = gr.Checkbox(
        #         label="Log config",
        #         value=self.config.get("advanced.log_config", False),
        #         info="Log training parameter to WANDB",
        #     )
        #     self.log_tracker_name = gr.Textbox(
        #         label="Log tracker name",
        #         value=self.config.get("advanced.log_tracker_name", ""),
        #         placeholder="(Optional)",
        #         info="Name of tracker to use for logging, default is script-specific default name",
        #     )
        #     self.log_tracker_config = gr.Dropdown(
        #         label="Log tracker config",
        #         choices=[self.config.get("log_tracker_config_dir", "")]
        #         + list_log_tracker_config_files(self.current_log_tracker_config_dir),
        #         value=self.config.get("log_tracker_config_dir", ""),
        #         info="Path to tracker config file to use for logging",
        #         interactive=True,
        #         allow_custom_value=True,
        #     )
        #     create_refresh_button(
        #         self.log_tracker_config,
        #         lambda: None,
        #         lambda: {
        #             "choices": [self.config.get("log_tracker_config_dir", "")]
        #             + list_log_tracker_config_files(self.current_log_tracker_config_dir)
        #         },
        #         "open_folder_small",
        #     )
        #     self.log_tracker_config_button = gr.Button(
        #         document_symbol, elem_id="open_folder_small", visible=(not headless)
        #     )
        #     self.log_tracker_config_button.click(
        #         get_any_file_path,
        #         outputs=self.log_tracker_config,
        #         show_progress=False,
        #     )
        #     self.log_tracker_config.change(
        #         fn=lambda path: gr.Dropdown(
        #             choices=[self.config.get("log_tracker_config_dir", "")]
        #             + list_log_tracker_config_files(path)
        #         ),
        #         inputs=self.log_tracker_config,
        #         outputs=self.log_tracker_config,
        #         show_progress=False,
        #     )
