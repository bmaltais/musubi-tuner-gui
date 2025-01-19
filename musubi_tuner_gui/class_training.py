import gradio as gr
import toml
from .class_gui_config import GUIConfig

class TrainingSettings:
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
            self.sdpa = gr.Checkbox(
                label="Use SDPA for CrossAttention",
                value=self.config.get("sdpa", False),
            )

            self.flash_attn = gr.Checkbox(
                label="FlashAttention",
                info="Use FlashAttention for CrossAttention",
                value=self.config.get("flash_attn", False),
            )

            self.sage_attn = gr.Checkbox(
                label="SageAttention",
                info="Use SageAttention for CrossAttention",
                value=self.config.get("sage_attn", False),
            )

            self.xformers = gr.Checkbox(
                label="xformers",
                info="Use xformers for CrossAttention",
                value=self.config.get("xformers", False),
            )

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

            self.max_train_epochs = gr.Number(
                label="Max Training Epochs",
                info='Overrides max_train_steps',
                value=self.config.get("max_train_epochs", None),
            )

            self.max_data_loader_n_workers = gr.Number(
                label="Max DataLoader Workers",
                info='Lower values reduce RAM usage and speed up epoch start',
                value=self.config.get("max_data_loader_n_workers", 8),
                interactive=True,
            )

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

            self.gradient_checkpointing = gr.Checkbox(
                label="Enable Gradient Checkpointing",
                info="Enable gradient checkpointing for memory savings",
                value=self.config.get("gradient_checkpointing", False),
            )

            self.gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                info="Number of steps to accumulate gradients before backward pass",
                value=self.config.get("gradient_accumulation_steps", 1),
                interactive=True,
            )

        with gr.Row():
            self.logging_dir = gr.Textbox(
                label="Logging Directory",
                placeholder="Directory for TensorBoard logs",
                value=self.config.get("logging_dir", ""),
            )

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

            self.wandb_api_key = gr.Textbox(
                label="WandB API Key",
                placeholder="Optional: Specify WandB API key to log in before training",
                value=self.config.get("wandb_api_key", ""),
            )

        with gr.Row():
            self.log_tracker_config = gr.Textbox(
                label="Log Tracker Config",
                placeholder="Path to the tracker config file for logging",
                value=self.config.get("log_tracker_config", ""),
            )

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

            self.ddp_gradient_as_bucket_view = gr.Checkbox(
                label="Enable Gradient as Bucket View for DDP",
                value=self.config.get("ddp_gradient_as_bucket_view", False),
            )

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

            self.sample_at_first = gr.Checkbox(
                label="Sample Before Training",
                value=self.config.get("sample_at_first", False),
            )

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