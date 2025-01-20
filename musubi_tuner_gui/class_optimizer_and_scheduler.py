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


class OptimizerAndScheduler:
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
        self.config = config

        with gr.Row():
            self.learning_rate = gr.Number(
                label="Learning Rate",
                info="Specify the learning rate (e.g., 2.0e-6)",
                value=self.config.get("learning_rate", 2.0e-6),
                interactive=True,
                step=1e-6,
            )
            
            self.optimizer_type = gr.Dropdown(
                label="Optimizer Type",
                info="Select the optimizer to use",
                choices=["AdamW", "AdamW8bit", "AdaFactor"],
                allow_custom_value=True,
                value=self.config.get("optimizer_type", "AdamW"),
                interactive=True,
            )

            self.optimizer_args = gr.Textbox(
                label="Optimizer Arguments",
                placeholder='Additional arguments for optimizer (e.g., "weight_decay=0.01 betas=0.9,0.999")',
                value=self.config.get("optimizer_args", ""),
            )

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

            self.lr_warmup_steps = gr.Number(
                label="LR Warmup Steps",
                info="Number of warmup steps or ratio of train steps (e.g., 0.1 for 10%)",
                value=self.config.get("lr_warmup_steps", 0),
                interactive=True,
                step=0.01,
                maximum=1,
            )

            self.lr_decay_steps = gr.Number(
                label="LR Decay Steps",
                info="Number of decay steps or ratio of train steps (e.g., 0.1 for 10%)",
                value=self.config.get("lr_decay_steps", 0),
                interactive=True,
                step=0.01,
                maximum=1,
            )

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

            self.lr_scheduler_timescale = gr.Number(
                label="LR Scheduler Timescale",
                info="Timescale for inverse sqrt scheduler (defaults to num_warmup_steps)",
                value=self.config.get("lr_scheduler_timescale", None),
                step=1,
                interactive=True,
            )

            self.lr_scheduler_min_lr_ratio = gr.Number(
                label="LR Scheduler Min LR Ratio",
                info="Minimum LR as a ratio of initial LR for cosine with min LR scheduler",
                value=self.config.get("lr_scheduler_min_lr_ratio", None),
                step=0.001,
                interactive=True,
            )

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