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

        with gr.Row():
            self.additional_parameters = gr.Textbox(
                label="Additional parameters",
                placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
                value=self.config.get("additional_parameters", ""),
            )