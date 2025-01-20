import gradio as gr
import toml
from .class_gui_config import GUIConfig

class SaveLoadSettings:
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
            self.output_dir = gr.Textbox(
                label="Output Directory",
                placeholder="Directory to save the trained model",
                value=self.config.get("output_dir", None),
                interactive=True,
            )
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

            self.save_last_n_epochs = gr.Number(
                label="Save Last N Epochs",
                info="Save only the last N checkpoints when saving every N epochs",
                value=self.config.get("save_last_n_epochs", None),
                step=1,
                interactive=True,
            )

            self.save_every_n_steps = gr.Number(
                label="Save Every N Steps",
                info="Save a checkpoint every N steps",
                value=self.config.get("save_every_n_steps", None),
                interactive=True,
                step=1,
            )

            self.save_last_n_steps = gr.Number(
                label="Save Last N Steps",
                info="Save checkpoints until N steps elapsed (remove older ones afterward)",
                value=self.config.get("save_last_n_steps", None),
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

            self.save_last_n_steps_state = gr.Number(
                label="Save Last N Steps State",
                info="Save states until N steps elapsed (overrides save_last_n_steps)",
                value=self.config.get("save_last_n_steps_state", None),
                step=1,
                interactive=True,
            )

            self.save_state = gr.Checkbox(
                label="Save Training State",
                value=self.config.get("save_state", False),
            )

            self.save_state_on_train_end = gr.Checkbox(
                label="Save State on Train End",
                value=self.config.get("save_state_on_train_end", False),
                interactive=True,
            )