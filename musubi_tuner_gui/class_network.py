import gradio as gr
import toml
from .class_gui_config import GUIConfig

class Network:
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
            self.no_metadata = gr.Checkbox(
                label="Do Not Save Metadata",
                value=self.config.get("no_metadata", False),
            )

            self.network_weights = gr.Textbox(
                label="Network Weights",
                placeholder="Path to pretrained weights for network",
                value=self.config.get("network_weights", None),
            )

            self.network_module = gr.Textbox(
                label="Network Module",
                placeholder="Module of the network to train",
                value=self.config.get("network_module", None),
            )

        with gr.Row():
            self.network_dim = gr.Number(
                label="Network Dimensions",
                info="Specify dimensions for the network (depends on the module)",
                value=self.config.get("network_dim", 32),
                step=1,
                interactive=True,
            )

            self.network_alpha = gr.Number(
                label="Network Alpha",
                info="Alpha value for LoRA weight scaling (default: 1)",
                value=self.config.get("network_alpha", 1),
                step=1,
                interactive=True,
            )

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

            self.base_weights_multiplier = gr.Textbox(
                label="Base Weights Multiplier",
                placeholder="Multipliers for network weights to merge into the model before training",
                value=self.config.get("base_weights_multiplier", ""),
            )