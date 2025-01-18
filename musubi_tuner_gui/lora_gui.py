import gradio as gr
from .class_advanced_training import AdvancedTraining
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig

def lora_tab(
    headless=False,
    config: GUIConfig = {},
):
    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)
    
    advanced_training = AdvancedTraining(
        headless=headless, training_type="lora", config=config
    )
    
    # advanced_training.color_aug.change(
    #     color_aug_changed,
    #     inputs=[advanced_training.color_aug],
    #     outputs=[basic_training.cache_latents],
    # )