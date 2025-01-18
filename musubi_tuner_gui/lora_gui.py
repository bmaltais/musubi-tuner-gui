import gradio as gr
import time

from datetime import datetime
from .class_accelerate_launch import AccelerateLaunch
from .class_advanced_training import AdvancedTraining
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig

from .common_gui import (
    print_command_and_toml,
    run_cmd_advanced_training,
    scriptdir,
    setup_environment,
)

# Setup command executor
executor = None

train_state_value = time.time()

def train_model(
    # control
    headless,
    print_only,
    
    # accelerate_launch
    mixed_precision,
    num_cpu_threads_per_process,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    
    # advanced_training
    additional_parameters
):
    run_cmd = [rf"uv", "run", "accelerate", "launch"]
    
    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
        dynamo_backend=dynamo_backend,
        dynamo_mode=dynamo_mode,
        dynamo_use_fullgraph=dynamo_use_fullgraph,
        dynamo_use_dynamic=dynamo_use_dynamic,
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        main_process_port=main_process_port,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
        mixed_precision=mixed_precision,
        extra_accelerate_launch_args=extra_accelerate_launch_args,
    )
    
    run_cmd.append(rf"{scriptdir}/musubi-tuner/hv_train_network.py")
    
    # Define a dictionary of parameters
    run_cmd_params = {
        "additional_parameters": additional_parameters,
    }
    
    # Use the ** syntax to unpack the dictionary when calling the function
    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **run_cmd_params)
    
    if print_only:
        print_command_and_toml(run_cmd, "")
    else:
        # # Saving config file for model
        # current_datetime = datetime.now()
        # formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        # # config_dir = os.path.dirname(os.path.dirname(train_data_dir))
        # file_path = os.path.join(output_dir, f"{output_name}_{formatted_datetime}.json")

        # log.info(f"Saving training config to {file_path}...")

        # SaveConfigFile(
        #     parameters=parameters,
        #     file_path=file_path,
        #     exclusion=["file_path", "save_as", "headless", "print_only"],
        # )

        # log.info(run_cmd)
        env = setup_environment()

        # Run the command

        executor.execute_command(run_cmd=run_cmd, env=env)

        train_state_value = time.time()

        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=train_state_value),
        )

def lora_tab(
    headless=False,
    config: GUIConfig = {},
):
    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)
    
    with gr.Accordion("Accelerate launch", open=False), gr.Column():
            accelerate_launch = AccelerateLaunch(config=config)
    
    advanced_training = AdvancedTraining(
        headless=headless, training_type="lora", config=config
    )
    
    # advanced_training.color_aug.change(
    #     color_aug_changed,
    #     inputs=[advanced_training.color_aug],
    #     outputs=[basic_training.cache_latents],
    # )
    
    settings_list = [
        accelerate_launch.mixed_precision,
        accelerate_launch.num_cpu_threads_per_process,
        accelerate_launch.num_processes,
        accelerate_launch.num_machines,
        accelerate_launch.multi_gpu,
        accelerate_launch.gpu_ids,
        accelerate_launch.main_process_port,
        accelerate_launch.dynamo_backend,
        accelerate_launch.dynamo_mode,
        accelerate_launch.dynamo_use_fullgraph,
        accelerate_launch.dynamo_use_dynamic,
        accelerate_launch.extra_accelerate_launch_args,
        advanced_training.additional_parameters,
    ]
    
    run_state = gr.Textbox(value=train_state_value, visible=False)
    
    with gr.Column(), gr.Group():
            with gr.Row():
                button_print = gr.Button("Print training command")
    
    global executor
    executor = CommandExecutor(headless=headless)
    
    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.button_stop_training],
    )
    
    button_print.click(
        train_model,
        inputs=[dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )
    
    executor.button_run.click(
        train_model,
        inputs=[dummy_headless, dummy_false] + settings_list,
        outputs=[executor.button_run, executor.button_stop_training, run_state],
        show_progress=False,
    )
    
    executor.button_stop_training.click(
        executor.kill_command,
        outputs=[executor.button_run, executor.button_stop_training],
    )