import gradio as gr
import os
import time
import toml

from datetime import datetime
from .class_accelerate_launch import AccelerateLaunch
from .class_advanced_training import AdvancedTraining
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig
from .common_gui import (
    get_file_path,
    get_saveasfile_path,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    scriptdir,
    setup_environment,
)
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

train_state_value = time.time()

def save_configuration(
    # control
    save_as_bool,
    file_path,
    
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
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    # If saving as a new file, get the file path for saving
    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(file_path, defaultextension=".toml", extension_name="TOML files (*.toml)")
    # If not saving as a new file, check if a file path was provided
    else:
        log.info("Save...")
        # If no file path was provided, get the file path for saving
        if file_path == None or file_path == "":
            file_path = get_saveasfile_path(file_path, defaultextension=".toml", extension_name="TOML files (*.toml)")

    # Log the file path for debugging purposes
    log.debug(file_path)

    # If no file path was provided, return the original file path
    if file_path == None or file_path == "":
        return original_file_path  # In case a file_path was provided and the user decide to cancel the open action

    # Extract the destination directory from the file path
    destination_directory = os.path.dirname(file_path)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Save the configuration file
    SaveConfigFile(
        parameters=parameters,
        file_path=file_path,
        exclusion=["file_path", "save_as"],
    )

    # Return the file path of the saved configuration
    return file_path

def open_configuration(
    # control
    ask_for_file,
    file_path,
    
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
    # Get list of function parameters and their values
    parameters = list(locals().items())

    # Store the original file path for potential reuse
    original_file_path = file_path

    # Request a file path from the user if required
    if ask_for_file:
        file_path = get_file_path(file_path, default_extension=".toml", extension_name="TOML files (*.toml)")

    # Proceed if the file path is valid (not empty or None)
    if not file_path == "" and not file_path == None:
        # Check if the file exists before opening it
        if not os.path.isfile(file_path):
            log.error(f"Config file {file_path} does not exist.")
            return

        # Load variables from TOML file
        with open(file_path, "r", encoding="utf-8") as f:
            my_data = toml.load(f)
            log.info("Loading config...")
    else:
        # Reset the file path to the original if the operation was cancelled or invalid
        file_path = original_file_path  # In case a file_path was provided and the user decides to cancel the open action
        my_data = {}  # Initialize an empty dict if no data was loaded

    values = [file_path]
    # Iterate over parameters to set their values from `my_data` or use default if not found
    for key, value in parameters:
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            toml_value = my_data.get(key)
            # Append the value from TOML if present; otherwise, use the parameter's default value
            values.append(toml_value if toml_value is not None else value)

    return tuple(values)

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
            
    # Setup Configuration Files Gradio
    with gr.Accordion("Configuration", open=False):
        configuration = ConfigurationFile(headless=headless, config=config)
    
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
    
    configuration.button_open_config.click(
            open_configuration,
            inputs=[dummy_true, configuration.config_file_name]
            + settings_list,
            outputs=[configuration.config_file_name]
            + settings_list,
            show_progress=False,
        )
    
    configuration.button_load_config.click(
            open_configuration,
            inputs=[dummy_false, configuration.config_file_name]
            + settings_list,
            outputs=[configuration.config_file_name]
            + settings_list,
            show_progress=False,
        )
    
    configuration.button_save_config.click(
        save_configuration,
        inputs=[dummy_false, configuration.config_file_name] + settings_list,
        outputs=[configuration.config_file_name],
        show_progress=False,
    )
    
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