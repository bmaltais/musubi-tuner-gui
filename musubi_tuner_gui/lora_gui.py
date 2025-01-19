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
    SaveConfigFileToRun,
    scriptdir,
    setup_environment,
)
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

# Setup huggingface
huggingface = None

train_state_value = time.time()

def gui_actions(
    # action type
    action_type,
    # control
    bool_value,
    file_path,
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
    additional_parameters,
    dataset_config,
    sdpa,
    flash_attn,
    sage_attn,
    xformers,
    split_attn,
    max_train_steps,
    max_train_epochs,
    max_data_loader_n_workers,
    persistent_data_loader_workers,
    seed,
    gradient_checkpointing,
    gradient_accumulation_steps,
    logging_dir,
    log_with,
    log_prefix,
    log_tracker_name,
    wandb_run_name,
    log_tracker_config,
    wandb_api_key,
    log_config,
    ddp_timeout,
    ddp_gradient_as_bucket_view,
    ddp_static_graph,
    sample_every_n_steps,
    sample_at_first,
    sample_every_n_epochs,
    sample_prompts,
    optimizer_type,
    optimizer_args,
    learning_rate,
    max_grad_norm,
    lr_scheduler,
    lr_warmup_steps,
    lr_decay_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    lr_scheduler_timescale,
    lr_scheduler_min_lr_ratio,
    lr_scheduler_type,
    lr_scheduler_args,
    dit,
    dit_dtype,
    vae,
    vae_dtype,
    vae_tiling,
    vae_chunk_size,
    vae_spatial_tile_sample_min_size,
    text_encoder1,
    text_encoder2,
    fp8_llm,
    fp8_base,
    blocks_to_swap,
    img_in_txt_in_offloading,
    guidance_scale,
    timestep_sampling,
    discrete_flow_shift,
    sigmoid_scale,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    min_timestep,
    max_timestep,
    show_timesteps,
    no_metadata,
    network_weights,
    network_module,
    network_dim,
    network_alpha,
    network_dropout,
    network_args,
    training_comment,
    dim_from_weights,
    scale_weight_norms,
    base_weights,
    base_weights_multiplier,
    output_dir,
    output_name,
    resume,
    save_every_n_epochs,
    save_every_n_steps,
    save_last_n_epochs,
    save_last_n_epochs_state,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
    metadata_author,
    metadata_description,
    metadata_license,
    metadata_tags,
    metadata_title,
):
    # Get list of function parameters and values
    parameters = [(k, v) for k, v in locals().items() if k not in ["action_type", "bool_value", "headless", "print_only"]]
    
    if action_type == "save_configuration":
        log.info("Save configuration...")
        return save_configuration(
            save_as_bool=bool_value,
            file_path=file_path,
            parameters=parameters,
        )
        
    if action_type == "open_configuration":
        log.info("Open configuration...")
        return open_configuration(
            ask_for_file=bool_value,
            file_path=file_path,
            parameters=parameters,
        )
        
    if action_type == "train_model":
        log.info("Train model...")
        return train_model(
            headless=headless,
            print_only=print_only,
            parameters=parameters,
        )

def save_configuration(
    # control
    save_as_bool,
    file_path,
    parameters,
):
    # Get list of function parameters and values
    # parameters = list(locals().items())

    # print(parameters)

    original_file_path = file_path

    # If saving as a new file, get the file path for saving
    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(
            file_path, defaultextension=".toml", extension_name="TOML files (*.toml)"
        )
    # If not saving as a new file, check if a file path was provided
    else:
        log.info("Save...")
        # If no file path was provided, get the file path for saving
        if file_path == None or file_path == "":
            file_path = get_saveasfile_path(
                file_path,
                defaultextension=".toml",
                extension_name="TOML files (*.toml)",
            )

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
        exclusion=[
            "file_path",
            "save_as",
            "save_as_bool",
            # "num_cpu_threads_per_process",
            # "num_processes",
            # "num_machines",
            # "multi_gpu",
            # "gpu_ids",
            # "main_process_port",
            # "dynamo_backend",
            # "dynamo_mode",
            # "dynamo_use_fullgraph",
            # "dynamo_use_dynamic",
            # "extra_accelerate_launch_args",
        ],
    )

    # Return the file path of the saved configuration
    return file_path


def open_configuration(
    # control
    ask_for_file,
    file_path,
    parameters,
):
    # Get list of function parameters and their values
    # parameters = list(locals().items())

    # Store the original file path for potential reuse
    original_file_path = file_path

    # Request a file path from the user if required
    if ask_for_file:
        file_path = get_file_path(
            file_path, default_extension=".toml", extension_name="TOML files (*.toml)"
        )

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
    parameters,
):
    # Get list of function parameters and their values
    # parameters = list(locals().items())
    
    run_cmd = [rf"uv", "run", "accelerate", "launch"]

    param_dict = dict(parameters)

    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
        dynamo_backend=param_dict.get("dynamo_backend"),
        dynamo_mode=param_dict.get("dynamo_mode"),
        dynamo_use_fullgraph=param_dict.get("dynamo_use_fullgraph"),
        dynamo_use_dynamic=param_dict.get("dynamo_use_dynamic"),
        num_processes=param_dict.get("num_processes"),
        num_machines=param_dict.get("num_machines"),
        multi_gpu=param_dict.get("multi_gpu"),
        gpu_ids=param_dict.get("gpu_ids"),
        main_process_port=param_dict.get("main_process_port"),
        num_cpu_threads_per_process=param_dict.get("num_cpu_threads_per_process"),
        mixed_precision=param_dict.get("mixed_precision"),
        extra_accelerate_launch_args=param_dict.get("extra_accelerate_launch_args"),
    )

    run_cmd.append(rf"{scriptdir}/musubi-tuner/hv_train_network.py")

    if print_only:
        print_command_and_toml(run_cmd, "")
    else:
        # Saving config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        # config_dir = os.path.dirname(os.path.dirname(train_data_dir))
        file_path = os.path.join(param_dict.get('output_dir'), f"{param_dict.get('output_name')}_{formatted_datetime}.toml")

        log.info(f"Saving training config to {file_path}...")

        SaveConfigFileToRun(
            parameters=parameters,
            file_path=file_path,
            exclusion=[
                "file_path",
                "save_as",
                "save_as_bool",
                "headless",
                "num_cpu_threads_per_process",
                "num_processes",
                "num_machines",
                "multi_gpu",
                "gpu_ids",
                "main_process_port",
                "dynamo_backend",
                "dynamo_mode",
                "dynamo_use_fullgraph",
                "dynamo_use_dynamic",
                "extra_accelerate_launch_args",
            ],
        )
        
        run_cmd.append("--config_file")
        run_cmd.append(rf"{file_path}")

        # Define a dictionary of parameters
        run_cmd_params = {
            "additional_parameters": param_dict.get("additional_parameters"),
        }

        # Use the ** syntax to unpack the dictionary when calling the function
        run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **run_cmd_params)

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

    with gr.Accordion("Metadata", open=False), gr.Group():
        metadata = MetaData(config=config)

    global huggingface
    with gr.Accordion("HuggingFace", open=False, elem_classes="huggingface_background"):
        huggingface = HuggingFace(config=config)

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
        advanced_training.dataset_config,
        advanced_training.sdpa,
        advanced_training.flash_attn,
        advanced_training.sage_attn,
        advanced_training.xformers,
        advanced_training.split_attn,
        advanced_training.max_train_steps,
        advanced_training.max_train_epochs,
        advanced_training.max_data_loader_n_workers,
        advanced_training.persistent_data_loader_workers,
        advanced_training.seed,
        advanced_training.gradient_checkpointing,
        advanced_training.gradient_accumulation_steps,
        advanced_training.logging_dir,
        advanced_training.log_with,
        advanced_training.log_prefix,
        advanced_training.log_tracker_name,
        advanced_training.wandb_run_name,
        advanced_training.log_tracker_config,
        advanced_training.wandb_api_key,
        advanced_training.log_config,
        advanced_training.ddp_timeout,
        advanced_training.ddp_gradient_as_bucket_view,
        advanced_training.ddp_static_graph,
        advanced_training.sample_every_n_steps,
        advanced_training.sample_at_first,
        advanced_training.sample_every_n_epochs,
        advanced_training.sample_prompts,
        advanced_training.optimizer_type,
        advanced_training.optimizer_args,
        advanced_training.learning_rate,
        advanced_training.max_grad_norm,
        advanced_training.lr_scheduler,
        advanced_training.lr_warmup_steps,
        advanced_training.lr_decay_steps,
        advanced_training.lr_scheduler_num_cycles,
        advanced_training.lr_scheduler_power,
        advanced_training.lr_scheduler_timescale,
        advanced_training.lr_scheduler_min_lr_ratio,
        advanced_training.lr_scheduler_type,
        advanced_training.lr_scheduler_args,
        advanced_training.dit,
        advanced_training.dit_dtype,
        advanced_training.vae,
        advanced_training.vae_dtype,
        advanced_training.vae_tiling,
        advanced_training.vae_chunk_size,
        advanced_training.vae_spatial_tile_sample_min_size,
        advanced_training.text_encoder1,
        advanced_training.text_encoder2,
        advanced_training.fp8_llm,
        advanced_training.fp8_base,
        advanced_training.blocks_to_swap,
        advanced_training.img_in_txt_in_offloading,
        advanced_training.guidance_scale,
        advanced_training.timestep_sampling,
        advanced_training.discrete_flow_shift,
        advanced_training.sigmoid_scale,
        advanced_training.weighting_scheme,
        advanced_training.logit_mean,
        advanced_training.logit_std,
        advanced_training.mode_scale,
        advanced_training.min_timestep,
        advanced_training.max_timestep,
        advanced_training.show_timesteps,
        advanced_training.no_metadata,
        advanced_training.network_weights,
        advanced_training.network_module,
        advanced_training.network_dim,
        advanced_training.network_alpha,
        advanced_training.network_dropout,
        advanced_training.network_args,
        advanced_training.training_comment,
        advanced_training.dim_from_weights,
        advanced_training.scale_weight_norms,
        advanced_training.base_weights,
        advanced_training.base_weights_multiplier,
        advanced_training.output_dir,
        advanced_training.output_name,
        advanced_training.resume,
        advanced_training.save_every_n_epochs,
        advanced_training.save_every_n_steps,
        advanced_training.save_last_n_epochs,
        advanced_training.save_last_n_epochs_state,
        advanced_training.save_last_n_steps,
        advanced_training.save_last_n_steps_state,
        advanced_training.save_state,
        advanced_training.save_state_on_train_end,
        huggingface.huggingface_repo_id,
        huggingface.huggingface_token,
        huggingface.huggingface_repo_type,
        huggingface.huggingface_repo_visibility,
        huggingface.huggingface_path_in_repo,
        huggingface.save_state_to_huggingface,
        huggingface.resume_from_huggingface,
        huggingface.async_upload,
        metadata.metadata_author,
        metadata.metadata_description,
        metadata.metadata_license,
        metadata.metadata_tags,
        metadata.metadata_title,
    ]

    run_state = gr.Textbox(value=train_state_value, visible=False)

    with gr.Column(), gr.Group():
        with gr.Row():
            button_print = gr.Button("Print training command")

    global executor
    executor = CommandExecutor(headless=headless)

    configuration.button_open_config.click(
        gui_actions,
        inputs=[gr.Textbox(value="open_configuration"), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name] + settings_list,
        show_progress=False,
    )

    configuration.button_load_config.click(
        gui_actions,
        inputs=[gr.Textbox(value="open_configuration"), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name] + settings_list,
        show_progress=False,
    )

    configuration.button_save_config.click(
        gui_actions,
        inputs=[gr.Textbox(value="save_configuration"), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name],
        show_progress=False,
    )

    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.button_stop_training],
    )

    button_print.click(
        gui_actions,
        inputs=[gr.Textbox(value="train_model"), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        gui_actions,
        inputs=[gr.Textbox(value="train_model"), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[executor.button_run, executor.button_stop_training, run_state],
        show_progress=False,
    )

    executor.button_stop_training.click(
        executor.kill_command,
        outputs=[executor.button_run, executor.button_stop_training],
    )
