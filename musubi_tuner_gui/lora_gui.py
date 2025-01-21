import gradio as gr
import os
import subprocess
import time
import toml

from datetime import datetime
from .class_accelerate_launch import AccelerateLaunch
from .class_advanced_training import AdvancedTraining
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig
from .class_latent_caching import LatentCaching
from .class_model import Model
from .class_network import Network
from .class_optimizer_and_scheduler import OptimizerAndScheduler
from .class_save_load import SaveLoadSettings
from .class_text_encoder_outputs_caching import TextEncoderOutputsCaching
from .class_training import TrainingSettings
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
    # Latent Caching
    caching_latent_device,
    caching_latent_batch_size,
    caching_latent_num_workers,
    caching_latent_skip_existing,
    caching_latent_keep_cache,
    caching_latent_debug_mode,
    caching_latent_console_width,
    caching_latent_console_back,
    caching_latent_console_num_images,
    
    # Text Encoder Outputs Caching
    caching_teo_text_encoder1,
    caching_teo_text_encoder2,
    caching_teo_text_encoder_dtype,
    caching_teo_device,
    caching_teo_fp8_llm,
    caching_teo_batch_size,
    caching_teo_num_workers,
    caching_teo_skip_existing,
    caching_teo_keep_cache,
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
    
    if not print_only:
        run_cache_latent_cmd = ["uv", "run", "./musubi-tuner/cache_latents.py",
                                "--dataset_config", str(param_dict.get("dataset_config")),
                                "--vae", str(param_dict.get("vae"))
        ]
        
        if param_dict.get("vae_dtype"):
            run_cache_latent_cmd.append("--vae_dtype")
            run_cache_latent_cmd.append(str(param_dict.get("vae_dtype")))   
            
        if param_dict.get("vae_chunk_size"):
            run_cache_latent_cmd.append("--vae_chunk_size")
            run_cache_latent_cmd.append(str(param_dict.get("vae_chunk_size")))
                                
        if param_dict.get("vae_tiling"):
            run_cache_latent_cmd.append("--vae_tiling")
            
        if param_dict.get("vae_spatial_tile_sample_min_size"):
            run_cache_latent_cmd.append("--vae_spatial_tile_sample_min_size")
            run_cache_latent_cmd.append(str(param_dict.get("vae_spatial_tile_sample_min_size")))
            
        if param_dict.get("caching_latent_device"):
            run_cache_latent_cmd.append("--device")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_device")))
        
        if param_dict.get("caching_latent_batch_size"):
            run_cache_latent_cmd.append("--batch_size")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_batch_size")))
        
        if param_dict.get("caching_latent_num_workers"):
            run_cache_latent_cmd.append("--num_workers")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_num_workers")))
            
        if param_dict.get("caching_latent_skip_existing"):
            run_cache_latent_cmd.append("--skip_existing")
            
        if param_dict.get("caching_latent_keep_cache"):
            run_cache_latent_cmd.append("--keep_cache")
            
        if param_dict.get("caching_latent_debug_mode"):
            run_cache_latent_cmd.append("--debug_mode")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_debug_mode")))
            
        if param_dict.get("caching_latent_console_width"):
            run_cache_latent_cmd.append("--console_width")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_console_width")))
            
        if param_dict.get("caching_latent_console_back"):
            run_cache_latent_cmd.append("--console_back")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_console_back")))
            
        if param_dict.get("caching_latent_console_num_images"):
            run_cache_latent_cmd.append("--console_num_images")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_console_num_images")))
        
        # Reconstruct the safe command string for display
        # command_to_run = " ".join(run_cache_latent_cmd)
        log.info(f"Executing command: {run_cache_latent_cmd}")

        # Execute the command securely
        log.info("Caching latents...")
        subprocess.run(run_cache_latent_cmd, env=setup_environment())
        #subprocess.Popen(run_cache_latent_cmd, setup_environment())
        log.debug("Command executed.")
        
        ######
        
        run_cache_teo_cmd = ["uv", "run", "./musubi-tuner/cache_text_encoder_outputs.py",
                                "--dataset_config", str(param_dict.get("dataset_config"))
        ]
        
        if param_dict.get("caching_teo_text_encoder1"):
            run_cache_teo_cmd.append("--text_encoder1")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_text_encoder1")))   
            
        if param_dict.get("caching_teo_text_encoder2"):
            run_cache_teo_cmd.append("--text_encoder2")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_text_encoder2")))
                                
        if param_dict.get("caching_teo_fp8_llm"):
            run_cache_teo_cmd.append("--fp8_llm")
            
        if param_dict.get("caching_teo_device"):
            run_cache_teo_cmd.append("--device")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_device")))
        
        if param_dict.get("caching_teo_text_encoder_dtype"):
            run_cache_teo_cmd.append("--text_encoder_dtype")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_text_encoder_dtype")))
        
        if param_dict.get("caching_teo_batch_size"):
            run_cache_teo_cmd.append("--batch_size")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_batch_size")))
            
        if param_dict.get("caching_teo_skip_existing"):
            run_cache_teo_cmd.append("--skip_existing")
            
        if param_dict.get("caching_teo__keep_cache"):
            run_cache_teo_cmd.append("--keep_cache")
            
        if param_dict.get("caching_teo_num_workers"):
            run_cache_teo_cmd.append("--num_workers")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_num_workers")))
        
        # Reconstruct the safe command string for display
        # command_to_run = " ".join(run_cache_teo_cmd)
        log.info(f"Executing command: {run_cache_teo_cmd}")

        # Execute the command securely
        log.info("Caching text encoder outputs...")
        subprocess.run(run_cache_teo_cmd, env=setup_environment())
        #subprocess.Popen(run_cache_teo_cmd, setup_environment())
        log.debug("Command executed.")
        
        #######

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

    # Saving config file for model
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    # config_dir = os.path.dirname(os.path.dirname(train_data_dir))
    file_path = os.path.join(param_dict.get('output_dir'), f"{param_dict.get('output_name')}_{formatted_datetime}.toml")

    log.info(f"Saving training config to {file_path}...")

    pattern_exclusion = []
    for key, _ in parameters:
        if key.startswith('caching_latent_') or key.startswith('caching_teo_'):
            pattern_exclusion.append(key)

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
        ] + pattern_exclusion,
    )
        
    if print_only:
        # log.info(rf"Printing configuration file {file_path}...")
        with open(file_path, 'r') as file:
            log.info("\n" + file.read())
    else:
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
    

    # Setup Configuration Files Gradio
    with gr.Accordion("Configuration file Settings", open=False):
        configuration = ConfigurationFile(headless=headless, config=config)

    with gr.Accordion("Accelerate launch Settings", open=False, elem_classes="flux1_background"), gr.Column():
        accelerate_launch = AccelerateLaunch(config=config)
        
    with gr.Accordion("Model Settings", open=True, elem_classes="preset_background"):
        model = Model(headless=headless, config=config)
        
    with gr.Accordion("Caching", open=True, elem_classes="samples_background"):
        with gr.Tab("Latent caching"):
            latentCaching = LatentCaching(headless=headless, config=config)
                
        with gr.Tab("Text encoder caching"):
            teoCaching = TextEncoderOutputsCaching(headless=headless, config=config)
        
    with gr.Accordion("Save Load Settings", open=True, elem_classes="samples_background"):
        saveLoadSettings = SaveLoadSettings(headless=headless, config=config)
        
    with gr.Accordion("Optimizer and Scheduler Settings", open=True, elem_classes="flux1_rank_layers_background"):
        OptimizerAndSchedulerSettings = OptimizerAndScheduler(headless=headless, config=config)
        
    with gr.Accordion("Network Settings", open=True, elem_classes="flux1_background"):
        network = Network(headless=headless, config=config)
        
    with gr.Accordion("Training Settings", open=True, elem_classes="preset_background"):
        trainingSettings = TrainingSettings(headless=headless, config=config)

    with gr.Accordion("Advanced Settings", open=True, elem_classes="samples_background"):
        advanced_training = AdvancedTraining(
            headless=headless, training_type="lora", config=config
        )

    with gr.Accordion("Metadata Settings", open=False, elem_classes="flux1_rank_layers_background"), gr.Group():
        metadata = MetaData(config=config)

    global huggingface
    with gr.Accordion("HuggingFace Settings", open=False, elem_classes="huggingface_background"):
        huggingface = HuggingFace(config=config)

    settings_list = [
        # accelerate_launch
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
        
        # advanced_training
        advanced_training.additional_parameters,
        
        # Dataset Settings
        model.dataset_config,
        
        # trainingSettings
        trainingSettings.sdpa,
        trainingSettings.flash_attn,
        trainingSettings.sage_attn,
        trainingSettings.xformers,
        trainingSettings.split_attn,
        trainingSettings.max_train_steps,
        trainingSettings.max_train_epochs,
        trainingSettings.max_data_loader_n_workers,
        trainingSettings.persistent_data_loader_workers,
        trainingSettings.seed,
        trainingSettings.gradient_checkpointing,
        trainingSettings.gradient_accumulation_steps,
        trainingSettings.logging_dir,
        trainingSettings.log_with,
        trainingSettings.log_prefix,
        trainingSettings.log_tracker_name,
        trainingSettings.wandb_run_name,
        trainingSettings.log_tracker_config,
        trainingSettings.wandb_api_key,
        trainingSettings.log_config,
        trainingSettings.ddp_timeout,
        trainingSettings.ddp_gradient_as_bucket_view,
        trainingSettings.ddp_static_graph,
        trainingSettings.sample_every_n_steps,
        trainingSettings.sample_at_first,
        trainingSettings.sample_every_n_epochs,
        trainingSettings.sample_prompts,
        
        # Latent Caching
        latentCaching.caching_latent_device,
        latentCaching.caching_latent_batch_size,
        latentCaching.caching_latent_num_workers,
        latentCaching.caching_latent_skip_existing,
        latentCaching.caching_latent_keep_cache,
        latentCaching.caching_latent_debug_mode,
        latentCaching.caching_latent_console_width,
        latentCaching.caching_latent_console_back,
        latentCaching.caching_latent_console_num_images,
        
        # Text Encoder Outputs Caching
        teoCaching.caching_teo_text_encoder1,
        teoCaching.caching_teo_text_encoder2,
        teoCaching.caching_teo_text_encoder_dtype,
        teoCaching.caching_teo_device,
        teoCaching.caching_teo_fp8_llm,
        teoCaching.caching_teo_batch_size,
        teoCaching.caching_teo_num_workers,
        teoCaching.caching_teo_skip_existing,
        teoCaching.caching_teo_keep_cache,
        
        # OptimizerAndSchedulerSettings
        OptimizerAndSchedulerSettings.optimizer_type,
        OptimizerAndSchedulerSettings.optimizer_args,
        OptimizerAndSchedulerSettings.learning_rate,
        OptimizerAndSchedulerSettings.max_grad_norm,
        OptimizerAndSchedulerSettings.lr_scheduler,
        OptimizerAndSchedulerSettings.lr_warmup_steps,
        OptimizerAndSchedulerSettings.lr_decay_steps,
        OptimizerAndSchedulerSettings.lr_scheduler_num_cycles,
        OptimizerAndSchedulerSettings.lr_scheduler_power,
        OptimizerAndSchedulerSettings.lr_scheduler_timescale,
        OptimizerAndSchedulerSettings.lr_scheduler_min_lr_ratio,
        OptimizerAndSchedulerSettings.lr_scheduler_type,
        OptimizerAndSchedulerSettings.lr_scheduler_args,
        
        # model
        model.dit,
        model.dit_dtype,
        model.vae,
        model.vae_dtype,
        model.vae_tiling,
        model.vae_chunk_size,
        model.vae_spatial_tile_sample_min_size,
        model.text_encoder1,
        model.text_encoder2,
        model.fp8_llm,
        model.fp8_base,
        model.blocks_to_swap,
        model.img_in_txt_in_offloading,
        model.guidance_scale,
        model.timestep_sampling,
        model.discrete_flow_shift,
        model.sigmoid_scale,
        model.weighting_scheme,
        model.logit_mean,
        model.logit_std,
        model.mode_scale,
        model.min_timestep,
        model.max_timestep,
        model.show_timesteps,
        
        # network
        network.no_metadata,
        network.network_weights,
        network.network_module,
        network.network_dim,
        network.network_alpha,
        network.network_dropout,
        network.network_args,
        network.training_comment,
        network.dim_from_weights,
        network.scale_weight_norms,
        network.base_weights,
        network.base_weights_multiplier,
        
        # saveLoadSettings
        saveLoadSettings.output_dir,
        saveLoadSettings.output_name,
        saveLoadSettings.resume,
        saveLoadSettings.save_every_n_epochs,
        saveLoadSettings.save_every_n_steps,
        saveLoadSettings.save_last_n_epochs,
        saveLoadSettings.save_last_n_epochs_state,
        saveLoadSettings.save_last_n_steps,
        saveLoadSettings.save_last_n_steps_state,
        saveLoadSettings.save_state,
        saveLoadSettings.save_state_on_train_end,
        
        # huggingface
        huggingface.huggingface_repo_id,
        huggingface.huggingface_token,
        huggingface.huggingface_repo_type,
        huggingface.huggingface_repo_visibility,
        huggingface.huggingface_path_in_repo,
        huggingface.save_state_to_huggingface,
        huggingface.resume_from_huggingface,
        huggingface.async_upload,
        
        # metadata
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
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name] + settings_list,
        show_progress=False,
    )

    configuration.button_load_config.click(
        gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name] + settings_list,
        show_progress=False,
    )

    configuration.button_save_config.click(
        gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name],
        show_progress=False,
    )

    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.button_stop_training],
    )

    button_print.click(
        gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[executor.button_run, executor.button_stop_training, run_state],
        show_progress=False,
    )

    executor.button_stop_training.click(
        executor.kill_command,
        outputs=[executor.button_run, executor.button_stop_training],
    )

    