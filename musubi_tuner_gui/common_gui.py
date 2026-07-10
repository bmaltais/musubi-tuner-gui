try:
    from tkinter import filedialog, Tk
except ImportError:
    pass
from .custom_logging import setup_logging

import os
import re
import gradio as gr
import sys
import shlex
import json
import toml

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001f4c4"  # 📄

scriptdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if os.name == "nt":
    scriptdir = scriptdir.replace("\\", "/")

# insert sd-scripts path into PYTHONPATH
sys.path.insert(0, os.path.join(scriptdir, "musubi-tuner"))

ENV_EXCLUSION = ["COLAB_GPU", "RUNPOD_POD_ID"]


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    """
    Creates a refresh button that can be used to update UI components.

    Parameters:
    refresh_component (list or object): The UI component(s) to be refreshed.
    refresh_method (callable): The method to be called when the button is clicked.
    refreshed_args (dict or callable): The arguments to be passed to the refresh method.
    elem_id (str): The ID of the button element.

    Returns:
    gr.Button: The configured refresh button.
    """
    # Converts refresh_component into a list for uniform processing. If it's already a list, keep it the same.
    refresh_components = (
        refresh_component
        if isinstance(refresh_component, list)
        else [refresh_component]
    )

    # Initialize label to None. This will store the label of the first component with a non-None label, if any.
    label = None
    # Iterate over each component to find the first non-None label and assign it to 'label'.
    for comp in refresh_components:
        label = getattr(comp, "label", None)
        if label is not None:
            break

    # Define the refresh function that will be triggered upon clicking the refresh button.
    def refresh():
        # Invoke the refresh_method, which is intended to perform the refresh operation.
        refresh_method()
        # Determine the arguments for the refresh: call refreshed_args if it's callable, otherwise use it directly.
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        # For each key-value pair in args, update the corresponding properties of each component.
        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        # Use gr.update to refresh the UI components. If multiple components are present, update each; else, update only the first.
        return (
            [gr.Dropdown(**(args or {})) for _ in refresh_components]
            if len(refresh_components) > 1
            else gr.Dropdown(**(args or {}))
        )

    # Create a refresh button with the specified label (via refresh_symbol), ID, and classes.
    # 'refresh_symbol' should be defined outside this function or passed as an argument, representing the button's label or icon.
    refresh_button = gr.Button(
        value=refresh_symbol, elem_id=elem_id, elem_classes=["tool"]
    )
    # Configure the button to invoke the refresh function.
    refresh_button.click(fn=refresh, inputs=[], outputs=refresh_components)
    # Return the configured refresh button to be used in the UI.
    return refresh_button


def list_dirs(path):
    if path is None or path == "None" or path == "":
        return

    if not os.path.exists(path):
        path = os.path.dirname(path)
        if not os.path.exists(path):
            return

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    def natural_sort_key(s, regex=re.compile("([0-9]+)")):
        return [
            int(text) if text.isdigit() else text.lower() for text in regex.split(s)
        ]

    subdirs = [
        (item, os.path.join(path, item))
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
    ]
    subdirs = [
        filename
        for item, filename in subdirs
        if item[0] != "." and item not in ["__pycache__"]
    ]
    subdirs = sorted(subdirs, key=natural_sort_key)
    if os.path.dirname(path) != "":
        dirs = [os.path.dirname(path), path] + subdirs
    else:
        dirs = [path] + subdirs

    if os.sep == "\\":
        dirs = [d.replace("\\", "/") for d in dirs]
    for d in dirs:
        yield d


def list_files(path, exts=None, all=False):
    if path is None or path == "None" or path == "":
        return

    if not os.path.exists(path):
        path = os.path.dirname(path)
        if not os.path.exists(path):
            return

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    files = [
        (item, os.path.join(path, item))
        for item in os.listdir(path)
        if all or os.path.isfile(os.path.join(path, item))
    ]
    files = [
        filename
        for item, filename in files
        if item[0] != "." and item not in ["__pycache__"]
    ]
    exts = set(exts) if exts is not None else None

    def natural_sort_key(s, regex=re.compile("([0-9]+)")):
        return [
            int(text) if text.isdigit() else text.lower() for text in regex.split(s)
        ]

    files = sorted(files, key=natural_sort_key)
    if os.path.dirname(path) != "":
        files = [os.path.dirname(path), path] + files
    else:
        files = [path] + files

    if os.sep == "\\":
        files = [d.replace("\\", "/") for d in files]

    for filename in files:
        if exts is not None:
            if os.path.isdir(filename):
                yield filename
            _, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                continue
            yield filename
        else:
            yield filename


def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)


def get_file_path(
    file_path="", default_extension=".json", extension_name="Config files"
):
    """
    Opens a file dialog to select a file, allowing the user to navigate and choose a file with a specific extension.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.
    - default_extension (str): The default file extension (e.g., ".json") for the file dialog.
    - extension_name (str): The display name for the type of files being selected (e.g., "Config files").

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path`, `default_extension`, or `extension_name` are not strings.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter types
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(default_extension, str):
        raise TypeError("default_extension must be a string")
    if not isinstance(extension_name, str):
        raise TypeError("extension_name must be a string")

    # Environment and platform check to decide on showing the file dialog
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_file_path = file_path  # Backup in case no file is selected

        initial_dir, initial_file = get_dir_and_file(
            file_path
        )  # Decompose file path for dialog setup

        # Initialize a hidden Tkinter window for the file dialog
        root = Tk()
        root.wm_attributes("-topmost", 1)  # Ensure the dialog is topmost
        root.withdraw()  # Hide the root window to show only the dialog

        # Open the file dialog and capture the selected file path
        file_path = filedialog.askopenfilename(
            filetypes=((extension_name, f"*{default_extension}"), ("All files", "*.*")),
            defaultextension=default_extension,
            initialfile=initial_file,
            initialdir=initial_dir,
        )

        root.destroy()  # Cleanup by destroying the Tkinter root window

        # Fallback to the initial path if no selection is made
        if not file_path:
            file_path = current_file_path

    # Return the selected or fallback file path
    return file_path


def get_any_file_path(file_path: str = "") -> str:
    """
    Opens a file dialog to select any file, allowing the user to navigate and choose a file.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the file dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter type
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    try:
        # Check for environment variable conditions
        if (
            not any(var in os.environ for var in ENV_EXCLUSION)
            and sys.platform != "darwin"
        ):
            current_file_path: str = file_path

            initial_dir, initial_file = get_dir_and_file(file_path)

            # Initialize a hidden Tkinter window for the file dialog
            root = Tk()
            root.wm_attributes("-topmost", 1)
            root.withdraw()

            try:
                # Open the file dialog and capture the selected file path
                file_path = filedialog.askopenfilename(
                    initialdir=initial_dir,
                    initialfile=initial_file,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to open file dialog: {e}")
            finally:
                root.destroy()

            # Fallback to the initial path if no selection is made
            if not file_path:
                file_path = current_file_path
    except KeyError as e:
        raise EnvironmentError(f"Failed to access environment variables: {e}")

    # Return the selected or fallback file path
    return file_path


def get_folder_path(folder_path: str = "") -> str:
    """
    Opens a folder dialog to select a folder, allowing the user to navigate and choose a folder.
    If no folder is selected, returns the initially provided folder path or an empty string if not provided.
    This function is conditioned to skip the folder dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - folder_path (str): The initial folder path or an empty string by default. Used as the fallback if no folder is selected.

    Returns:
    - str: The path of the folder selected by the user, or the initial `folder_path` if no selection is made.

    Raises:
    - TypeError: If `folder_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the folder dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the folder dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter type
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    try:
        # Check for environment variable conditions
        if any(var in os.environ for var in ENV_EXCLUSION) or sys.platform == "darwin":
            return folder_path or ""

        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=folder_path or ".")
        root.destroy()
        return selected_folder or folder_path
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e


def path_field(
    label: str,
    value=None,
    placeholder: str = None,
    info: str = None,
    is_folder: bool = False,
    default_extension: str = None,
    extension_name: str = "Files",
    scale: int = 4,
):
    """
    Renders a Textbox paired with a "📁 Browse" button that opens a native
    file/folder dialog and writes the selection back into the Textbox.

    Parameters:
    - label, value, placeholder, info: passed straight through to the Textbox.
    - is_folder: use a folder-picker dialog instead of a file-picker.
    - default_extension, extension_name: file-picker filter (ignored when
      is_folder is True, or when default_extension is None, which opens an
      unfiltered "any file" dialog).
    - scale: relative width of the Textbox versus the Browse button.

    Returns:
    - gr.Textbox: the path field. The Browse button is wired but not returned,
      matching how other field helpers in this codebase only expose the
      value-carrying component.
    """
    with gr.Row():
        textbox = gr.Textbox(
            label=label,
            value=value,
            placeholder=placeholder,
            info=info,
            scale=scale,
        )
        button = gr.Button("📁 Browse", scale=1, elem_classes="path-field-browse")

    if is_folder:
        button.click(
            fn=get_folder_path, inputs=textbox, outputs=textbox, show_progress=False
        )
    elif default_extension:
        button.click(
            fn=lambda p: get_file_path(p, default_extension, extension_name),
            inputs=textbox,
            outputs=textbox,
            show_progress=False,
        )
    else:
        button.click(
            fn=get_any_file_path, inputs=textbox, outputs=textbox, show_progress=False
        )

    return textbox


def get_saveasfile_path(
    file_path: str = "",
    defaultextension: str = ".json",
    extension_name: str = "Config files",
) -> str:
    # Check if the current environment is not macOS and if the environment variables do not match the exclusion list
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        # Store the initial file path to use as a fallback in case no file is selected
        current_file_path = file_path

        # Logging the current file path for debugging purposes; helps in tracking the flow of file selection
        # log.info(f'current file path: {current_file_path}')

        # Split the file path into directory and file name for setting the file dialog start location and filename
        initial_dir, initial_file = get_dir_and_file(file_path)

        # Initialize a hidden Tkinter window to act as the parent for the file dialog, ensuring it appears on top
        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        save_file_path = filedialog.asksaveasfile(
            filetypes=(
                (f"{extension_name}", f"{defaultextension}"),
                ("All files", "*"),
            ),
            defaultextension=defaultextension,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        # Close the Tkinter root window to clean up the UI
        root.destroy()

        # Logging the save file path for auditing purposes; useful in confirming the user's file choice
        # log.info(save_file_path)

        # Default to the current file path if no file is selected, ensuring there's always a valid file path
        if save_file_path == None:
            file_path = current_file_path
        else:
            # Log the selected file name for transparency and tracking user actions
            # log.info(save_file_path.name)

            # Update the file path with the user-selected file name, facilitating the save operation
            file_path = save_file_path.name

        # Log the final file path for verification, ensuring the intended file is being used
        # log.info(file_path)

    # Return the final file path, either the user-selected file or the fallback path
    return file_path


def get_saveasfilename_path(
    file_path: str = "",
    extensions: str = "*",
    extension_name: str = "Config files",
) -> str:
    """
    Opens a file dialog to select a file name for saving, allowing the user to specify a file name and location.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.
    - extensions (str): The file extensions to filter the file dialog by. Defaults to "*" for all files.
    - extension_name (str): The name to display for the file extensions in the file dialog. Defaults to "Config files".

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the file dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform == "darwin"`) as a specific behavior adjustment.
    """
    # Check if the current environment is not macOS and if the environment variables do not match the exclusion list
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        # Store the initial file path to use as a fallback in case no file is selected
        current_file_path: str = file_path
        # log.info(f'current file path: {current_file_path}')

        # Split the file path into directory and file name for setting the file dialog start location and filename
        initial_dir, initial_file = get_dir_and_file(file_path)

        # Initialize a hidden Tkinter window to act as the parent for the file dialog, ensuring it appears on top
        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        # Open the file dialog and capture the selected file path
        save_file_path = filedialog.asksaveasfilename(
            filetypes=(
                (f"{extension_name}", f"{extensions}"),
                ("All files", "*"),
            ),
            defaultextension=extensions,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        # Close the Tkinter root window to clean up the UI
        root.destroy()

        # Default to the current file path if no file is selected, ensuring there's always a valid file path
        if save_file_path == "":
            file_path = current_file_path
        else:
            # Logging the save file path for auditing purposes; useful in confirming the user's file choice
            # log.info(save_file_path)
            # Update the file path with the user-selected file name, facilitating the save operation
            file_path = save_file_path

    # Return the final file path, either the user-selected file or the fallback path
    return file_path


def add_pre_postfix(
    folder: str = "",
    prefix: str = "",
    postfix: str = "",
    caption_file_ext: str = ".caption",
    recursive: bool = False,
) -> None:
    """
    Add prefix and/or postfix to the content of caption files within a folder.
    If no caption files are found, create one with the requested prefix and/or postfix.

    Args:
        folder (str): Path to the folder containing caption files.
        prefix (str, optional): Prefix to add to the content of the caption files.
        postfix (str, optional): Postfix to add to the content of the caption files.
        caption_file_ext (str, optional): Extension of the caption files.
        recursive (bool, optional): Whether to search for caption files recursively.
    """
    # If neither prefix nor postfix is provided, return early
    if prefix == "" and postfix == "":
        return

    # Define the image file extensions to filter
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")

    # If recursive is true, list all image files in the folder and its subfolders
    if recursive:
        image_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        # List all image files in the folder
        image_files = [
            f for f in os.listdir(folder) if f.lower().endswith(image_extensions)
        ]

    # Iterate over the list of image files
    for image_file in image_files:
        # Construct the caption file name by appending the caption file extension to the image file name
        caption_file_name = f"{os.path.splitext(image_file)[0]}{caption_file_ext}"
        # Construct the full path to the caption file
        caption_file_path = os.path.join(folder, caption_file_name)

        # Check if the caption file does not exist
        if not os.path.exists(caption_file_path):
            # Create a new caption file with the specified prefix and/or postfix
            try:
                with open(caption_file_path, "w", encoding="utf-8") as f:
                    # Determine the separator based on whether both prefix and postfix are provided
                    separator = " " if prefix and postfix else ""
                    f.write(f"{prefix}{separator}{postfix}")
            except Exception as e:
                log.error(f"Error writing to file {caption_file_path}: {e}")
        else:
            # Open the existing caption file for reading and writing
            try:
                with open(caption_file_path, "r+", encoding="utf-8") as f:
                    # Read the content of the caption file, stripping any trailing whitespace
                    content = f.read().rstrip()
                    # Move the file pointer to the beginning of the file
                    f.seek(0, 0)

                    # Determine the separator based on whether only prefix is provided
                    prefix_separator = " " if prefix else ""
                    # Determine the separator based on whether only postfix is provided
                    postfix_separator = " " if postfix else ""
                    # Write the updated content to the caption file, adding prefix and/or postfix
                    f.write(
                        f"{prefix}{prefix_separator}{content}{postfix_separator}{postfix}"
                    )
            except Exception as e:
                log.error(f"Error writing to file {caption_file_path}: {e}")


def has_ext_files(folder_path: str, file_extension: str) -> bool:
    """
    Determines whether any files within a specified folder have a given file extension.

    This function iterates through each file in the specified folder and checks if
    its extension matches the provided file_extension argument. The search is case-sensitive
    and expects file_extension to include the dot ('.') if applicable (e.g., '.txt').

    Args:
        folder_path (str): The absolute or relative path to the folder to search within.
        file_extension (str): The file extension to search for, including the dot ('.') if applicable.

    Returns:
        bool: True if at least one file with the specified extension is found, False otherwise.
    """
    # Iterate directly over files in the specified folder path
    for file in os.listdir(folder_path):
        # Return True at the first occurrence of a file with the specified extension
        if file.endswith(file_extension):
            return True

    # If no file with the specified extension is found, return False
    return False


###
### Gradio common GUI section
###
def run_cmd_advanced_training(run_cmd: list = [], **kwargs):
    """
    This function, run_cmd_advanced_training, dynamically constructs a command line string for advanced training
    configurations based on provided keyword arguments (kwargs). Each argument represents a different training parameter
    or flag that can be used to customize the training process. The function checks for the presence and validity of
    arguments, appending them to the command line string with appropriate formatting.

    Purpose
        The primary purpose of this function is to enable flexible and customizable training configurations for machine
        learning models. It allows users to specify a wide range of parameters and flags that control various aspects of
        the training process, such as learning rates, batch sizes, augmentation options, precision settings, and many more.

    Args:
        kwargs (dict): A variable number of keyword arguments that represent different training parameters or flags.
                       Each argument has a specific expected data type and format, which the function checks before
                       appending to the command line string.

    Returns:
        str: A command line string constructed based on the provided keyword arguments. This string includes the base
             command and additional parameters and flags tailored to the user's specifications for the training process
    """
    if "additional_parameters" in kwargs and kwargs["additional_parameters"] != "":
        additional_parameters = kwargs["additional_parameters"].replace('"', "")
        for arg in additional_parameters.split():
            run_cmd.append(shlex.quote(arg))

    if "max_data_loader_n_workers" in kwargs:
        max_data_loader_n_workers = kwargs.get("max_data_loader_n_workers")
        if max_data_loader_n_workers != "":
            run_cmd.append("--max_data_loader_n_workers")
            run_cmd.append(str(max_data_loader_n_workers))

    return run_cmd


def verify_image_folder_pattern(folder_path: str) -> bool:
    """
    Verify the image folder pattern in the given folder path.

    Args:
        folder_path (str): The path to the folder containing image folders.

    Returns:
        bool: True if the image folder pattern is valid, False otherwise.
    """
    # Initialize the return value to True
    return_value = True

    # Log the start of the verification process
    log.info(f"Verifying image folder pattern of {folder_path}...")

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        # Log an error message if the folder does not exist
        log.error(
            f"...the provided path '{folder_path}' is not a valid folder. "
            "Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        # Return False to indicate that the folder pattern is not valid
        return False

    # Create a regular expression pattern to match the required sub-folder names
    # The pattern should start with one or more digits (\d+) followed by an underscore (_)
    # After the underscore, it should match one or more word characters (\w+), which can be letters, numbers, or underscores
    # Example of a valid pattern matching name: 123_example_folder
    pattern = r"^\d+_\w+"

    # Get the list of sub-folders in the directory
    subfolders = [
        os.path.join(folder_path, subfolder)
        for subfolder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subfolder))
    ]

    # Check the pattern of each sub-folder
    matching_subfolders = [
        subfolder
        for subfolder in subfolders
        if re.match(pattern, os.path.basename(subfolder))
    ]

    # Print non-matching sub-folders
    non_matching_subfolders = set(subfolders) - set(matching_subfolders)
    if non_matching_subfolders:
        # Log an error message if any sub-folders do not match the pattern
        log.error(
            f"...the following folders do not match the required pattern <number>_<text>: {', '.join(non_matching_subfolders)}"
        )
        # Log an error message suggesting to follow the folder structure documentation
        log.error(
            f"...please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        # Return False to indicate that the folder pattern is not valid
        return False

    # Check if no sub-folders exist
    if not matching_subfolders:
        # Log an error message if no image folders are found
        log.error(
            f"...no image folders found in {folder_path}. "
            "Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        # Return False to indicate that the folder pattern is not valid
        return False

    # Log the successful verification
    log.info(f"...valid")
    # Return True to indicate that the folder pattern is valid
    return return_value


def SaveConfigFile(
    parameters,
    file_path: str,
    exclusion: list = ["file_path", "save_as", "headless", "print_only"],
) -> None:
    """
    Saves the configuration parameters to a TOML file, excluding specified keys.

    This function iterates over a dictionary of parameters, filters out keys listed
    in the `exclusion` list, and saves the remaining parameters to a TOML file
    specified by `file_path`.

    Args:
        parameters (dict): Dictionary containing the configuration parameters.
        file_path (str): Path to the file where the filtered parameters should be saved.
        exclusion (list): List of keys to exclude from saving. Defaults to ["file_path", "save_as", "headless", "print_only"].
    """
    variables = {
        name: value
        for name, value in sorted(parameters, key=lambda x: x[0])
        if name not in exclusion
    }

    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        log.info(f"Creating folder {folder_path} for the configuration file...")

    with open(file_path, "w", encoding="utf-8") as file:
        toml.dump(variables, file)


def SaveConfigFileToRun(
    parameters,
    file_path: str,
    exclusion: list = ["file_path", "save_as", "headless", "print_only"],
) -> None:
    """
    Saves the configuration parameters to a TOML file, excluding specified keys.

    This function iterates over a dictionary of parameters, filters out keys listed
    in the `exclusion` list, and saves the remaining parameters to a TOML file
    specified by `file_path`.

    Args:
        parameters (dict): Dictionary containing the configuration parameters.
        file_path (str): Path to the file where the filtered parameters should be saved.
        exclusion (list): List of keys to exclude from saving. Defaults to ["file_path", "save_as", "headless", "print_only"].
    """
    variables = {
        name: value
        for name, value in sorted(parameters, key=lambda x: x[0])
        if name not in exclusion and value != 0 and value != "" and value is not False
    }

    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        log.info(f"Creating folder {folder_path} for the configuration file...")

    with open(file_path, "w", encoding="utf-8") as file:
        toml.dump(variables, file)


def save_to_file(content):
    """
    Appends the given content to a file named 'print_command.txt' within a 'logs' directory.

    This function checks for the existence of a 'logs' directory and creates it if
    it doesn't exist. Then, it appends the provided content along with a newline character
    to the 'print_command.txt' file within this directory.

    Args:
        content (str): The content to be saved to the file.
    """
    logs_directory = "logs"
    file_path = os.path.join(logs_directory, "print_command.txt")

    # Ensure the 'logs' directory exists
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    # Append content to the specified file
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(content + "\n")
    except IOError as e:
        print(f"Error: Could not write to file - {e}")
    except OSError as e:
        print(f"Error: Could not create 'logs' directory - {e}")


def validate_file_path(file_path: str) -> bool:
    if file_path == "":
        return True
    msg = f"Validating {file_path} existence..."
    if not os.path.isfile(file_path):
        log.error(f"{msg} FAILED: does not exist")
        return False
    log.info(f"{msg} SUCCESS")
    return True


def validate_folder_path(
    folder_path: str,
    can_be_written_to: bool = False,
    create_if_not_exists: bool = False,
) -> bool:
    if folder_path == "":
        return True
    msg = f"Validating {folder_path} existence{' and writability' if can_be_written_to else ''}..."
    if not os.path.isdir(folder_path):
        if create_if_not_exists:
            os.makedirs(folder_path)
            log.info(f"{msg} SUCCESS")
            return True
        else:
            log.error(f"{msg} FAILED: does not exist")
            return False
    if can_be_written_to and not os.access(folder_path, os.W_OK):
        log.error(f"{msg} FAILED: is not writable.")
        return False
    log.info(f"{msg} SUCCESS")
    return True


def validate_toml_file(file_path: str) -> bool:
    if file_path == "":
        return True
    msg = f"Validating toml {file_path} existence and validity..."
    if not os.path.isfile(file_path):
        log.error(f"{msg} FAILED: does not exist")
        return False

    try:
        toml.load(file_path)
    except:
        log.error(f"{msg} FAILED: is not a valid toml file.")
        return False
    log.info(f"{msg} SUCCESS")
    return True


def print_command_and_toml(run_cmd, tmpfilename=""):
    log.warning(
        "Here is the trainer command as a reference. It will not be executed:\n"
    )
    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)

    print(command_to_run)
    print("")

    if tmpfilename != "":
        log.info(f"Showing toml config file: {tmpfilename}")
        print("")
        with open(tmpfilename, "r", encoding="utf-8") as toml_file:
            log.info(toml_file.read())
        log.info(f"end of toml config file: {tmpfilename}")

        save_to_file(command_to_run)


def validate_args_setting(input_string):
    # Regex pattern to handle multiple conditions:
    # - Empty string is valid
    # - Single or multiple key/value pairs with exactly one space between pairs
    # - No spaces around '=' and no spaces within keys or values
    pattern = r"^(\S+=\S+)( \S+=\S+)*$|^$"
    if re.match(pattern, input_string):
        return True
    else:
        log.info(f"'{input_string}' is not a valid settings string.")
        log.info(
            "A valid settings string must consist of one or more key/value pairs formatted as key=value, with no spaces around the equals sign or within the value. Multiple pairs should be separated by a space."
        )
        return False


def setup_environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if os.name == "nt":
        env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    return env
