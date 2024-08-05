class AutoGGUF(QMainWindow):
    """
    AutoGGUF is a PyQt6-based graphical user interface for managing and quantizing large language models.
    
    This class provides functionality for:
    - Loading and displaying models (including sharded models)
    - Quantizing models with various options
    - Downloading llama.cpp releases
    - Generating importance matrices
    - Converting and exporting LoRA models
    - Managing quantization tasks
    
    The GUI allows users to interact with these features in an intuitive way, providing
    options for model selection, quantization parameters, and task management.
    """

    def __init__(self):
        """
        Initialize the AutoGGUF application window.

        This method sets up the main window, initializes the UI components,
        sets up layouts, and connects various signals to their respective slots.
        It also initializes the logger and sets up the system info update timer.
        """

    def refresh_backends(self):
        """
        Refresh the list of available llama.cpp backends.

        This method scans the 'llama_bin' directory for valid backends,
        populates the backend combo box with the found backends, and
        enables/disables the combo box based on the availability of backends.
        """

    def update_base_model_visibility(self, index):
        """
        Update the visibility of the base model selection widgets.

        Args:
            index (int): The current index of the LoRA output type combo box.

        This method shows or hides the base model selection widgets based on
        whether the selected LoRA output type is GGUF or not.
        """

    def save_preset(self):
        """
        Save the current quantization settings as a preset.

        This method collects all the current quantization settings and saves
        them to a JSON file. The user is prompted to choose the save location.
        """

    def load_preset(self):
        """
        Load a previously saved quantization preset.

        This method prompts the user to select a preset JSON file and then
        applies the saved settings to the current UI state.
        """

    def save_task_preset(self, task_item):
        """
        Save the settings of a specific quantization task as a preset.

        Args:
            task_item (TaskListItem): The task item to save as a preset.

        This method saves the command, backend path, and log file of the
        specified task to a JSON file.
        """

    def browse_export_lora_model(self):
        """
        Open a file dialog to select a model file for LoRA export.

        This method updates the LoRA model input field with the selected file path.
        """

    def browse_export_lora_output(self):
        """
        Open a file dialog to select an output file for LoRA export.

        This method updates the LoRA output input field with the selected file path.
        """

    def add_lora_adapter(self):
        """
        Add a new LoRA adapter to the list of adapters for export.

        This method opens a file dialog to select a LoRA adapter file and adds
        it to the list with an associated scale input.
        """

    def browse_base_model(self):
        """
        Open a file dialog to select a base model folder for LoRA conversion.

        This method updates the base model path input field with the selected folder path.
        """

    def delete_lora_adapter_item(self, adapter_widget):
        """
        Remove a LoRA adapter item from the list of adapters.

        Args:
            adapter_widget (QWidget): The widget representing the adapter to be removed.
        """

    def export_lora(self):
        """
        Start the LoRA export process.

        This method collects all the necessary information for LoRA export,
        constructs the export command, and starts a new thread to run the export process.
        """

    def load_models(self):
        """
        Load and display the available models in the model tree.

        This method scans the models directory, detects sharded and single models,
        and populates the model tree widget with the found models.
        """

    def quantize_model(self):
        """
        Start the model quantization process.

        This method collects all the quantization settings, constructs the
        quantization command, and starts a new thread to run the quantization process.
        """

    def update_model_info(self, model_info):
        """
        Update the model information.

        Args:
            model_info (dict): A dictionary containing updated model information.

        This method is a placeholder for future functionality to update and display
        model information during or after quantization.
        """

    def parse_progress(self, line, task_item):
        """
        Parse the output of the quantization process to update the progress bar.

        Args:
            line (str): A line of output from the quantization process.
            task_item (TaskListItem): The task item to update.

        This method looks for progress information in the output and updates
        the task item's progress bar accordingly.
        """

    def task_finished(self, thread):
        """
        Handle the completion of a quantization task.

        Args:
            thread (QuantizationThread): The thread that has finished.

        This method removes the finished thread from the list of active threads.
        """

    def show_task_details(self, item):
        """
        Display the details of a quantization task.

        Args:
            item (QListWidgetItem): The list item representing the task.

        This method opens a dialog showing the log file contents for the selected task.
        """

    def browse_imatrix_datafile(self):
        """
        Open a file dialog to select a data file for importance matrix generation.

        This method updates the imatrix data file input field with the selected file path.
        """

    def browse_imatrix_model(self):
        """
        Open a file dialog to select a model file for importance matrix generation.

        This method updates the imatrix model input field with the selected file path.
        """

    def browse_imatrix_output(self):
        """
        Open a file dialog to select an output file for importance matrix generation.

        This method updates the imatrix output input field with the selected file path.
        """

    def update_gpu_offload_spinbox(self, value):
        """
        Update the GPU offload spinbox value.

        Args:
            value (int): The new value for the GPU offload spinbox.
        """

    def update_gpu_offload_slider(self, value):
        """
        Update the GPU offload slider value.

        Args:
            value (int): The new value for the GPU offload slider.
        """

    def toggle_gpu_offload_auto(self, state):
        """
        Toggle the automatic GPU offload option.

        Args:
            state (Qt.CheckState): The new state of the auto checkbox.

        This method enables or disables the GPU offload slider and spinbox
        based on the state of the auto checkbox.
        """

    def generate_imatrix(self):
        """
        Start the importance matrix generation process.

        This method collects all the necessary information for imatrix generation,
        constructs the generation command, and starts a new thread to run the process.
        """

    def show_error(self, message):
        """
        Display an error message to the user.

        Args:
            message (str): The error message to display.

        This method logs the error and shows a message box with the error details.
        """

    def handle_error(self, error_message, task_item, task_exists=True):
        """
        Handle an error that occurred during a task.

        Args:
            error_message (str): The error message.
            task_item (TaskListItem): The task item associated with the error.
            task_exists (bool): Whether the task still exists in the UI.

        This method logs the error, displays it to the user, and updates the
        task item's status if it still exists.
        """

    def closeEvent(self, event: QCloseEvent):
        """
        Handle the window close event.

        Args:
            event (QCloseEvent): The close event.

        This method checks if there are any running tasks before closing the
        application. If tasks are running, it prompts the user for confirmation.
        """

    def refresh_releases(self):
        """
        Refresh the list of available llama.cpp releases from GitHub.

        This method fetches the latest releases from the llama.cpp GitHub repository
        and populates the release combo box with the fetched information.
        """

    def update_assets(self):
        """
        Update the list of assets for the selected llama.cpp release.

        This method populates the asset combo box with the available assets
        for the currently selected release.
        """

    def download_llama_cpp(self):
        """
        Start the download process for the selected llama.cpp asset.

        This method initiates the download of the selected llama.cpp asset,
        sets up a progress bar, and manages the download thread.
        """

    def update_cuda_option(self):
        """
        Update the visibility and options for CUDA-related widgets.

        This method shows or hides CUDA-related options based on whether
        the selected asset is CUDA-compatible.
        """

    def update_cuda_backends(self):
        """
        Update the list of available CUDA-capable backends.

        This method populates the CUDA backend combo box with the available
        CUDA-capable backends found in the llama_bin directory.
        """

    def update_download_progress(self, progress):
        """
        Update the download progress bar.

        Args:
            progress (int): The current progress percentage.

        This method updates the download progress bar with the given percentage.
        """

    def download_finished(self, extract_dir):
        """
        Handle the completion of a llama.cpp download.

        Args:
            extract_dir (str): The directory where the download was extracted.

        This method updates the UI after a successful download, handles CUDA
        file extraction if applicable, and refreshes the backend list.
        """

    def extract_cuda_files(self, extract_dir, destination):
        """
        Extract CUDA-related files from the downloaded archive.

        Args:
            extract_dir (str): The directory containing the extracted files.
            destination (str): The destination directory for CUDA files.

        This method copies CUDA-related DLL files to the specified destination.
        """

    def download_error(self, error_message):
        """
        Handle errors that occur during the llama.cpp download process.

        Args:
            error_message (str): The error message describing the download failure.

        This method displays the error message, resets the download UI elements,
        and cleans up any partially downloaded files.
        """

    def show_task_context_menu(self, position):
        """
        Display a context menu for a task in the task list.

        Args:
            position (QPoint): The position where the context menu should be displayed.

        This method creates and shows a context menu with options to view properties,
        cancel, restart, save preset, or delete a task.
        """

    def show_task_properties(self, item):
        """
        Display the properties of a quantization task.

        Args:
            item (QListWidgetItem): The list item representing the task.

        This method opens a dialog showing detailed information about the selected task.
        """

    def update_threads_spinbox(self, value):
        """
        Update the threads spinbox value.

        Args:
            value (int): The new value for the threads spinbox.
        """

    def update_threads_slider(self, value):
        """
        Update the threads slider value.

        Args:
            value (int): The new value for the threads slider.
        """

    def cancel_task(self, item):
        """
        Cancel a running quantization task.

        Args:
            item (QListWidgetItem): The list item representing the task to cancel.

        This method terminates the thread associated with the task and updates its status.
        """

    def retry_task(self, item):
        """
        Retry a failed or canceled quantization task.

        Args:
            item (QListWidgetItem): The list item representing the task to retry.

        This method is a placeholder for future implementation of task retry functionality.
        """

    def delete_task(self, item):
        """
        Delete a task from the task list.

        Args:
            item (QListWidgetItem): The list item representing the task to delete.

        This method removes the task from the UI and terminates any associated thread.
        """

    def create_label(self, text, tooltip):
        """
        Create a QLabel with text and tooltip.

        Args:
            text (str): The text to display on the label.
            tooltip (str): The tooltip text for the label.

        Returns:
            QLabel: A new QLabel instance with the specified text and tooltip.
        """

    def browse_models(self):
        """
        Open a file dialog to select the models directory.

        This method updates the models input field with the selected directory path
        and reloads the list of available models.
        """

    def browse_output(self):
        """
        Open a file dialog to select the output directory for quantized models.

        This method updates the output input field with the selected directory path.
        """

    def browse_logs(self):
        """
        Open a file dialog to select the logs directory.

        This method updates the logs input field with the selected directory path.
        """

    def browse_imatrix(self):
        """
        Open a file dialog to select an importance matrix file.

        This method updates the imatrix input field with the selected file path.
        """

    def update_system_info(self):
        """
        Update the displayed system information (RAM and CPU usage).

        This method is called periodically to refresh the RAM usage bar and CPU usage label.
        """

    def validate_quantization_inputs(self):
        """
        Validate the inputs required for model quantization.

        This method checks if all necessary inputs for quantization are provided
        and raises a ValueError with details of any missing inputs.
        """

    def add_kv_override(self, override_string=None):
        """
        Add a new key-value override entry to the UI.

        Args:
            override_string (str, optional): A string representation of an existing override.

        This method adds a new KVOverrideEntry widget to the UI, optionally populating
        it with values from the provided override string.
        """

    def remove_kv_override(self, entry):
        """
        Remove a key-value override entry from the UI.

        Args:
            entry (KVOverrideEntry): The entry widget to remove.

        This method removes the specified KVOverrideEntry widget from the UI.
        """
