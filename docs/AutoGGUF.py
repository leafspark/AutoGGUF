import importlib
import json
import re
import shutil
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Tuple

import requests
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from dotenv import load_dotenv

import lora_conversion
import presets
import ui_update
import utils
from CustomTitleBar import CustomTitleBar
from GPUMonitor import GPUMonitor
from Localizations import *
from Logger import Logger
from QuantizationThread import QuantizationThread
from TaskListItem import TaskListItem
from error_handling import handle_error, show_error
from imports_and_globals import (
    ensure_directory,
    open_file_safe,
    resource_path,
    show_about,
)


class CustomTitleBar(QWidget):
    """
    Custom title bar for the main window, providing drag-and-drop functionality
    and minimize/close buttons.
    """

    def __init__(self, parent=None):
        """
        Initializes the custom title bar.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """


class AutoGGUF(QMainWindow):
    """
    Main application window for AutoGGUF, providing a user interface for
    quantizing and converting large language models.
    """

    def __init__(self):
        """
        Initializes the main window, setting up the UI, logger, and other
        necessary components.
        """

    def keyPressEvent(self, event):
        """
        Handles key press events for window resizing.

        Args:
            event (QKeyEvent): The key press event.
        """

    def resize_window(self, larger):
        """
        Resizes the window by a specified factor.

        Args:
            larger (bool): Whether to make the window larger or smaller.
        """

    def reset_size(self):
        """Resets the window to its default size."""

    def parse_resolution(self):
        """
        Parses the resolution from the AUTOGGUF_RESOLUTION environment variable.

        Returns:
            tuple: The width and height of the window.
        """

    def resizeEvent(self, event):
        """
        Handles resize events to maintain rounded corners.

        Args:
            event (QResizeEvent): The resize event.
        """

    def refresh_backends(self):
        """Refreshes the list of available backends."""

    def save_task_preset(self, task_item):
        """
        Saves the preset for a specific task.

        Args:
            task_item (TaskListItem): The task item to save the preset for.
        """

    def browse_export_lora_model(self):
        """Opens a file dialog to browse for the export LORA model file."""

    def browse_export_lora_output(self):
        """Opens a file dialog to browse for the export LORA output file."""

    def add_lora_adapter(self):
        """Adds a LORA adapter to the export LORA list."""

    def browse_base_model(self):
        """Opens a file dialog to browse for the base model folder."""

    def delete_lora_adapter_item(self, adapter_widget):
        """
        Deletes a LORA adapter item from the export LORA list.

        Args:
            adapter_widget (QWidget): The widget containing the adapter information.
        """

    def browse_hf_model_input(self):
        """Opens a file dialog to browse for the HuggingFace model directory."""

    def browse_hf_outfile(self):
        """Opens a file dialog to browse for the HuggingFace to GGUF output file."""

    def convert_hf_to_gguf(self):
        """Converts a HuggingFace model to GGUF format."""

    def export_lora(self):
        """Exports a LORA from a GGML model."""

    def restart_task(self, task_item):
        """
        Restarts a specific task.

        Args:
            task_item (TaskListItem): The task item to restart.
        """

    def lora_conversion_finished(self, thread, input_path, output_path):
        """
        Handles the completion of a LORA conversion task.

        Args:
            thread (QuantizationThread): The thread that handled the conversion.
            input_path (str): The path to the input LORA file.
            output_path (str): The path to the output GGML file.
        """

    def download_finished(self, extract_dir):
        """
        Handles the completion of a download, extracting files and updating the UI.

        Args:
            extract_dir (str): The directory where the downloaded files were extracted.
        """

    def extract_cuda_files(self, extract_dir, destination):
        """
        Extracts CUDA files from a downloaded archive.

        Args:
            extract_dir (str): The directory where the downloaded files were extracted.
            destination (str): The destination directory for the CUDA files.
        """

    def download_error(self, error_message):
        """
        Handles download errors, displaying an error message and cleaning up.

        Args:
            error_message (str): The error message.
        """

    def show_task_context_menu(self, position):
        """
        Shows the context menu for a task item in the task list.

        Args:
            position (QPoint): The position of the context menu.
        """

    def show_task_properties(self, item):
        """
        Shows the properties dialog for a specific task.

        Args:
            item (QListWidgetItem): The task item.
        """

    def toggle_gpu_offload_auto(self, state):
        """
        Toggles the automatic GPU offload option.

        Args:
            state (Qt.CheckState): The state of the checkbox.
        """

    def cancel_task_by_item(self, item):
        """
        Cancels a task by its item in the task list.

        Args:
            item (QListWidgetItem): The task item.
        """

    def cancel_task(self, item):
        """
        Cancels a specific task.

        Args:
            item (QListWidgetItem): The task item.
        """

    def delete_task(self, item):
        """
        Deletes a specific task.

        Args:
            item (QListWidgetItem): The task item.
        """

    def create_label(self, text, tooltip):
        """
        Creates a QLabel with a tooltip.

        Args:
            text (str): The text for the label.
            tooltip (str): The tooltip for the label.

        Returns:
            QLabel: The created label.
        """

    def load_models(self):
        """Loads the available models and displays them in the model tree."""

    def browse_models(self):
        """Opens a file dialog to browse for the models directory."""

    def browse_output(self):
        """Opens a file dialog to browse for the output directory."""

    def browse_logs(self):
        """Opens a file dialog to browse for the logs directory."""

    def browse_imatrix(self):
        """Opens a file dialog to browse for the imatrix file."""

    def validate_quantization_inputs(self):
        """Validates the inputs for quantization."""

    def add_kv_override(self, override_string=None):
        """Adds a KV override entry to the list."""

    def remove_kv_override(self, entry):
        """Removes a KV override entry from the list."""

    def quantize_model(self):
        """Quantizes the selected model."""

    def parse_progress(self, line, task_item):
        """
        Parses the progress from the output line and updates the task item.

        Args:
            line (str): The output line.
            task_item (TaskListItem): The task item.
        """

    def task_finished(self, thread, task_item):
        """
        Handles the completion of a task.

        Args:
            thread (QuantizationThread): The thread that handled the task.
            task_item (TaskListItem): The task item.
        """

    def show_task_details(self, item):
        """
        Shows the details of a specific task.

        Args:
            item (QListWidgetItem): The task item.
        """

    def browse_imatrix_datafile(self):
        """Opens a file dialog to browse for the imatrix data file."""

    def browse_imatrix_model(self):
        """Opens a file dialog to browse for the imatrix model file."""

    def browse_imatrix_output(self):
        """Opens a file dialog to browse for the imatrix output file."""

    def get_models_data(self):
        """Retrieves data for all loaded models."""

    def get_tasks_data(self):
        """Retrieves data for all tasks in the task list."""

    def generate_imatrix(self):
        """Generates an imatrix file."""

    def closeEvent(self, event: QCloseEvent):
        """
        Handles close events, prompting the user if there are running tasks.

        Args:
            event (QCloseEvent): The close event.
        """
