from datetime import datetime

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from QuantizationThread import QuantizationThread
from TaskListItem import TaskListItem
from error_handling import handle_error, show_error
from globals import ensure_directory
from Localizations import *


def export_lora(self) -> None:
    self.logger.info(STARTING_LORA_EXPORT)
    try:
        model_path = self.export_lora_model.text()
        output_path = self.export_lora_output.text()
        lora_adapters = []

        for i in range(self.export_lora_adapters.count()):
            item = self.export_lora_adapters.item(i)
            adapter_widget = self.export_lora_adapters.itemWidget(item)
            path_input = adapter_widget.layout().itemAt(0).widget()
            scale_input = adapter_widget.layout().itemAt(1).widget()
            adapter_path = path_input.text()
            adapter_scale = scale_input.text()
            lora_adapters.append((adapter_path, adapter_scale))

        if not model_path:
            raise ValueError(MODEL_PATH_REQUIRED)
        if not output_path:
            raise ValueError(OUTPUT_PATH_REQUIRED)
        if not lora_adapters:
            raise ValueError(AT_LEAST_ONE_LORA_ADAPTER_REQUIRED)

        backend_path = self.backend_combo.currentData()
        if not backend_path:
            raise ValueError(NO_BACKEND_SELECTED)

        command = [
            os.path.join(backend_path, "llama-export-lora"),
            "--model",
            model_path,
            "--output",
            output_path,
        ]

        for adapter_path, adapter_scale in lora_adapters:
            if adapter_path:
                if adapter_scale:
                    try:
                        scale_value = float(adapter_scale)
                        command.extend(
                            ["--lora-scaled", adapter_path, str(scale_value)]
                        )
                    except ValueError:
                        raise ValueError(INVALID_LORA_SCALE_VALUE)
                else:
                    command.extend(["--lora", adapter_path])

        threads = self.export_lora_threads.value()
        command.extend(["--threads", str(threads)])

        logs_path = self.logs_input.text()
        ensure_directory(logs_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_path, f"lora_export_{timestamp}.log")

        command_str = " ".join(command)
        self.logger.info(f"{LORA_EXPORT_COMMAND}: {command_str}")

        thread = QuantizationThread(command, backend_path, log_file)
        self.quant_threads.append(thread)

        task_item = TaskListItem(EXPORTING_LORA, log_file, show_progress_bar=False)
        list_item = QListWidgetItem(self.task_list)
        list_item.setSizeHint(task_item.sizeHint())
        self.task_list.addItem(list_item)
        self.task_list.setItemWidget(list_item, task_item)

        thread.status_signal.connect(task_item.update_status)
        thread.finished_signal.connect(lambda: self.task_finished(thread))
        thread.error_signal.connect(
            lambda err: handle_error(self.logger, err, task_item)
        )
        thread.start()
        self.logger.info(LORA_EXPORT_TASK_STARTED)
    except ValueError as e:
        show_error(self.logger, str(e))
    except Exception as e:
        show_error(self.logger, ERROR_STARTING_LORA_EXPORT.format(str(e)))


def lora_conversion_finished(self, thread) -> None:
    self.logger.info(LORA_CONVERSION_FINISHED)
    if thread in self.quant_threads:
        self.quant_threads.remove(thread)


def delete_lora_adapter_item(self, adapter_widget) -> None:
    self.logger.info(DELETING_LORA_ADAPTER)
    # Find the QListWidgetItem containing the adapter_widget
    for i in range(self.export_lora_adapters.count()):
        item = self.export_lora_adapters.item(i)
        if self.export_lora_adapters.itemWidget(item) == adapter_widget:
            self.export_lora_adapters.takeItem(i)  # Remove the item
            break


def browse_export_lora_model(self) -> None:
    self.logger.info(BROWSING_FOR_EXPORT_LORA_MODEL_FILE)
    model_file, _ = QFileDialog.getOpenFileName(self, SELECT_MODEL_FILE, "", GGUF_FILES)
    if model_file:
        self.export_lora_model.setText(os.path.abspath(model_file))


def browse_export_lora_output(self) -> None:
    self.logger.info(BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE)
    output_file, _ = QFileDialog.getSaveFileName(
        self, SELECT_OUTPUT_FILE, "", GGUF_FILES
    )
    if output_file:
        self.export_lora_output.setText(os.path.abspath(output_file))


def add_lora_adapter(self) -> None:
    self.logger.info(ADDING_LORA_ADAPTER)
    adapter_path, _ = QFileDialog.getOpenFileName(
        self, SELECT_LORA_ADAPTER_FILE, "", LORA_FILES
    )
    if adapter_path:
        # Create a widget to hold the path and scale input
        adapter_widget = QWidget()
        adapter_layout = QHBoxLayout(adapter_widget)

        path_input = QLineEdit(adapter_path)
        path_input.setReadOnly(True)
        adapter_layout.addWidget(path_input)

        scale_input = QLineEdit("1.0")  # Default scale value
        adapter_layout.addWidget(scale_input)

        delete_button = QPushButton(DELETE_ADAPTER)
        delete_button.clicked.connect(
            lambda: self.delete_lora_adapter_item(adapter_widget)
        )
        adapter_layout.addWidget(delete_button)

        # Add the widget to the list
        list_item = QListWidgetItem(self.export_lora_adapters)
        list_item.setSizeHint(adapter_widget.sizeHint())
        self.export_lora_adapters.addItem(list_item)
        self.export_lora_adapters.setItemWidget(list_item, adapter_widget)


def convert_lora(self) -> None:
    self.logger.info(STARTING_LORA_CONVERSION)
    try:
        lora_input_path = self.lora_input.text()
        lora_output_path = self.lora_output.text()
        lora_output_type = self.lora_output_type_combo.currentText()

        if not lora_input_path:
            raise ValueError(LORA_INPUT_PATH_REQUIRED)
        if not lora_output_path:
            raise ValueError(LORA_OUTPUT_PATH_REQUIRED)

        if lora_output_type == "GGUF":  # Use new file and parameters for GGUF
            command = [
                "python",
                "src/convert_lora_to_gguf.py",
                "--outfile",
                lora_output_path,
                lora_input_path,
            ]
            base_model_path = self.base_model_path.text()
            if not base_model_path:
                raise ValueError(BASE_MODEL_PATH_REQUIRED)
            command.extend(["--base", base_model_path])
        else:  # Use old GGML parameters for GGML
            command = [
                "python",
                "src/convert_lora_to_ggml.py",
                lora_input_path,
                lora_output_path,
            ]

        logs_path = self.logs_input.text()
        ensure_directory(logs_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_path, f"lora_conversion_{timestamp}.log")

        command_str = " ".join(command)
        self.logger.info(f"{LORA_CONVERSION_COMMAND}: {command_str}")

        thread = QuantizationThread(command, os.getcwd(), log_file)
        self.quant_threads.append(thread)

        task_name = LORA_CONVERSION_FROM_TO.format(
            os.path.basename(lora_input_path), os.path.basename(lora_output_path)
        )
        task_item = TaskListItem(task_name, log_file, show_progress_bar=False)
        list_item = QListWidgetItem(self.task_list)
        list_item.setSizeHint(task_item.sizeHint())
        self.task_list.addItem(list_item)
        self.task_list.setItemWidget(list_item, task_item)

        thread.status_signal.connect(task_item.update_status)
        thread.finished_signal.connect(lambda: self.lora_conversion_finished(thread))
        thread.error_signal.connect(
            lambda err: handle_error(self.logger, err, task_item)
        )
        thread.start()
        self.logger.info(LORA_CONVERSION_TASK_STARTED)
    except ValueError as e:
        show_error(self.logger, str(e))
    except Exception as e:
        show_error(self.logger, ERROR_STARTING_LORA_CONVERSION.format(str(e)))
