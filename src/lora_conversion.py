from datetime import datetime

from PySide6.QtWidgets import QListWidgetItem

from QuantizationThread import QuantizationThread
from TaskListItem import TaskListItem
from error_handling import handle_error, show_error
from imports_and_globals import ensure_directory
from Localizations import *


def convert_lora(self):
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
            command = ["python", "src/convert_lora_to_ggml.py", lora_input_path]

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
        thread.finished_signal.connect(
            lambda: self.lora_conversion_finished(
                thread, lora_input_path, lora_output_path
            )
        )
        thread.error_signal.connect(
            lambda err: handle_error(self.logger, err, task_item)
        )
        thread.start()
        self.logger.info(LORA_CONVERSION_TASK_STARTED)
    except ValueError as e:
        show_error(self.logger, str(e))
    except Exception as e:
        show_error(self.logger, ERROR_STARTING_LORA_CONVERSION.format(str(e)))
