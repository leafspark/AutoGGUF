from PySide6.QtCore import QTimer
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QFileDialog, QLabel

from Localizations import *
import psutil
from error_handling import show_error


def browse_base_model(self) -> None:
    self.logger.info(BROWSING_FOR_BASE_MODEL_FOLDER)  # Updated log message
    base_model_folder = QFileDialog.getExistingDirectory(self, SELECT_BASE_MODEL_FOLDER)
    if base_model_folder:
        self.base_model_path.setText(os.path.abspath(base_model_folder))


def browse_hf_model_input(self) -> None:
    self.logger.info(BROWSE_FOR_HF_MODEL_DIRECTORY)
    model_dir = QFileDialog.getExistingDirectory(self, SELECT_HF_MODEL_DIRECTORY)
    if model_dir:
        self.hf_model_input.setText(os.path.abspath(model_dir))


def browse_hf_outfile(self) -> None:
    self.logger.info(BROWSE_FOR_HF_TO_GGUF_OUTPUT)
    outfile, _ = QFileDialog.getSaveFileName(self, SELECT_OUTPUT_FILE, "", GGUF_FILES)
    if outfile:
        self.hf_outfile.setText(os.path.abspath(outfile))


def browse_imatrix_datafile(self) -> None:
    self.logger.info(BROWSING_FOR_IMATRIX_DATA_FILE)
    datafile, _ = QFileDialog.getOpenFileName(self, SELECT_DATA_FILE, "", ALL_FILES)
    if datafile:
        self.imatrix_datafile.setText(os.path.abspath(datafile))


def browse_imatrix_model(self) -> None:
    self.logger.info(BROWSING_FOR_IMATRIX_MODEL_FILE)
    model_file, _ = QFileDialog.getOpenFileName(self, SELECT_MODEL_FILE, "", GGUF_FILES)
    if model_file:
        self.imatrix_model.setText(os.path.abspath(model_file))


def browse_imatrix_output(self) -> None:
    self.logger.info(BROWSING_FOR_IMATRIX_OUTPUT_FILE)
    output_file, _ = QFileDialog.getSaveFileName(
        self, SELECT_OUTPUT_FILE, "", DAT_FILES
    )
    if output_file:
        self.imatrix_output.setText(os.path.abspath(output_file))


def create_label(self, text, tooltip) -> QLabel:
    label = QLabel(text)
    label.setToolTip(tooltip)
    return label


def toggle_gpu_offload_auto(self, state) -> None:
    is_auto = state == Qt.CheckState.Checked
    self.gpu_offload_slider.setEnabled(not is_auto)
    self.gpu_offload_spinbox.setEnabled(not is_auto)


def update_model_info(logger, model_info) -> None:
    logger.debug(UPDATING_MODEL_INFO.format(model_info))
    pass


def update_system_info(self) -> None:
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent()

    # Smooth transition for RAM bar
    animate_bar(self, self.ram_bar, ram.percent)

    # Smooth transition for CPU bar
    animate_bar(self, self.cpu_bar, cpu)

    self.ram_bar.setFormat(
        RAM_USAGE_FORMAT.format(
            ram.percent, ram.used // 1024 // 1024, ram.total // 1024 // 1024
        )
    )
    self.cpu_label.setText(CPU_USAGE_FORMAT.format(cpu))

    # Collect CPU and RAM usage data
    self.cpu_data.append(cpu)
    self.ram_data.append(ram.percent)

    if len(self.cpu_data) > 60:
        self.cpu_data.pop(0)
        self.ram_data.pop(0)


def animate_bar(self, bar, target_value) -> None:
    current_value = bar.value()
    difference = target_value - current_value

    if abs(difference) <= 1:  # Avoid animation for small changes
        bar.setValue(target_value)
        return

    step = 1 if difference > 0 else -1  # Increment or decrement based on difference
    timer = QTimer(self)
    timer.timeout.connect(lambda: _animate_step(bar, target_value, step, timer))
    timer.start(10)  # Adjust the interval for animation speed


def _animate_step(bar, target_value, step, timer) -> None:
    current_value = bar.value()
    new_value = current_value + step

    if (step > 0 and new_value > target_value) or (
        step < 0 and new_value < target_value
    ):
        bar.setValue(target_value)
        timer.stop()
    else:
        bar.setValue(new_value)


def update_download_progress(self, progress) -> None:
    self.download_progress.setValue(progress)


def update_cuda_backends(self) -> None:
    self.logger.debug(UPDATING_CUDA_BACKENDS)
    self.backend_combo_cuda.clear()
    llama_bin = os.path.abspath("llama_bin")
    if os.path.exists(llama_bin):
        for item in os.listdir(llama_bin):
            item_path = os.path.join(llama_bin, item)
            if os.path.isdir(item_path) and "cudart-llama" not in item.lower():
                if "cu1" in item.lower():  # Only include CUDA-capable backends
                    self.backend_combo_cuda.addItem(item, userData=item_path)

    if self.backend_combo_cuda.count() == 0:
        self.backend_combo_cuda.addItem(NO_SUITABLE_CUDA_BACKENDS)
        self.backend_combo_cuda.setEnabled(False)
    else:
        self.backend_combo_cuda.setEnabled(True)


def update_threads_spinbox(self, value) -> None:
    self.threads_spinbox.setValue(value)


def update_threads_slider(self, value) -> None:
    self.threads_slider.setValue(value)


def update_gpu_offload_spinbox(self, value) -> None:
    self.gpu_offload_spinbox.setValue(value)


def update_gpu_offload_slider(self, value) -> None:
    self.gpu_offload_slider.setValue(value)


def update_cuda_option(self) -> None:
    self.logger.debug(UPDATING_CUDA_OPTIONS)
    asset = self.asset_combo.currentData()

    # Handle the case where asset is None
    if asset is None:
        self.logger.warning(NO_ASSET_SELECTED_FOR_CUDA_CHECK)
        self.cuda_extract_checkbox.setVisible(False)
        self.cuda_backend_label.setVisible(False)
        self.backend_combo_cuda.setVisible(False)
        return  # Exit the function early

    is_cuda = asset and "cudart" in asset["name"].lower()
    self.cuda_extract_checkbox.setVisible(is_cuda)
    self.cuda_backend_label.setVisible(is_cuda)
    self.backend_combo_cuda.setVisible(is_cuda)
    if is_cuda:
        self.update_cuda_backends()


def update_assets(self) -> None:
    self.logger.debug(UPDATING_ASSET_LIST)
    self.asset_combo.clear()
    release = self.release_combo.currentData()
    if release:
        if "assets" in release:
            for asset in release["assets"]:
                self.asset_combo.addItem(asset["name"], userData=asset)
        else:
            show_error(
                self.logger, NO_ASSETS_FOUND_FOR_RELEASE.format(release["tag_name"])
            )
    self.update_cuda_option()


def update_base_model_visibility(self, index) -> None:
    is_gguf = self.lora_output_type_combo.itemText(index) == "GGUF"
    self.base_model_wrapper.setVisible(is_gguf)
