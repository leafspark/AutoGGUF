from PySide6.QtCore import QTimer

from Localizations import *
import psutil
from error_handling import show_error


def update_model_info(logger, self, model_info) -> None:
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
