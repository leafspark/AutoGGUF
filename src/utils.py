from PySide6.QtWidgets import QFileDialog

from error_handling import show_error
from Localizations import *
import requests

from DownloadThread import DownloadThread
from imports_and_globals import ensure_directory


def browse_lora_input(self):
    self.logger.info(BROWSING_FOR_LORA_INPUT_DIRECTORY)
    lora_input_path = QFileDialog.getExistingDirectory(
        self, SELECT_LORA_INPUT_DIRECTORY
    )
    if lora_input_path:
        self.lora_input.setText(os.path.abspath(lora_input_path))
        ensure_directory(lora_input_path)


def browse_lora_output(self):
    self.logger.info(BROWSING_FOR_LORA_OUTPUT_FILE)
    lora_output_file, _ = QFileDialog.getSaveFileName(
        self, SELECT_LORA_OUTPUT_FILE, "", GGUF_AND_BIN_FILES
    )
    if lora_output_file:
        self.lora_output.setText(os.path.abspath(lora_output_file))


def download_llama_cpp(self):
    self.logger.info(STARTING_LLAMACPP_DOWNLOAD)
    asset = self.asset_combo.currentData()
    if not asset:
        show_error(self.logger, NO_ASSET_SELECTED)
        return

    llama_bin = os.path.abspath("llama_bin")
    os.makedirs(llama_bin, exist_ok=True)

    save_path = os.path.join(llama_bin, asset["name"])

    self.download_thread = DownloadThread(asset["browser_download_url"], save_path)
    self.download_thread.progress_signal.connect(self.update_download_progress)
    self.download_thread.finished_signal.connect(self.download_finished)
    self.download_thread.error_signal.connect(self.download_error)
    self.download_thread.start()

    self.download_button.setEnabled(False)
    self.download_progress.setValue(0)


def refresh_releases(self):
    self.logger.info(REFRESHING_LLAMACPP_RELEASES)
    try:
        response = requests.get(
            "https://api.github.com/repos/ggerganov/llama.cpp/releases"
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        releases = response.json()
        self.release_combo.clear()
        for release in releases:
            self.release_combo.addItem(release["tag_name"], userData=release)
        self.release_combo.currentIndexChanged.connect(self.update_assets)
        self.update_assets()
    except requests.exceptions.RequestException as e:
        show_error(self.logger, ERROR_FETCHING_RELEASES.format(str(e)))
