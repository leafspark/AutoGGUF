from typing import Any, Union

import requests
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMenu

from DownloadThread import DownloadThread
from Localizations import *
from error_handling import show_error
from globals import ensure_directory
from KVOverrideEntry import KVOverrideEntry


def show_model_context_menu(self, position):
    item = self.model_tree.itemAt(position)
    if item:
        # Child of a sharded model or top-level item without children
        if item.parent() is not None or item.childCount() == 0:
            menu = QMenu()
            rename_action = menu.addAction(RENAME)
            delete_action = menu.addAction(DELETE)

            action = menu.exec(self.model_tree.viewport().mapToGlobal(position))
            if action == rename_action:
                self.rename_model(item)
            elif action == delete_action:
                self.delete_model(item)


def rename_model(self, item):
    old_name = item.text(0)
    new_name, ok = QInputDialog.getText(self, RENAME, f"New name for {old_name}:")
    if ok and new_name:
        old_path = os.path.join(self.models_input.text(), old_name)
        new_path = os.path.join(self.models_input.text(), new_name)
        try:
            os.rename(old_path, new_path)
            item.setText(0, new_name)
            self.logger.info(MODEL_RENAMED_SUCCESSFULLY.format(old_name, new_name))
        except Exception as e:
            show_error(self.logger, f"Error renaming model: {e}")


def add_kv_override(self, override_string=None) -> None:
    entry = KVOverrideEntry()
    entry.deleted.connect(self.remove_kv_override)
    if override_string:
        key, value = override_string.split("=")
        type_, val = value.split(":")
        entry.key_input.setText(key)
        entry.type_combo.setCurrentText(type_)
        entry.value_input.setText(val)
    self.kv_override_layout.addWidget(entry)
    self.kv_override_entries.append(entry)


def remove_kv_override(self, entry) -> None:
    self.kv_override_layout.removeWidget(entry)
    self.kv_override_entries.remove(entry)
    entry.deleteLater()


def get_models_data(self) -> list[dict[str, Union[str, Any]]]:
    models = []
    root = self.model_tree.invisibleRootItem()
    child_count = root.childCount()
    for i in range(child_count):
        item = root.child(i)
        model_name = item.text(0)
        model_type = "sharded" if "sharded" in model_name.lower() else "single"
        model_path = item.data(0, Qt.ItemDataRole.UserRole)
        models.append({"name": model_name, "type": model_type, "path": model_path})
    return models


def get_tasks_data(self) -> list[dict[str, Union[int, Any]]]:
    tasks = []
    for i in range(self.task_list.count()):
        item = self.task_list.item(i)
        task_widget = self.task_list.itemWidget(item)
        if task_widget:
            tasks.append(
                {
                    "name": task_widget.task_name,
                    "status": task_widget.status,
                    "progress": (
                        task_widget.progress_bar.value()
                        if hasattr(task_widget, "progress_bar")
                        else 0
                    ),
                    "log_file": task_widget.log_file,
                }
            )
    return tasks


def browse_models(self) -> None:
    self.logger.info(BROWSING_FOR_MODELS_DIRECTORY)
    models_path = QFileDialog.getExistingDirectory(self, SELECT_MODELS_DIRECTORY)
    if models_path:
        self.models_input.setText(os.path.abspath(models_path))
        ensure_directory(models_path)
        self.load_models()


def browse_output(self) -> None:
    self.logger.info(BROWSING_FOR_OUTPUT_DIRECTORY)
    output_path = QFileDialog.getExistingDirectory(self, SELECT_OUTPUT_DIRECTORY)
    if output_path:
        self.output_input.setText(os.path.abspath(output_path))
        ensure_directory(output_path)


def browse_logs(self) -> None:
    self.logger.info(BROWSING_FOR_LOGS_DIRECTORY)
    logs_path = QFileDialog.getExistingDirectory(self, SELECT_LOGS_DIRECTORY)
    if logs_path:
        self.logs_input.setText(os.path.abspath(logs_path))
        ensure_directory(logs_path)


def browse_imatrix(self) -> None:
    self.logger.info(BROWSING_FOR_IMATRIX_FILE)
    imatrix_file, _ = QFileDialog.getOpenFileName(
        self, SELECT_IMATRIX_FILE, "", DAT_FILES
    )
    if imatrix_file:
        self.imatrix.setText(os.path.abspath(imatrix_file))


def browse_lora_input(self) -> None:
    self.logger.info(BROWSING_FOR_LORA_INPUT_DIRECTORY)
    lora_input_path = QFileDialog.getExistingDirectory(
        self, SELECT_LORA_INPUT_DIRECTORY
    )
    if lora_input_path:
        self.lora_input.setText(os.path.abspath(lora_input_path))
        ensure_directory(lora_input_path)


def browse_lora_output(self) -> None:
    self.logger.info(BROWSING_FOR_LORA_OUTPUT_FILE)
    lora_output_file, _ = QFileDialog.getSaveFileName(
        self, SELECT_LORA_OUTPUT_FILE, "", GGUF_AND_BIN_FILES
    )
    if lora_output_file:
        self.lora_output.setText(os.path.abspath(lora_output_file))


def download_llama_cpp(self) -> None:
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


def get_repo_from_env() -> tuple[str, str]:
    repo = os.getenv("AUTOGGUF_BACKEND_REPO", "ggerganov/llama.cpp")

    if not repo or "/" not in repo:
        raise ValueError(INVALID_REPOSITORY_FORMAT)

    owner, repo_name = repo.split("/", 1)
    if not all(part.strip() for part in (owner, repo_name)):
        raise ValueError(REPO_CANNOT_BE_EMPTY)

    return owner, repo_name


def refresh_releases(self) -> None:
    self.logger.info(REFRESHING_LLAMACPP_RELEASES)
    try:
        owner, repo = get_repo_from_env()
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"

        response = requests.get(url)
        response.raise_for_status()

        releases = response.json()
        self.release_combo.clear()
        for release in releases:
            self.release_combo.addItem(release["tag_name"], userData=release)
        self.release_combo.currentIndexChanged.connect(self.update_assets)
        self.update_assets()
    except ValueError as e:
        show_error(self.logger, f"Invalid repository configuration: {str(e)}")
    except requests.exceptions.RequestException as e:
        show_error(self.logger, ERROR_FETCHING_RELEASES.format(str(e)))
