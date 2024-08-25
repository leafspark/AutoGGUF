import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox
from Localizations import (
    SAVING_PRESET,
    SAVE_PRESET,
    JSON_FILES,
    PRESET_SAVED,
    PRESET_SAVED_TO,
    LOADING_PRESET,
    LOAD_PRESET,
    PRESET_LOADED,
    PRESET_LOADED_FROM,
)


def save_preset(self) -> None:
    self.logger.info(SAVING_PRESET)
    preset = {
        "quant_types": [item.text() for item in self.quant_type.selectedItems()],
        "allow_requantize": self.allow_requantize.isChecked(),
        "leave_output_tensor": self.leave_output_tensor.isChecked(),
        "pure": self.pure.isChecked(),
        "imatrix": self.imatrix.text(),
        "include_weights": self.include_weights.text(),
        "exclude_weights": self.exclude_weights.text(),
        "use_output_tensor_type": self.use_output_tensor_type.isChecked(),
        "output_tensor_type": self.output_tensor_type.currentText(),
        "use_token_embedding_type": self.use_token_embedding_type.isChecked(),
        "token_embedding_type": self.token_embedding_type.currentText(),
        "keep_split": self.keep_split.isChecked(),
        "kv_overrides": [
            entry.get_raw_override_string() for entry in self.kv_override_entries
        ],
        "extra_arguments": self.extra_arguments.text(),
    }

    file_name, _ = QFileDialog.getSaveFileName(self, SAVE_PRESET, "", JSON_FILES)
    if file_name:
        with open(file_name, "w") as f:
            json.dump(preset, f, indent=4)
        QMessageBox.information(self, PRESET_SAVED, PRESET_SAVED_TO.format(file_name))
    self.logger.info(PRESET_SAVED_TO.format(file_name))


def load_preset(self) -> None:
    self.logger.info(LOADING_PRESET)
    file_name, _ = QFileDialog.getOpenFileName(self, LOAD_PRESET, "", JSON_FILES)
    if file_name:
        with open(file_name, "r") as f:
            preset = json.load(f)

        self.quant_type.clearSelection()
        for quant_type in preset.get("quant_types", []):
            items = self.quant_type.findItems(quant_type, Qt.MatchExactly)
            if items:
                items[0].setSelected(True)
        self.allow_requantize.setChecked(preset.get("allow_requantize", False))
        self.leave_output_tensor.setChecked(preset.get("leave_output_tensor", False))
        self.pure.setChecked(preset.get("pure", False))
        self.imatrix.setText(preset.get("imatrix", ""))
        self.include_weights.setText(preset.get("include_weights", ""))
        self.exclude_weights.setText(preset.get("exclude_weights", ""))
        self.use_output_tensor_type.setChecked(
            preset.get("use_output_tensor_type", False)
        )
        self.output_tensor_type.setCurrentText(preset.get("output_tensor_type", ""))
        self.use_token_embedding_type.setChecked(
            preset.get("use_token_embedding_type", False)
        )
        self.token_embedding_type.setCurrentText(preset.get("token_embedding_type", ""))
        self.keep_split.setChecked(preset.get("keep_split", False))
        self.extra_arguments.setText(preset.get("extra_arguments", ""))

        # Clear existing KV overrides and add new ones
        for entry in self.kv_override_entries:
            self.remove_kv_override(entry)
        for override in preset.get("kv_overrides", []):
            self.add_kv_override(override)

        QMessageBox.information(
            self, PRESET_LOADED, PRESET_LOADED_FROM.format(file_name)
        )
    self.logger.info(PRESET_LOADED_FROM.format(file_name))
