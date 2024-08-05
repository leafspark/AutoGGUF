from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QComboBox, QPushButton
from PyQt6.QtCore import pyqtSignal, QRegularExpression
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QRegularExpressionValidator
from datetime import datetime
import time
import os
import socket
import platform

class KVOverrideEntry(QWidget):
    deleted = pyqtSignal(QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Key")
        # Set validator for key input (letters and dots only)
        key_validator = QRegularExpressionValidator(QRegularExpression(r"[A-Za-z.]+"))
        self.key_input.setValidator(key_validator)
        layout.addWidget(self.key_input)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["int", "str", "float"])
        layout.addWidget(self.type_combo)

        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("Value")
        layout.addWidget(self.value_input)

        delete_button = QPushButton("X")
        delete_button.setFixedSize(30, 30)
        delete_button.clicked.connect(self.delete_clicked)
        layout.addWidget(delete_button)

        # Connect type change to validator update
        self.type_combo.currentTextChanged.connect(self.update_validator)

        # Initialize validator
        self.update_validator(self.type_combo.currentText())

    def delete_clicked(self):
        self.deleted.emit(self)

    def get_override_string(self, model_name=None, quant_type=None, output_path=None):  # Add arguments
        key = self.key_input.text()
        type_ = self.type_combo.currentText()
        value = self.value_input.text()

        dynamic_params = {
            "{system.time.milliseconds}": lambda: str(int(time.time() * 1000)),
            "{system.time.seconds}": lambda: str(int(time.time())),
            "{system.date.iso}": lambda: datetime.now().strftime("%Y-%m-%d"),
            "{system.datetime.iso}": lambda: datetime.now().isoformat(),
            "{system.username}": lambda: os.getlogin(),
            "{system.hostname}": lambda: socket.gethostname(),
            "{system.platform}": lambda: platform.system(),
            "{system.python.version}": lambda: platform.python_version(),
            "{system.time.milliseconds}": lambda: str(int(time.time() * 1000)),
            "{system.date}": lambda: datetime.now().strftime("%Y-%m-%d"),
            "{model.name}": lambda: model_name if model_name is not None else "Unknown Model",
            "{quant.type}": lambda: quant_type if quant_type is not None else "Unknown Quant",
            "{output.path}": lambda: output_path if output_path is not None else "Unknown Output Path",
        }

        for param, func in dynamic_params.items():
            value = value.replace(param, func())

        return f"{key}={type_}:{value}"
        
    def get_raw_override_string(self):
        # Return the raw override string with placeholders intact
        return f"{self.key_input.text()}={self.type_combo.currentText()}:{self.value_input.text()}"    

    def update_validator(self, type_):
        if type_ == "int":
            self.value_input.setValidator(QIntValidator())
        elif type_ == "float":
            self.value_input.setValidator(QDoubleValidator())
        else:  # str
            self.value_input.setValidator(None)
