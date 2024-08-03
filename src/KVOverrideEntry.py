from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QComboBox, QPushButton
from PyQt6.QtCore import pyqtSignal

class KVOverrideEntry(QWidget):
    deleted = pyqtSignal(QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Key")
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

    def delete_clicked(self):
        self.deleted.emit(self)

    def get_override_string(self):
        return f"{self.key_input.text()}={self.type_combo.currentText()}:{self.value_input.text()}"
