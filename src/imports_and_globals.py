import os
import sys
import psutil
import subprocess
import time
import signal
import json
import platform
import requests
import zipfile
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QListWidget,
    QLineEdit,
    QLabel,
    QFileDialog,
    QProgressBar,
    QComboBox,
    QTextEdit,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QScrollArea,
    QSlider,
    QSpinBox,
    QListWidgetItem,
    QMessageBox,
    QDialog,
    QPlainTextEdit,
    QMenu,
)
from PySide6.QtCore import QTimer, Signal, QThread, Qt, QSize
from PySide6.QtGui import QCloseEvent, QAction

from src.Localizations import *


def show_about(self):
    about_text = (
        "AutoGGUF\n\n"
        f"Version: {AUTOGGUF_VERSION}\n\n"
        "A tool for managing and converting GGUF models."
    )
    QMessageBox.about(self, "About AutoGGUF", about_text)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def open_file_safe(file_path, mode="r"):
    encodings = ["utf-8", "latin-1", "ascii", "utf-16"]
    for encoding in encodings:
        try:
            return open(file_path, mode, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"Unable to open file {file_path} with any of the encodings: {encodings}"
    )


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller path
        base_path = sys._MEIPASS
    elif "__compiled__" in globals():
        # Nuitka path
        base_path = os.path.dirname(sys.executable)
    else:
        # Regular Python path
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
