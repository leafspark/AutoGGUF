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
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
