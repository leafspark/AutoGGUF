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
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
                             QListWidget, QLineEdit, QLabel, QFileDialog, QProgressBar, QComboBox, QTextEdit,
                             QCheckBox, QGroupBox, QFormLayout, QScrollArea, QSlider, QSpinBox, QListWidgetItem,
                             QMessageBox, QDialog, QPlainTextEdit, QMenu)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QSize
from PyQt6.QtGui import QCloseEvent, QAction

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)