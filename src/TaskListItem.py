from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
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

class TaskListItem(QWidget):
    def __init__(self, task_name, log_file, show_progress_bar=True, parent=None):
        super().__init__(parent)
        self.task_name = task_name
        self.log_file = log_file
        self.status = "Pending"
        layout = QHBoxLayout(self)
        self.task_label = QLabel(task_name)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel(self.status)
        layout.addWidget(self.task_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        # Hide progress bar if show_progress_bar is False
        self.progress_bar.setVisible(show_progress_bar)

        # Use indeterminate progress bar if not showing percentage
        if not show_progress_bar:
            self.progress_bar.setRange(0, 0)

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0

    def update_status(self, status):
        self.status = status
        self.status_label.setText(status)
        if status == "In Progress":
            # Only start timer if showing percentage progress
            if self.progress_bar.isVisible():
                self.progress_bar.setRange(0, 100)
                self.progress_timer.start(100)
        elif status == "Completed":
            self.progress_timer.stop()
            self.progress_bar.setValue(100)
        elif status == "Canceled":
            self.progress_timer.stop()
            self.progress_bar.setValue(0)

    def set_error(self):
        self.status = "Error"
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: red;")
        self.progress_bar.setRange(0, 100)
        self.progress_timer.stop()

    def update_progress(self, value=None):
        if value is not None:
            # Update progress bar with specific value
            self.progress_value = value
            self.progress_bar.setValue(self.progress_value)
        else:
            # Increment progress bar for indeterminate progress
            self.progress_value = (self.progress_value + 1) % 101
            self.progress_bar.setValue(self.progress_value)