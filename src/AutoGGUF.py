from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import os
import sys
import psutil
import shutil
import subprocess
import time
import signal
import json
import platform
import requests
import zipfile
from datetime import datetime
from imports_and_globals import ensure_directory, open_file_safe
from DownloadThread import DownloadThread
from ModelInfoDialog import ModelInfoDialog
from TaskListItem import TaskListItem
from QuantizationThread import QuantizationThread
from KVOverrideEntry import KVOverrideEntry

class AutoGGUF(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoGGUF (automated GGUF model quantizer)")
        self.setGeometry(100, 100, 1300, 1100)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # System info
        self.ram_bar = QProgressBar()
        self.cpu_label = QLabel("CPU Usage: ")
        left_layout.addWidget(QLabel("RAM Usage:"))
        left_layout.addWidget(self.ram_bar)
        left_layout.addWidget(self.cpu_label)
        
        # Modify the backend selection
        backend_layout = QHBoxLayout()
        self.backend_combo = QComboBox()
        self.refresh_backends_button = QPushButton("Refresh Backends")
        self.refresh_backends_button.clicked.connect(self.refresh_backends)
        backend_layout.addWidget(QLabel("Llama.cpp Backend:"))
        backend_layout.addWidget(self.backend_combo)
        backend_layout.addWidget(self.refresh_backends_button)
        left_layout.addLayout(backend_layout)

        # Modify the Download llama.cpp section
        download_group = QGroupBox("Download llama.cpp")
        download_layout = QFormLayout()

        self.release_combo = QComboBox()
        self.refresh_releases_button = QPushButton("Refresh Releases")
        self.refresh_releases_button.clicked.connect(self.refresh_releases)
        release_layout = QHBoxLayout()
        release_layout.addWidget(self.release_combo)
        release_layout.addWidget(self.refresh_releases_button)
        download_layout.addRow("Select Release:", release_layout)

        self.asset_combo = QComboBox()
        self.asset_combo.currentIndexChanged.connect(self.update_cuda_option)
        download_layout.addRow("Select Asset:", self.asset_combo)

        self.cuda_extract_checkbox = QCheckBox("Extract CUDA files")
        self.cuda_extract_checkbox.setVisible(False)
        download_layout.addRow(self.cuda_extract_checkbox)

        self.cuda_backend_label = QLabel("Select CUDA Backend:")
        self.cuda_backend_label.setVisible(False)
        self.backend_combo_cuda = QComboBox()
        self.backend_combo_cuda.setVisible(False)
        download_layout.addRow(self.cuda_backend_label, self.backend_combo_cuda)

        self.download_progress = QProgressBar()
        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.download_llama_cpp)
        download_layout.addRow(self.download_progress)
        download_layout.addRow(self.download_button)

        download_group.setLayout(download_layout)
        right_layout.addWidget(download_group)

        # Initialize releases and backends
        self.refresh_releases()
        self.refresh_backends()    
        
        # Models path
        models_layout = QHBoxLayout()
        self.models_input = QLineEdit(os.path.abspath("models"))
        models_button = QPushButton("Browse")
        models_button.clicked.connect(self.browse_models)
        models_layout.addWidget(QLabel("Models Path:"))
        models_layout.addWidget(self.models_input)
        models_layout.addWidget(models_button)
        left_layout.addLayout(models_layout)
        
        # Output path
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit(os.path.abspath("quantized_models"))
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel("Output Path:"))
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_button)
        left_layout.addLayout(output_layout)
        
        # Logs path
        logs_layout = QHBoxLayout()
        self.logs_input = QLineEdit(os.path.abspath("logs"))
        logs_button = QPushButton("Browse")
        logs_button.clicked.connect(self.browse_logs)
        logs_layout.addWidget(QLabel("Logs Path:"))
        logs_layout.addWidget(self.logs_input)
        logs_layout.addWidget(logs_button)
        left_layout.addLayout(logs_layout)
        
        # Model list
        self.model_list = QListWidget()
        self.load_models()
        left_layout.addWidget(QLabel("Available Models:"))
        left_layout.addWidget(self.model_list)
        
        # Quantization options
        quant_options_scroll = QScrollArea()
        quant_options_widget = QWidget()
        quant_options_layout = QFormLayout()

        self.quant_type = QComboBox()
        self.quant_type.addItems([
            "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M",
            "Q2_K", "Q2_K_S", "IQ3_XXS", "IQ3_S", "IQ3_M", "Q3_K", "IQ3_XS", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "IQ4_NL", "IQ4_XS", "Q4_K", "Q4_K_S", "Q4_K_M", "Q5_K", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
            "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "F16", "BF16", "F32", "COPY"
        ])
        quant_options_layout.addRow(self.create_label("Quantization Type:", "Select the quantization type"), self.quant_type)

        self.allow_requantize = QCheckBox("Allow Requantize")
        self.leave_output_tensor = QCheckBox("Leave Output Tensor")
        self.pure = QCheckBox("Pure")
        quant_options_layout.addRow(self.create_label("", "Allows requantizing tensors that have already been quantized"), self.allow_requantize)
        quant_options_layout.addRow(self.create_label("", "Will leave output.weight un(re)quantized"), self.leave_output_tensor)
        quant_options_layout.addRow(self.create_label("", "Disable k-quant mixtures and quantize all tensors to the same type"), self.pure)

        self.imatrix = QLineEdit()
        self.imatrix_button = QPushButton("Browse")
        self.imatrix_button.clicked.connect(self.browse_imatrix)
        imatrix_layout = QHBoxLayout()
        imatrix_layout.addWidget(self.imatrix)
        imatrix_layout.addWidget(self.imatrix_button)
        quant_options_layout.addRow(self.create_label("IMatrix:", "Use data in file as importance matrix for quant optimizations"), imatrix_layout)

        self.include_weights = QLineEdit()
        self.exclude_weights = QLineEdit()
        quant_options_layout.addRow(self.create_label("Include Weights:", "Use importance matrix for these tensors"), self.include_weights)
        quant_options_layout.addRow(self.create_label("Exclude Weights:", "Don't use importance matrix for these tensors"), self.exclude_weights)

        self.use_output_tensor_type = QCheckBox("Use Output Tensor Type")
        self.output_tensor_type = QComboBox()
        self.output_tensor_type.addItems(["F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
        self.output_tensor_type.setEnabled(False)
        self.use_output_tensor_type.toggled.connect(lambda checked: self.output_tensor_type.setEnabled(checked))
        output_tensor_layout = QHBoxLayout()
        output_tensor_layout.addWidget(self.use_output_tensor_type)
        output_tensor_layout.addWidget(self.output_tensor_type)
        quant_options_layout.addRow(self.create_label("Output Tensor Type:", "Use this type for the output.weight tensor"), output_tensor_layout)

        self.use_token_embedding_type = QCheckBox("Use Token Embedding Type")
        self.token_embedding_type = QComboBox()
        self.token_embedding_type.addItems(["F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
        self.token_embedding_type.setEnabled(False)
        self.use_token_embedding_type.toggled.connect(lambda checked: self.token_embedding_type.setEnabled(checked))
        token_embedding_layout = QHBoxLayout()
        token_embedding_layout.addWidget(self.use_token_embedding_type)
        token_embedding_layout.addWidget(self.token_embedding_type)
        quant_options_layout.addRow(self.create_label("Token Embedding Type:", "Use this type for the token embeddings tensor"), token_embedding_layout)

        self.keep_split = QCheckBox("Keep Split")
        self.override_kv = QLineEdit()
        quant_options_layout.addRow(self.create_label("", "Will generate quantized model in the same shards as input"), self.keep_split)
        # KV Override section
        self.kv_override_widget = QWidget()
        self.kv_override_layout = QVBoxLayout(self.kv_override_widget)
        self.kv_override_entries = []

        add_override_button = QPushButton("Add new override")
        add_override_button.clicked.connect(self.add_kv_override)
        
        kv_override_scroll = QScrollArea()
        kv_override_scroll.setWidgetResizable(True)
        kv_override_scroll.setWidget(self.kv_override_widget)
        kv_override_scroll.setMinimumHeight(200)
        
        kv_override_main_layout = QVBoxLayout()
        kv_override_main_layout.addWidget(kv_override_scroll)
        kv_override_main_layout.addWidget(add_override_button)

        quant_options_layout.addRow(self.create_label("KV Overrides:", "Override model metadata"), kv_override_main_layout)

        quant_options_widget.setLayout(quant_options_layout)
        quant_options_scroll.setWidget(quant_options_widget)
        quant_options_scroll.setWidgetResizable(True)
        left_layout.addWidget(quant_options_scroll)
        
        # Quantize button
        quantize_button = QPushButton("Quantize Selected Model")
        quantize_button.clicked.connect(self.quantize_model)
        left_layout.addWidget(quantize_button)
        
        # Task list
        self.task_list = QListWidget()
        self.task_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.task_list.itemDoubleClicked.connect(self.show_task_details)
        left_layout.addWidget(QLabel("Tasks:"))
        left_layout.addWidget(self.task_list)
        
        # IMatrix section
        imatrix_group = QGroupBox("IMatrix Generation")
        imatrix_layout = QFormLayout()
        
        self.imatrix_datafile = QLineEdit()
        self.imatrix_datafile_button = QPushButton("Browse")
        self.imatrix_datafile_button.clicked.connect(self.browse_imatrix_datafile)
        imatrix_datafile_layout = QHBoxLayout()
        imatrix_datafile_layout.addWidget(self.imatrix_datafile)
        imatrix_datafile_layout.addWidget(self.imatrix_datafile_button)
        imatrix_layout.addRow(self.create_label("Data File:", "Input data file for IMatrix generation"), imatrix_datafile_layout)
        
        self.imatrix_model = QLineEdit()
        self.imatrix_model_button = QPushButton("Browse")
        self.imatrix_model_button.clicked.connect(self.browse_imatrix_model)
        imatrix_model_layout = QHBoxLayout()
        imatrix_model_layout.addWidget(self.imatrix_model)
        imatrix_model_layout.addWidget(self.imatrix_model_button)
        imatrix_layout.addRow(self.create_label("Model:", "Model to be quantized"), imatrix_model_layout)
        
        self.imatrix_output = QLineEdit()
        self.imatrix_output_button = QPushButton("Browse")
        self.imatrix_output_button.clicked.connect(self.browse_imatrix_output)
        imatrix_output_layout = QHBoxLayout()
        imatrix_output_layout.addWidget(self.imatrix_output)
        imatrix_output_layout.addWidget(self.imatrix_output_button)
        imatrix_layout.addRow(self.create_label("Output:", "Output path for the generated IMatrix"), imatrix_output_layout)
        
        self.imatrix_frequency = QLineEdit()
        imatrix_layout.addRow(self.create_label("Output Frequency:", "How often to save the IMatrix"), self.imatrix_frequency)
        
        # GPU Offload for IMatrix
        gpu_offload_layout = QHBoxLayout()
        self.gpu_offload_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_offload_slider.setRange(0, 200)
        self.gpu_offload_slider.valueChanged.connect(self.update_gpu_offload_spinbox)

        self.gpu_offload_spinbox = QSpinBox()
        self.gpu_offload_spinbox.setRange(0, 1000)
        self.gpu_offload_spinbox.valueChanged.connect(self.update_gpu_offload_slider)
        self.gpu_offload_spinbox.setMinimumWidth(75)  # Set the minimum width to 75 pixels

        self.gpu_offload_auto = QCheckBox("Auto")
        self.gpu_offload_auto.stateChanged.connect(self.toggle_gpu_offload_auto)

        gpu_offload_layout.addWidget(self.gpu_offload_slider)
        gpu_offload_layout.addWidget(self.gpu_offload_spinbox)
        gpu_offload_layout.addWidget(self.gpu_offload_auto)
        imatrix_layout.addRow(self.create_label("GPU Offload:", "Set GPU offload value (-ngl)"), gpu_offload_layout)
        
        imatrix_generate_button = QPushButton("Generate IMatrix")
        imatrix_generate_button.clicked.connect(self.generate_imatrix)
        imatrix_layout.addRow(imatrix_generate_button)
        
        imatrix_group.setLayout(imatrix_layout)
        right_layout.addWidget(imatrix_group)
        
        main_widget = QWidget()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Modify the task list to support right-click menu
        self.task_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.task_list.customContextMenuRequested.connect(self.show_task_context_menu)                
        
        # Timer for updating system info
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(200)

        # Initialize threads
        self.quant_threads = []
        
    def refresh_backends(self):
        llama_bin = os.path.abspath("llama_bin")
        if not os.path.exists(llama_bin):
            os.makedirs(llama_bin)
    
        self.backend_combo.clear()
        valid_backends = []
        for item in os.listdir(llama_bin):
            item_path = os.path.join(llama_bin, item)
            if os.path.isdir(item_path) and "cudart-llama" not in item.lower():
                valid_backends.append((item, item_path))
    
        if valid_backends:
            for name, path in valid_backends:
                self.backend_combo.addItem(name, userData=path)
            self.backend_combo.setEnabled(True)  # Enable the combo box if there are valid backends
        else:
            self.backend_combo.addItem("No backends available")
            self.backend_combo.setEnabled(False)

    def download_finished(self, extract_dir):
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)
        
        if self.cuda_extract_checkbox.isChecked() and self.cuda_extract_checkbox.isVisible():
            cuda_backend = self.backend_combo_cuda.currentData()
            if cuda_backend and cuda_backend != "No suitable CUDA backends found":
                self.extract_cuda_files(extract_dir, cuda_backend)
                QMessageBox.information(self, "Download Complete", f"llama.cpp binary downloaded and extracted to {extract_dir}\nCUDA files extracted to {cuda_backend}")
            else:
                QMessageBox.warning(self, "CUDA Extraction Failed", "No suitable CUDA backend found for extraction")
        else:
            QMessageBox.information(self, "Download Complete", f"llama.cpp binary downloaded and extracted to {extract_dir}")
        
        self.refresh_backends()  # Refresh the backends after successful download
        self.update_cuda_option()  # Update CUDA options in case a CUDA-capable backend was downloaded
        
        # Select the newly downloaded backend
        new_backend_name = os.path.basename(extract_dir)
        index = self.backend_combo.findText(new_backend_name)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)    

    def refresh_releases(self):
        try:
            response = requests.get("https://api.github.com/repos/ggerganov/llama.cpp/releases")
            releases = response.json()
            self.release_combo.clear()
            for release in releases:
                self.release_combo.addItem(release['tag_name'], userData=release)
            self.release_combo.currentIndexChanged.connect(self.update_assets)
            self.update_assets()
        except Exception as e:
            self.show_error(f"Error fetching releases: {str(e)}")

    def update_assets(self):
        self.asset_combo.clear()
        release = self.release_combo.currentData()
        if release:
            for asset in release['assets']:
                self.asset_combo.addItem(asset['name'], userData=asset)
        self.update_cuda_option()

    def update_cuda_option(self):
        asset = self.asset_combo.currentData()
        is_cuda = asset and "cudart" in asset['name'].lower()
        self.cuda_extract_checkbox.setVisible(is_cuda)
        self.cuda_backend_label.setVisible(is_cuda)
        self.backend_combo_cuda.setVisible(is_cuda)
        if is_cuda:
            self.update_cuda_backends()

    def download_llama_cpp(self):
        asset = self.asset_combo.currentData()
        if not asset:
            self.show_error("No asset selected")
            return

        llama_bin = os.path.abspath("llama_bin")
        if not os.path.exists(llama_bin):
            os.makedirs(llama_bin)

        save_path = os.path.join(llama_bin, asset['name'])

        self.download_thread = DownloadThread(asset['browser_download_url'], save_path)
        self.download_thread.progress_signal.connect(self.update_download_progress)
        self.download_thread.finished_signal.connect(self.download_finished)
        self.download_thread.error_signal.connect(self.download_error)
        self.download_thread.start()

        self.download_button.setEnabled(False)
        self.download_progress.setValue(0)

    def update_cuda_backends(self):
        self.backend_combo_cuda.clear()
        llama_bin = os.path.abspath("llama_bin")
        if os.path.exists(llama_bin):
            for item in os.listdir(llama_bin):
                item_path = os.path.join(llama_bin, item)
                if os.path.isdir(item_path) and "cudart-llama" not in item.lower():
                    if "cu1" in item.lower():  # Only include CUDA-capable backends
                        self.backend_combo_cuda.addItem(item, userData=item_path)
        
        if self.backend_combo_cuda.count() == 0:
            self.backend_combo_cuda.addItem("No suitable CUDA backends found")
            self.backend_combo_cuda.setEnabled(False)
        else:
            self.backend_combo_cuda.setEnabled(True)

    def update_download_progress(self, progress):
        self.download_progress.setValue(progress)

    def download_finished(self, extract_dir):
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)
        
        if self.cuda_extract_checkbox.isChecked() and self.cuda_extract_checkbox.isVisible():
            cuda_backend = self.backend_combo_cuda.currentData()
            if cuda_backend:
                self.extract_cuda_files(extract_dir, cuda_backend)
                QMessageBox.information(self, "Download Complete", f"llama.cpp binary downloaded and extracted to {extract_dir}\nCUDA files extracted to {cuda_backend}")
            else:
                QMessageBox.warning(self, "CUDA Extraction Failed", "No CUDA backend selected for extraction")
        else:
            QMessageBox.information(self, "Download Complete", f"llama.cpp binary downloaded and extracted to {extract_dir}")
        
        self.refresh_backends()

    def extract_cuda_files(self, extract_dir, destination):
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith('.dll'):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(destination, file)
                    shutil.copy2(source_path, dest_path)
        
     
    def download_error(self, error_message):
        self.download_button.setEnabled(True)
        self.download_progress.setValue(0)
        self.show_error(f"Download failed: {error_message}")
        
        # Clean up any partially downloaded files
        asset = self.asset_combo.currentData()
        if asset:
            partial_file = os.path.join(os.path.abspath("llama_bin"), asset['name'])
            if os.path.exists(partial_file):
                os.remove(partial_file)
     
    def show_task_context_menu(self, position):
        item = self.task_list.itemAt(position)
        if item is not None:
            context_menu = QMenu(self)
            
            properties_action = QAction("Properties", self)
            properties_action.triggered.connect(lambda: self.show_task_properties(item))
            context_menu.addAction(properties_action)

            if self.task_list.itemWidget(item).status != "Completed":
                cancel_action = QAction("Cancel", self)
                cancel_action.triggered.connect(lambda: self.cancel_task(item))
                context_menu.addAction(cancel_action)

            if self.task_list.itemWidget(item).status == "Canceled":
                retry_action = QAction("Retry", self)
                retry_action.triggered.connect(lambda: self.retry_task(item))
                context_menu.addAction(retry_action)

            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(lambda: self.delete_task(item))
            context_menu.addAction(delete_action)

            context_menu.exec(self.task_list.viewport().mapToGlobal(position))
            
    def show_task_properties(self, item):
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                model_info_dialog = ModelInfoDialog(thread.model_info, self)
                model_info_dialog.exec()
                break

    def cancel_task(self, item):
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                thread.terminate()
                task_item.update_status("Canceled")
                break

    def retry_task(self, item):
        task_item = self.task_list.itemWidget(item)
        # TODO: Implement the logic to restart the task
        pass
        
    def delete_task(self, item):
        reply = QMessageBox.question(self, 'Confirm Deletion',
                                     "Are you sure you want to delete this task?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            row = self.task_list.row(item)
            self.task_list.takeItem(row)
            # If the task is still running, terminate it
            task_item = self.task_list.itemWidget(item)
            for thread in self.quant_threads:
                if thread.log_file == task_item.log_file:
                    thread.terminate()
                    self.quant_threads.remove(thread)
                    break        

    def create_label(self, text, tooltip):
        label = QLabel(text)
        label.setToolTip(tooltip)
        return label
    
    def load_models(self):
        models_dir = self.models_input.text()
        ensure_directory(models_dir)
        self.model_list.clear()
        for file in os.listdir(models_dir):
            if file.endswith(".gguf"):
                self.model_list.addItem(file)
    
        
    def browse_models(self):
        models_path = QFileDialog.getExistingDirectory(self, "Select Models Directory")
        if models_path:
            self.models_input.setText(os.path.abspath(models_path))
            ensure_directory(models_path)
            self.load_models()
    
    def browse_output(self):
        output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_path:
            self.output_input.setText(os.path.abspath(output_path))
            ensure_directory(output_path)
    
    def browse_logs(self):
        logs_path = QFileDialog.getExistingDirectory(self, "Select Logs Directory")
        if logs_path:
            self.logs_input.setText(os.path.abspath(logs_path))
            ensure_directory(logs_path)
    
    def browse_imatrix(self):
        imatrix_file, _ = QFileDialog.getOpenFileName(self, "Select IMatrix File", "", "DAT Files (*.dat)")
        if imatrix_file:
            self.imatrix.setText(os.path.abspath(imatrix_file))
    
    def update_system_info(self):
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        self.ram_bar.setValue(int(ram.percent))
        self.ram_bar.setFormat(f"{ram.percent:.1f}% ({ram.used // 1024 // 1024} MB / {ram.total // 1024 // 1024} MB)")
        self.cpu_label.setText(f"CPU Usage: {cpu:.1f}%")

    def validate_quantization_inputs(self):
        if not self.backend_combo.currentData():
            raise ValueError("No backend selected")
        if not self.models_input.text():
            raise ValueError("Models path is required")
        if not self.output_input.text():
            raise ValueError("Output path is required")
        if not self.logs_input.text():
            raise ValueError("Logs path is required")

    def add_kv_override(self):
        entry = KVOverrideEntry()
        entry.deleted.connect(self.remove_kv_override)
        self.kv_override_layout.addWidget(entry)
        self.kv_override_entries.append(entry)

    def remove_kv_override(self, entry):
        self.kv_override_layout.removeWidget(entry)
        self.kv_override_entries.remove(entry)
        entry.deleteLater()

    def quantize_model(self):
        try:
            self.validate_quantization_inputs()
            selected_model = self.model_list.currentItem()
            if not selected_model:
                raise ValueError("No model selected")

            model_name = selected_model.text()
            backend_path = self.backend_combo.currentData()
            if not backend_path:
                raise ValueError("No backend selected")
            quant_type = self.quant_type.currentText()
            
            input_path = os.path.join(self.models_input.text(), model_name)
            output_name = f"{os.path.splitext(model_name)[0]}_{quant_type}.gguf"
            output_path = os.path.join(self.output_input.text(), output_name)
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

            command = [os.path.join(backend_path, "llama-quantize")]
            
            if self.allow_requantize.isChecked():
                command.append("--allow-requantize")
            if self.leave_output_tensor.isChecked():
                command.append("--leave-output-tensor")
            if self.pure.isChecked():
                command.append("--pure")
            if self.imatrix.text():
                command.extend(["--imatrix", self.imatrix.text()])
            if self.include_weights.text():
                command.extend(["--include-weights", self.include_weights.text()])
            if self.exclude_weights.text():
                command.extend(["--exclude-weights", self.exclude_weights.text()])
            if self.use_output_tensor_type.isChecked():
                command.extend(["--output-tensor-type", self.output_tensor_type.currentText()])
            if self.use_token_embedding_type.isChecked():
                command.extend(["--token-embedding-type", self.token_embedding_type.currentText()])
            if self.keep_split.isChecked():
                command.append("--keep-split")
            if self.override_kv.text():
                for entry in self.kv_override_entries:
                    override_string = entry.get_override_string()
                    if override_string:
                        command.extend(["--override-kv", override_string])
            
            command.extend([input_path, output_path, quant_type])
            
            logs_path = self.logs_input.text()
            ensure_directory(logs_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_path, f"{model_name}_{timestamp}_{quant_type}.log")
            
            thread = QuantizationThread(command, backend_path, log_file)
            self.quant_threads.append(thread)
            
            task_item = TaskListItem(f"Quantizing {model_name} to {quant_type}", log_file)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)
            
            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(lambda: self.task_finished(thread))
            thread.error_signal.connect(lambda err: self.handle_error(err, task_item))
            thread.model_info_signal.connect(self.update_model_info)
            thread.start()
        except Exception as e:
            self.show_error(f"Error starting quantization: {str(e)}")
            
    def update_model_info(self, model_info):
        # TODO: Do something with this
        pass    
    
    def task_finished(self, thread):
        if thread in self.quant_threads:
            self.quant_threads.remove(thread)
    
    def show_task_details(self, item):
        task_item = self.task_list.itemWidget(item)
        if task_item:
            log_dialog = QDialog(self)
            log_dialog.setWindowTitle(f"Log for {task_item.task_name}")
            log_dialog.setGeometry(200, 200, 800, 600)
            
            log_text = QPlainTextEdit()
            log_text.setReadOnly(True)
            
            layout = QVBoxLayout()
            layout.addWidget(log_text)
            log_dialog.setLayout(layout)
            
            # Load existing content
            if os.path.exists(task_item.log_file):
                with open_file_safe(task_item.log_file, 'r') as f:
                    log_text.setPlainText(f.read())
            
            # Connect to the thread if it's still running
            for thread in self.quant_threads:
                if thread.log_file == task_item.log_file:
                    thread.output_signal.connect(log_text.appendPlainText)
                    break
            
            log_dialog.exec()
                
    def browse_imatrix_datafile(self):
        datafile, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*)")
        if datafile:
            self.imatrix_datafile.setText(os.path.abspath(datafile))

    def browse_imatrix_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "GGUF Files (*.gguf)")
        if model_file:
            self.imatrix_model.setText(os.path.abspath(model_file))

    def browse_imatrix_output(self):
        output_file, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "DAT Files (*.dat)")
        if output_file:
            self.imatrix_output.setText(os.path.abspath(output_file))

    def update_gpu_offload_spinbox(self, value):
        self.gpu_offload_spinbox.setValue(value)

    def update_gpu_offload_slider(self, value):
        self.gpu_offload_slider.setValue(value)

    def toggle_gpu_offload_auto(self, state):
        is_auto = state == Qt.CheckState.Checked
        self.gpu_offload_slider.setEnabled(not is_auto)
        self.gpu_offload_spinbox.setEnabled(not is_auto)

    def generate_imatrix(self):
        try:
            backend_path = self.backend_combo.currentData()
            if not os.path.exists(backend_path):
                raise FileNotFoundError(f"Backend path does not exist: {backend_path}")

            command = [
                os.path.join(backend_path, "llama-imatrix"),
                "-f", self.imatrix_datafile.text(),
                "-m", self.imatrix_model.text(),
                "-o", self.imatrix_output.text(),
                "--output-frequency", self.imatrix_frequency.text()
            ]

            if self.gpu_offload_auto.isChecked():
                command.extend(["-ngl", "99"])
            elif self.gpu_offload_spinbox.value() > 0:
                command.extend(["-ngl", str(self.gpu_offload_spinbox.value())])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.logs_input.text(), f"imatrix_{timestamp}.log")
            
            thread = QuantizationThread(command, backend_path, log_file)
            self.quant_threads.append(thread)
            
            task_item = TaskListItem("Generating IMatrix", log_file)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)
            
            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(lambda: self.task_finished(thread))
            thread.error_signal.connect(lambda err: self.handle_error(err, task_item))
            thread.start()
        except Exception as e:
            self.show_error(f"Error starting IMatrix generation: {str(e)}")
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def handle_error(self, error_message, task_item):
        self.show_error(error_message)
        task_item.set_error()

    def closeEvent(self, event: QCloseEvent):
        if self.quant_threads:
            reply = QMessageBox.question(self, 'Warning',
                                         "Some tasks are still running. Are you sure you want to quit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                for thread in self.quant_threads:
                    thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoGGUF()
    window.show()
    sys.exit(app.exec())