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
from Logger import Logger
from localizations import *

class AutoGGUF(QMainWindow):
    def __init__(self):        
        super().__init__()
        self.logger = Logger("AutoGGUF", "logs")

        self.logger.info(INITIALIZING_AUTOGGUF)        
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(100, 100, 1300, 1100)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # System info
        self.ram_bar = QProgressBar()
        self.cpu_label = QLabel(CPU_USAGE)
        left_layout.addWidget(QLabel(RAM_USAGE))
        left_layout.addWidget(self.ram_bar)
        left_layout.addWidget(self.cpu_label)
        
        # Modify the backend selection
        backend_layout = QHBoxLayout()
        self.backend_combo = QComboBox()
        self.refresh_backends_button = QPushButton(REFRESH_BACKENDS)
        self.refresh_backends_button.clicked.connect(self.refresh_backends)
        backend_layout.addWidget(QLabel(BACKEND))
        backend_layout.addWidget(self.backend_combo)
        backend_layout.addWidget(self.refresh_backends_button)
        left_layout.addLayout(backend_layout)

        # Modify the Download llama.cpp section
        download_group = QGroupBox(DOWNLOAD_LLAMACPP)
        download_layout = QFormLayout()

        self.release_combo = QComboBox()
        self.refresh_releases_button = QPushButton(REFRESH_RELEASES)
        self.refresh_releases_button.clicked.connect(self.refresh_releases)
        release_layout = QHBoxLayout()
        release_layout.addWidget(self.release_combo)
        release_layout.addWidget(self.refresh_releases_button)
        download_layout.addRow(SELECT_RELEASE, release_layout)

        self.asset_combo = QComboBox()
        self.asset_combo.currentIndexChanged.connect(self.update_cuda_option)
        download_layout.addRow(SELECT_ASSET, self.asset_combo)

        self.cuda_extract_checkbox = QCheckBox(EXTRACT_CUDA_FILES)
        self.cuda_extract_checkbox.setVisible(False)
        download_layout.addRow(self.cuda_extract_checkbox)

        self.cuda_backend_label = QLabel(SELECT_CUDA_BACKEND)
        self.cuda_backend_label.setVisible(False)
        self.backend_combo_cuda = QComboBox()
        self.backend_combo_cuda.setVisible(False)
        download_layout.addRow(self.cuda_backend_label, self.backend_combo_cuda)

        self.download_progress = QProgressBar()
        self.download_button = QPushButton(DOWNLOAD)
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
        models_button = QPushButton(BROWSE)
        models_button.clicked.connect(self.browse_models)
        models_layout.addWidget(QLabel(MODELS_PATH))
        models_layout.addWidget(self.models_input)
        models_layout.addWidget(models_button)
        left_layout.addLayout(models_layout)
        
        # Output path
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit(os.path.abspath("quantized_models"))
        output_button = QPushButton(BROWSE)
        output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel(OUTPUT_PATH))
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_button)
        left_layout.addLayout(output_layout)
        
        # Logs path
        logs_layout = QHBoxLayout()
        self.logs_input = QLineEdit(os.path.abspath("logs"))
        logs_button = QPushButton(BROWSE)
        logs_button.clicked.connect(self.browse_logs)
        logs_layout.addWidget(QLabel(LOGS_PATH))
        logs_layout.addWidget(self.logs_input)
        logs_layout.addWidget(logs_button)
        left_layout.addLayout(logs_layout)
        
        # Model list
        self.model_list = QListWidget()
        self.load_models()
        left_layout.addWidget(QLabel(AVAILABLE_MODELS))
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
        quant_options_layout.addRow(self.create_label(QUANTIZATION_TYPE, SELECT_QUANTIZATION_TYPE), self.quant_type)

        self.allow_requantize = QCheckBox(ALLOW_REQUANTIZE)
        self.leave_output_tensor = QCheckBox(LEAVE_OUTPUT_TENSOR)
        self.pure = QCheckBox(PURE)
        quant_options_layout.addRow(self.create_label("", ALLOWS_REQUANTIZING), self.allow_requantize)
        quant_options_layout.addRow(self.create_label("", LEAVE_OUTPUT_WEIGHT), self.leave_output_tensor)
        quant_options_layout.addRow(self.create_label("", DISABLE_K_QUANT_MIXTURES), self.pure)

        self.imatrix = QLineEdit()
        self.imatrix_button = QPushButton(BROWSE)
        self.imatrix_button.clicked.connect(self.browse_imatrix)
        imatrix_layout = QHBoxLayout()
        imatrix_layout.addWidget(self.imatrix)
        imatrix_layout.addWidget(self.imatrix_button)
        quant_options_layout.addRow(self.create_label(IMATRIX, USE_DATA_AS_IMPORTANCE_MATRIX), imatrix_layout)

        self.include_weights = QLineEdit()
        self.exclude_weights = QLineEdit()
        quant_options_layout.addRow(self.create_label(INCLUDE_WEIGHTS, USE_IMPORTANCE_MATRIX_FOR_TENSORS), self.include_weights)
        quant_options_layout.addRow(self.create_label(EXCLUDE_WEIGHTS, DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS), self.exclude_weights)

        self.use_output_tensor_type = QCheckBox(USE_OUTPUT_TENSOR_TYPE)
        self.output_tensor_type = QComboBox()
        self.output_tensor_type.addItems(["F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
        self.output_tensor_type.setEnabled(False)
        self.use_output_tensor_type.toggled.connect(lambda checked: self.output_tensor_type.setEnabled(checked))
        output_tensor_layout = QHBoxLayout()
        output_tensor_layout.addWidget(self.use_output_tensor_type)
        output_tensor_layout.addWidget(self.output_tensor_type)
        quant_options_layout.addRow(self.create_label(OUTPUT_TENSOR_TYPE, USE_THIS_TYPE_FOR_OUTPUT_WEIGHT), output_tensor_layout)

        self.use_token_embedding_type = QCheckBox(USE_TOKEN_EMBEDDING_TYPE)
        self.token_embedding_type = QComboBox()
        self.token_embedding_type.addItems(["F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
        self.token_embedding_type.setEnabled(False)
        self.use_token_embedding_type.toggled.connect(lambda checked: self.token_embedding_type.setEnabled(checked))
        token_embedding_layout = QHBoxLayout()
        token_embedding_layout.addWidget(self.use_token_embedding_type)
        token_embedding_layout.addWidget(self.token_embedding_type)
        quant_options_layout.addRow(self.create_label(TOKEN_EMBEDDING_TYPE, USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS), token_embedding_layout)

        self.keep_split = QCheckBox(KEEP_SPLIT)
        self.override_kv = QLineEdit()
        quant_options_layout.addRow(self.create_label("", WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS), self.keep_split)
        # KV Override section
        self.kv_override_widget = QWidget()
        self.kv_override_layout = QVBoxLayout(self.kv_override_widget)
        self.kv_override_entries = []

        add_override_button = QPushButton(ADD_NEW_OVERRIDE)
        add_override_button.clicked.connect(self.add_kv_override)
        
        kv_override_scroll = QScrollArea()
        kv_override_scroll.setWidgetResizable(True)
        kv_override_scroll.setWidget(self.kv_override_widget)
        kv_override_scroll.setMinimumHeight(200)
        
        kv_override_main_layout = QVBoxLayout()
        kv_override_main_layout.addWidget(kv_override_scroll)
        kv_override_main_layout.addWidget(add_override_button)

        quant_options_layout.addRow(self.create_label(KV_OVERRIDES, OVERRIDE_MODEL_METADATA), kv_override_main_layout)

        quant_options_widget.setLayout(quant_options_layout)
        quant_options_scroll.setWidget(quant_options_widget)
        quant_options_scroll.setWidgetResizable(True)
        left_layout.addWidget(quant_options_scroll)
        
        # Quantize button layout
        quantize_layout = QHBoxLayout()
        quantize_button = QPushButton(QUANTIZE_MODEL)
        quantize_button.clicked.connect(self.quantize_model)
        save_preset_button = QPushButton(SAVE_PRESET)
        save_preset_button.clicked.connect(self.save_preset)
        load_preset_button = QPushButton(LOAD_PRESET)
        load_preset_button.clicked.connect(self.load_preset)
        quantize_layout.addWidget(quantize_button)
        quantize_layout.addWidget(save_preset_button)
        quantize_layout.addWidget(load_preset_button)
        left_layout.addLayout(quantize_layout)
        
        # Task list
        self.task_list = QListWidget()
        self.task_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.task_list.itemDoubleClicked.connect(self.show_task_details)
        left_layout.addWidget(QLabel(TASKS))
        left_layout.addWidget(self.task_list)
        
        # IMatrix section
        imatrix_group = QGroupBox(IMATRIX_GENERATION)
        imatrix_layout = QFormLayout()
        
        self.imatrix_datafile = QLineEdit()
        self.imatrix_datafile_button = QPushButton(BROWSE)
        self.imatrix_datafile_button.clicked.connect(self.browse_imatrix_datafile)
        imatrix_datafile_layout = QHBoxLayout()
        imatrix_datafile_layout.addWidget(self.imatrix_datafile)
        imatrix_datafile_layout.addWidget(self.imatrix_datafile_button)
        imatrix_layout.addRow(self.create_label(DATA_FILE, INPUT_DATA_FILE_FOR_IMATRIX), imatrix_datafile_layout)
        
        self.imatrix_model = QLineEdit()
        self.imatrix_model_button = QPushButton(BROWSE)
        self.imatrix_model_button.clicked.connect(self.browse_imatrix_model)
        imatrix_model_layout = QHBoxLayout()
        imatrix_model_layout.addWidget(self.imatrix_model)
        imatrix_model_layout.addWidget(self.imatrix_model_button)
        imatrix_layout.addRow(self.create_label(MODEL, MODEL_TO_BE_QUANTIZED), imatrix_model_layout)
        
        self.imatrix_output = QLineEdit()
        self.imatrix_output_button = QPushButton(BROWSE)
        self.imatrix_output_button.clicked.connect(self.browse_imatrix_output)
        imatrix_output_layout = QHBoxLayout()
        imatrix_output_layout.addWidget(self.imatrix_output)
        imatrix_output_layout.addWidget(self.imatrix_output_button)
        imatrix_layout.addRow(self.create_label(OUTPUT, OUTPUT_PATH_FOR_GENERATED_IMATRIX), imatrix_output_layout)
        
        self.imatrix_frequency = QLineEdit()
        imatrix_layout.addRow(self.create_label(OUTPUT_FREQUENCY, HOW_OFTEN_TO_SAVE_IMATRIX), self.imatrix_frequency)
        
        # GPU Offload for IMatrix
        gpu_offload_layout = QHBoxLayout()
        self.gpu_offload_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_offload_slider.setRange(0, 200)
        self.gpu_offload_slider.valueChanged.connect(self.update_gpu_offload_spinbox)

        self.gpu_offload_spinbox = QSpinBox()
        self.gpu_offload_spinbox.setRange(0, 1000)
        self.gpu_offload_spinbox.valueChanged.connect(self.update_gpu_offload_slider)
        self.gpu_offload_spinbox.setMinimumWidth(75)  # Set the minimum width to 75 pixels

        self.gpu_offload_auto = QCheckBox(AUTO)
        self.gpu_offload_auto.stateChanged.connect(self.toggle_gpu_offload_auto)

        gpu_offload_layout.addWidget(self.gpu_offload_slider)
        gpu_offload_layout.addWidget(self.gpu_offload_spinbox)
        gpu_offload_layout.addWidget(self.gpu_offload_auto)
        imatrix_layout.addRow(self.create_label(GPU_OFFLOAD, SET_GPU_OFFLOAD_VALUE), gpu_offload_layout)
        
        imatrix_generate_button = QPushButton(GENERATE_IMATRIX)
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
        
        self.logger.info(AUTOGGUF_INITIALIZATION_COMPLETE)            
        
    def refresh_backends(self):
        self.logger.info(REFRESHING_BACKENDS)    
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
            self.backend_combo.addItem(NO_BACKENDS_AVAILABLE)
            self.backend_combo.setEnabled(False)
        self.logger.info(FOUND_VALID_BACKENDS.format(self.backend_combo.count()))

    def save_preset(self):
        self.logger.info(SAVING_PRESET)    
        preset = {
            "quant_type": self.quant_type.currentText(),
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
            "kv_overrides": [entry.get_override_string() for entry in self.kv_override_entries]
        }
        
        file_name, _ = QFileDialog.getSaveFileName(self, SAVE_PRESET, "", JSON_FILES)
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(preset, f, indent=4)
            QMessageBox.information(self, PRESET_SAVED, PRESET_SAVED_TO.format(file_name))
        self.logger.info(PRESET_SAVED_TO.format(file_name))

    def load_preset(self):
        self.logger.info(LOADING_PRESET)    
        file_name, _ = QFileDialog.getOpenFileName(self, LOAD_PRESET, "", JSON_FILES)
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    preset = json.load(f)
                
                self.quant_type.setCurrentText(preset.get("quant_type", ""))
                self.allow_requantize.setChecked(preset.get("allow_requantize", False))
                self.leave_output_tensor.setChecked(preset.get("leave_output_tensor", False))
                self.pure.setChecked(preset.get("pure", False))
                self.imatrix.setText(preset.get("imatrix", ""))
                self.include_weights.setText(preset.get("include_weights", ""))
                self.exclude_weights.setText(preset.get("exclude_weights", ""))
                self.use_output_tensor_type.setChecked(preset.get("use_output_tensor_type", False))
                self.output_tensor_type.setCurrentText(preset.get("output_tensor_type", ""))
                self.use_token_embedding_type.setChecked(preset.get("use_token_embedding_type", False))
                self.token_embedding_type.setCurrentText(preset.get("token_embedding_type", ""))
                self.keep_split.setChecked(preset.get("keep_split", False))
                
                # Clear existing KV overrides and add new ones
                for entry in self.kv_override_entries:
                    self.remove_kv_override(entry)
                for override in preset.get("kv_overrides", []):
                    self.add_kv_override(override)
                
                QMessageBox.information(self, PRESET_LOADED, PRESET_LOADED_FROM.format(file_name))
            except Exception as e:
                QMessageBox.critical(self, ERROR, FAILED_TO_LOAD_PRESET.format(str(e)))
        self.logger.info(PRESET_LOADED_FROM.format(file_name))

    def add_kv_override(self, override_string=None):
        self.logger.debug(ADDING_KV_OVERRIDE.format(override_string))   
        entry = KVOverrideEntry()
        entry.deleted.connect(self.remove_kv_override)
        if override_string:
            key, value = override_string.split('=')
            type_, val = value.split(':')
            entry.key_input.setText(key)
            entry.type_combo.setCurrentText(type_)
            entry.value_input.setText(val)
        self.kv_override_layout.addWidget(entry)
        self.kv_override_entries.append(entry)

    def save_task_preset(self, task_item):
        self.logger.info(SAVING_TASK_PRESET.format(task_item.task_name))    
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                preset = {
                    "command": thread.command,
                    "backend_path": thread.cwd,
                    "log_file": thread.log_file
                }
                file_name, _ = QFileDialog.getSaveFileName(self, SAVE_TASK_PRESET, "", JSON_FILES)
                if file_name:
                    with open(file_name, 'w') as f:
                        json.dump(preset, f, indent=4)
                    QMessageBox.information(self, TASK_PRESET_SAVED, TASK_PRESET_SAVED_TO.format(file_name))
                break

    def restart_task(self, task_item):
        self.logger.info(RESTARTING_TASK.format(task_item.task_name))    
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                new_thread = QuantizationThread(thread.command, thread.cwd, thread.log_file)
                self.quant_threads.append(new_thread)
                new_thread.status_signal.connect(task_item.update_status)
                new_thread.finished_signal.connect(lambda: self.task_finished(new_thread))
                new_thread.error_signal.connect(lambda err: self.handle_error(err, task_item))
                new_thread.model_info_signal.connect(self.update_model_info)
                new_thread.start()
                task_item.update_status(IN_PROGRESS)
                break

    def download_finished(self, extract_dir):
        self.logger.info(DOWNLOAD_FINISHED_EXTRACTED_TO.format(extract_dir))    
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)
        
        if self.cuda_extract_checkbox.isChecked() and self.cuda_extract_checkbox.isVisible():
            cuda_backend = self.backend_combo_cuda.currentData()
            if cuda_backend and cuda_backend != NO_SUITABLE_CUDA_BACKENDS:
                self.extract_cuda_files(extract_dir, cuda_backend)
                QMessageBox.information(self, DOWNLOAD_COMPLETE, LLAMACPP_DOWNLOADED_AND_EXTRACTED.format(extract_dir, cuda_backend))
            else:
                QMessageBox.warning(self, CUDA_EXTRACTION_FAILED, NO_SUITABLE_CUDA_BACKEND_FOUND)
        else:
            QMessageBox.information(self, DOWNLOAD_COMPLETE, LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED.format(extract_dir))
        
        self.refresh_backends()  # Refresh the backends after successful download
        self.update_cuda_option()  # Update CUDA options in case a CUDA-capable backend was downloaded
        
        # Select the newly downloaded backend
        new_backend_name = os.path.basename(extract_dir)
        index = self.backend_combo.findText(new_backend_name)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)    

    def refresh_releases(self):
        self.logger.info(REFRESHING_LLAMACPP_RELEASES)    
        try:
            response = requests.get("https://api.github.com/repos/ggerganov/llama.cpp/releases")
            releases = response.json()
            self.release_combo.clear()
            for release in releases:
                self.release_combo.addItem(release['tag_name'], userData=release)
            self.release_combo.currentIndexChanged.connect(self.update_assets)
            self.update_assets()
        except Exception as e:
            self.show_error(ERROR_FETCHING_RELEASES.format(str(e)))

    def update_assets(self):
        self.logger.debug(UPDATING_ASSET_LIST)    
        self.asset_combo.clear()
        release = self.release_combo.currentData()
        if release:
            for asset in release['assets']:
                self.asset_combo.addItem(asset['name'], userData=asset)
        self.update_cuda_option()

    def update_cuda_option(self):
        self.logger.debug(UPDATING_CUDA_OPTIONS)    
        asset = self.asset_combo.currentData()
        is_cuda = asset and "cudart" in asset['name'].lower()
        self.cuda_extract_checkbox.setVisible(is_cuda)
        self.cuda_backend_label.setVisible(is_cuda)
        self.backend_combo_cuda.setVisible(is_cuda)
        if is_cuda:
            self.update_cuda_backends()

    def download_llama_cpp(self):
        self.logger.info(STARTING_LLAMACPP_DOWNLOAD)    
        asset = self.asset_combo.currentData()
        if not asset:
            self.show_error(NO_ASSET_SELECTED)
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
        self.logger.debug(UPDATING_CUDA_BACKENDS)    
        self.backend_combo_cuda.clear()
        llama_bin = os.path.abspath("llama_bin")
        if os.path.exists(llama_bin):
            for item in os.listdir(llama_bin):
                item_path = os.path.join(llama_bin, item)
                if os.path.isdir(item_path) and "cudart-llama" not in item.lower():
                    if "cu1" in item.lower():  # Only include CUDA-capable backends
                        self.backend_combo_cuda.addItem(item, userData=item_path)
        
        if self.backend_combo_cuda.count() == 0:
            self.backend_combo_cuda.addItem(NO_SUITABLE_CUDA_BACKENDS)
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
                QMessageBox.information(self, DOWNLOAD_COMPLETE, LLAMACPP_DOWNLOADED_AND_EXTRACTED.format(extract_dir, cuda_backend))
            else:
                QMessageBox.warning(self, CUDA_EXTRACTION_FAILED, NO_CUDA_BACKEND_SELECTED)
        else:
            QMessageBox.information(self, DOWNLOAD_COMPLETE, LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED.format(extract_dir))
        
        self.refresh_backends()

    def extract_cuda_files(self, extract_dir, destination):
        self.logger.info(EXTRACTING_CUDA_FILES.format(extract_dir, destination))    
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith('.dll'):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(destination, file)
                    shutil.copy2(source_path, dest_path)
        
     
    def download_error(self, error_message):
        self.logger.error(DOWNLOAD_ERROR.format(error_message))    
        self.download_button.setEnabled(True)
        self.download_progress.setValue(0)
        self.show_error(DOWNLOAD_FAILED.format(error_message))
        
        # Clean up any partially downloaded files
        asset = self.asset_combo.currentData()
        if asset:
            partial_file = os.path.join(os.path.abspath("llama_bin"), asset['name'])
            if os.path.exists(partial_file):
                os.remove(partial_file)
     
    def show_task_context_menu(self, position):
        self.logger.debug(SHOWING_TASK_CONTEXT_MENU)    
        item = self.task_list.itemAt(position)
        if item is not None:
            context_menu = QMenu(self)
            
            properties_action = QAction(PROPERTIES, self)
            properties_action.triggered.connect(lambda: self.show_task_properties(item))
            context_menu.addAction(properties_action)

            task_item = self.task_list.itemWidget(item)
            if task_item.status != COMPLETED:
                cancel_action = QAction(CANCEL, self)
                cancel_action.triggered.connect(lambda: self.cancel_task(item))
                context_menu.addAction(cancel_action)

            if task_item.status == CANCELED:
                restart_action = QAction(RESTART, self)
                restart_action.triggered.connect(lambda: self.restart_task(task_item))
                context_menu.addAction(restart_action)

            save_preset_action = QAction(SAVE_PRESET, self)
            save_preset_action.triggered.connect(lambda: self.save_task_preset(task_item))
            context_menu.addAction(save_preset_action)

            delete_action = QAction(DELETE, self)
            delete_action.triggered.connect(lambda: self.delete_task(item))
            context_menu.addAction(delete_action)

            context_menu.exec(self.task_list.viewport().mapToGlobal(position))
            
    def show_task_properties(self, item):
        self.logger.debug(SHOWING_PROPERTIES_FOR_TASK.format(item.text()))    
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                model_info_dialog = ModelInfoDialog(thread.model_info, self)
                model_info_dialog.exec()
                break

    def cancel_task(self, item):
        self.logger.info(CANCELLING_TASK.format(item.text()))    
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                thread.terminate()
                task_item.update_status(CANCELED)
                break

    def retry_task(self, item):
        task_item = self.task_list.itemWidget(item)
        # TODO: Implement the logic to restart the task
        pass
        
    def delete_task(self, item):
        self.logger.info(DELETING_TASK.format(item.text()))    
        reply = QMessageBox.question(self, CONFIRM_DELETION_TITLE,
                                     CONFIRM_DELETION,
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
        self.logger.info(LOADING_MODELS)    
        models_dir = self.models_input.text()
        ensure_directory(models_dir)
        self.model_list.clear()
        for file in os.listdir(models_dir):
            if file.endswith(".gguf"):
                self.model_list.addItem(file)
        self.logger.info(LOADED_MODELS.format(self.model_list.count()))                
    
        
    def browse_models(self):
        self.logger.info(BROWSING_FOR_MODELS_DIRECTORY)    
        models_path = QFileDialog.getExistingDirectory(self, SELECT_MODELS_DIRECTORY)
        if models_path:
            self.models_input.setText(os.path.abspath(models_path))
            ensure_directory(models_path)
            self.load_models()
    
    def browse_output(self):
        self.logger.info(BROWSING_FOR_OUTPUT_DIRECTORY)    
        output_path = QFileDialog.getExistingDirectory(self, SELECT_OUTPUT_DIRECTORY)
        if output_path:
            self.output_input.setText(os.path.abspath(output_path))
            ensure_directory(output_path)
    
    def browse_logs(self):
        self.logger.info(BROWSING_FOR_LOGS_DIRECTORY)    
        logs_path = QFileDialog.getExistingDirectory(self, SELECT_LOGS_DIRECTORY)
        if logs_path:
            self.logs_input.setText(os.path.abspath(logs_path))
            ensure_directory(logs_path)
    
    def browse_imatrix(self):
        self.logger.info(BROWSING_FOR_IMATRIX_FILE)    
        imatrix_file, _ = QFileDialog.getOpenFileName(self, SELECT_IMATRIX_FILE, "", DAT_FILES)
        if imatrix_file:
            self.imatrix.setText(os.path.abspath(imatrix_file))
    
    def update_system_info(self):
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        self.ram_bar.setValue(int(ram.percent))
        self.ram_bar.setFormat(RAM_USAGE_FORMAT.format(ram.percent, ram.used // 1024 // 1024, ram.total // 1024 // 1024))
        self.cpu_label.setText(CPU_USAGE_FORMAT.format(cpu))

    def validate_quantization_inputs(self):
        self.logger.debug(VALIDATING_QUANTIZATION_INPUTS)    
        errors = []
        if not self.backend_combo.currentData():
            errors.append(NO_BACKEND_SELECTED)
        if not self.models_input.text():
            errors.append(MODELS_PATH_REQUIRED)
        if not self.output_input.text():
            errors.append(OUTPUT_PATH_REQUIRED)
        if not self.logs_input.text():
            errors.append(LOGS_PATH_REQUIRED)
        if not self.model_list.currentItem():
            errors.append(NO_MODEL_SELECTED)
        
        if errors:
            raise ValueError("\n".join(errors))

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
        self.logger.info(STARTING_MODEL_QUANTIZATION)    
        try:
            self.validate_quantization_inputs()
            selected_model = self.model_list.currentItem()
            if not selected_model:
                raise ValueError(NO_MODEL_SELECTED)

            model_name = selected_model.text()
            backend_path = self.backend_combo.currentData()
            if not backend_path:
                raise ValueError(NO_BACKEND_SELECTED)
            quant_type = self.quant_type.currentText()
            
            input_path = os.path.join(self.models_input.text(), model_name)
            output_name = f"{os.path.splitext(model_name)[0]}_{quant_type}.gguf"
            output_path = os.path.join(self.output_input.text(), output_name)
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(INPUT_FILE_NOT_EXIST.format(input_path))

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
            
            task_item = TaskListItem(QUANTIZING_MODEL_TO.format(model_name, quant_type), log_file)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)
            
            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(lambda: self.task_finished(thread))
            thread.error_signal.connect(lambda err: self.handle_error(err, task_item))
            thread.model_info_signal.connect(self.update_model_info)
            thread.start()
            self.logger.info(QUANTIZATION_TASK_STARTED.format(model_name))             
        except ValueError as e:
            self.show_error(str(e))
        except Exception as e:
            self.show_error(ERROR_STARTING_QUANTIZATION.format(str(e)))           
            
    def update_model_info(self, model_info):
        self.logger.debug(UPDATING_MODEL_INFO.format(model_info))    
        # TODO: Do something with this
        pass    
    
    def task_finished(self, thread):
        self.logger.info(TASK_FINISHED.format(thread.log_file))    
        if thread in self.quant_threads:
            self.quant_threads.remove(thread)
    
    def show_task_details(self, item):
        self.logger.debug(SHOWING_TASK_DETAILS_FOR.format(item.text()))    
        task_item = self.task_list.itemWidget(item)
        if task_item:
            log_dialog = QDialog(self)
            log_dialog.setWindowTitle(LOG_FOR.format(task_item.task_name))
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
        self.logger.info(BROWSING_FOR_IMATRIX_DATA_FILE)    
        datafile, _ = QFileDialog.getOpenFileName(self, SELECT_DATA_FILE, "", ALL_FILES)
        if datafile:
            self.imatrix_datafile.setText(os.path.abspath(datafile))

    def browse_imatrix_model(self):
        self.logger.info(BROWSING_FOR_IMATRIX_MODEL_FILE)    
        model_file, _ = QFileDialog.getOpenFileName(self, SELECT_MODEL_FILE, "", GGUF_FILES)
        if model_file:
            self.imatrix_model.setText(os.path.abspath(model_file))

    def browse_imatrix_output(self):
        self.logger.info(BROWSING_FOR_IMATRIX_OUTPUT_FILE)    
        output_file, _ = QFileDialog.getSaveFileName(self, SELECT_OUTPUT_FILE, "", DAT_FILES)
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
        self.logger.info(STARTING_IMATRIX_GENERATION)    
        try:
            backend_path = self.backend_combo.currentData()
            if not os.path.exists(backend_path):
                raise FileNotFoundError(BACKEND_PATH_NOT_EXIST.format(backend_path))

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
            
            task_item = TaskListItem(GENERATING_IMATRIX, log_file)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)
            
            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(lambda: self.task_finished(thread))
            thread.error_signal.connect(lambda err: self.handle_error(err, task_item))
            thread.start()
        except Exception as e:
            self.show_error(ERROR_STARTING_IMATRIX_GENERATION.format(str(e)))
        self.logger.info(IMATRIX_GENERATION_TASK_STARTED)    
    
    def show_error(self, message):
        self.logger.error(ERROR_MESSAGE.format(message))    
        QMessageBox.critical(self, ERROR, message)

    def handle_error(self, error_message, task_item):
        self.logger.error(TASK_ERROR.format(error_message))    
        self.show_error(error_message)
        task_item.set_error()

    def closeEvent(self, event: QCloseEvent):
        self.logger.info(APPLICATION_CLOSING)    
        if self.quant_threads:
            reply = QMessageBox.question(self, WARNING,
                                         TASK_RUNNING_WARNING,
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
        self.logger.info(APPLICATION_CLOSED)            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoGGUF()
    window.show()
    sys.exit(app.exec())
