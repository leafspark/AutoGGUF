import json
import re
import shutil
from datetime import datetime

import psutil
import requests
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from flask import Flask, jsonify

from DownloadThread import DownloadThread
from GPUMonitor import GPUMonitor
from KVOverrideEntry import KVOverrideEntry
from Logger import Logger
from ModelInfoDialog import ModelInfoDialog
from QuantizationThread import QuantizationThread
from TaskListItem import TaskListItem
from error_handling import show_error, handle_error
from imports_and_globals import ensure_directory, open_file_safe, resource_path
from localizations import *


class AutoGGUF(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = Logger("AutoGGUF", "logs")

        self.logger.info(INITIALIZING_AUTOGGUF)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QIcon(resource_path("assets/favicon.ico")))
        self.setGeometry(100, 100, 1600, 1200)

        ensure_directory(os.path.abspath("quantized_models"))
        ensure_directory(os.path.abspath("models"))

        # Create a central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # Create a scroll area and set it as the central widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central_widget)
        self.setCentralWidget(scroll)

        # Create left and right widgets
        left_widget = QWidget()
        right_widget = QWidget()

        # Set minimum widths to maintain proportions
        left_widget.setMinimumWidth(800)
        right_widget.setMinimumWidth(400)

        left_layout = QVBoxLayout(left_widget)
        right_layout = QVBoxLayout(right_widget)

        # Add left and right widgets to the main layout
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 1)

        # System info
        self.ram_bar = QProgressBar()
        self.cpu_label = QLabel(CPU_USAGE)
        self.gpu_monitor = GPUMonitor()
        left_layout.addWidget(QLabel(RAM_USAGE))
        left_layout.addWidget(self.ram_bar)
        left_layout.addWidget(self.cpu_label)
        left_layout.addWidget(QLabel(GPU_USAGE))
        left_layout.addWidget(self.gpu_monitor)

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
        if os.environ.get("AUTOGGUF_CHECK_BACKEND", "").lower() == "enabled":
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
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderHidden(True)
        left_layout.addWidget(QLabel(AVAILABLE_MODELS))
        left_layout.addWidget(self.model_tree)

        # Refresh models button
        refresh_models_button = QPushButton(REFRESH_MODELS)
        refresh_models_button.clicked.connect(self.load_models)
        left_layout.addWidget(refresh_models_button)

        # Quantization options
        quant_options_scroll = QScrollArea()
        quant_options_widget = QWidget()
        quant_options_layout = QFormLayout()

        self.quant_type = QListWidget()
        self.quant_type.setMinimumHeight(100)
        self.quant_type.setMinimumWidth(150)
        self.quant_type.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        quant_types = [
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ2_M",
            "IQ1_S",
            "IQ1_M",
            "Q2_K",
            "Q2_K_S",
            "IQ3_XXS",
            "IQ3_S",
            "IQ3_M",
            "Q3_K",
            "IQ3_XS",
            "Q3_K_S",
            "Q3_K_M",
            "Q3_K_L",
            "IQ4_NL",
            "IQ4_XS",
            "Q4_K",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_K",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "Q4_0",
            "Q4_1",
            "Q5_0",
            "Q5_1",
            "Q4_0_4_4",
            "Q4_0_4_8",
            "Q4_0_8_8",
            "BF16",
            "F16",
            "F32",
            "COPY",
        ]
        self.quant_type.addItems(quant_types)
        quant_options_layout.addRow(
            self.create_label(QUANTIZATION_TYPE, SELECT_QUANTIZATION_TYPE),
            self.quant_type,
        )

        self.allow_requantize = QCheckBox(ALLOW_REQUANTIZE)
        self.leave_output_tensor = QCheckBox(LEAVE_OUTPUT_TENSOR)
        self.pure = QCheckBox(PURE)
        quant_options_layout.addRow(
            self.create_label("", ALLOWS_REQUANTIZING), self.allow_requantize
        )
        quant_options_layout.addRow(
            self.create_label("", LEAVE_OUTPUT_WEIGHT), self.leave_output_tensor
        )
        quant_options_layout.addRow(
            self.create_label("", DISABLE_K_QUANT_MIXTURES), self.pure
        )

        self.imatrix = QLineEdit()
        self.imatrix_button = QPushButton(BROWSE)
        self.imatrix_button.clicked.connect(self.browse_imatrix)
        imatrix_layout = QHBoxLayout()
        imatrix_layout.addWidget(self.imatrix)
        imatrix_layout.addWidget(self.imatrix_button)
        quant_options_layout.addRow(
            self.create_label(IMATRIX, USE_DATA_AS_IMPORTANCE_MATRIX), imatrix_layout
        )

        self.include_weights = QLineEdit()
        self.exclude_weights = QLineEdit()
        quant_options_layout.addRow(
            self.create_label(INCLUDE_WEIGHTS, USE_IMPORTANCE_MATRIX_FOR_TENSORS),
            self.include_weights,
        )
        quant_options_layout.addRow(
            self.create_label(EXCLUDE_WEIGHTS, DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS),
            self.exclude_weights,
        )

        tensor_types = [
            "Q2_K",
            "Q2_K_S",
            "Q3_K_S",
            "Q3_K_M",
            "Q3_K_L",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "Q4_0",
            "Q4_1",
            "Q5_0",
            "Q5_1",
            "BF16",
            "F16",
            "F32",
        ]

        self.use_output_tensor_type = QCheckBox(USE_OUTPUT_TENSOR_TYPE)
        self.output_tensor_type = QComboBox()
        self.output_tensor_type.addItems(tensor_types)
        self.output_tensor_type.setEnabled(False)
        self.use_output_tensor_type.toggled.connect(
            lambda checked: self.output_tensor_type.setEnabled(checked)
        )
        output_tensor_layout = QHBoxLayout()
        output_tensor_layout.addWidget(self.use_output_tensor_type)
        output_tensor_layout.addWidget(self.output_tensor_type)
        quant_options_layout.addRow(
            self.create_label(OUTPUT_TENSOR_TYPE, USE_THIS_TYPE_FOR_OUTPUT_WEIGHT),
            output_tensor_layout,
        )

        self.use_token_embedding_type = QCheckBox(USE_TOKEN_EMBEDDING_TYPE)
        self.token_embedding_type = QComboBox()
        self.token_embedding_type.addItems(tensor_types)
        self.token_embedding_type.setEnabled(False)
        self.use_token_embedding_type.toggled.connect(
            lambda checked: self.token_embedding_type.setEnabled(checked)
        )
        token_embedding_layout = QHBoxLayout()
        token_embedding_layout.addWidget(self.use_token_embedding_type)
        token_embedding_layout.addWidget(self.token_embedding_type)
        quant_options_layout.addRow(
            self.create_label(TOKEN_EMBEDDING_TYPE, USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS),
            token_embedding_layout,
        )

        self.keep_split = QCheckBox(KEEP_SPLIT)
        self.override_kv = QLineEdit()
        quant_options_layout.addRow(
            self.create_label("", WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS),
            self.keep_split,
        )

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

        quant_options_layout.addRow(
            self.create_label(KV_OVERRIDES, OVERRIDE_MODEL_METADATA),
            kv_override_main_layout,
        )

        quant_options_widget.setLayout(quant_options_layout)
        quant_options_scroll.setWidget(quant_options_widget)
        quant_options_scroll.setWidgetResizable(True)
        left_layout.addWidget(quant_options_scroll)

        # Add this after the KV override section
        self.extra_arguments = QLineEdit()
        quant_options_layout.addRow(
            self.create_label(EXTRA_ARGUMENTS, "Additional command-line arguments"),
            self.extra_arguments,
        )

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
        imatrix_layout.addRow(
            self.create_label(DATA_FILE, INPUT_DATA_FILE_FOR_IMATRIX),
            imatrix_datafile_layout,
        )

        self.imatrix_model = QLineEdit()
        self.imatrix_model_button = QPushButton(BROWSE)
        self.imatrix_model_button.clicked.connect(self.browse_imatrix_model)
        imatrix_model_layout = QHBoxLayout()
        imatrix_model_layout.addWidget(self.imatrix_model)
        imatrix_model_layout.addWidget(self.imatrix_model_button)
        imatrix_layout.addRow(
            self.create_label(MODEL, MODEL_TO_BE_QUANTIZED), imatrix_model_layout
        )

        self.imatrix_output = QLineEdit()
        self.imatrix_output_button = QPushButton(BROWSE)
        self.imatrix_output_button.clicked.connect(self.browse_imatrix_output)
        imatrix_output_layout = QHBoxLayout()
        imatrix_output_layout.addWidget(self.imatrix_output)
        imatrix_output_layout.addWidget(self.imatrix_output_button)
        imatrix_layout.addRow(
            self.create_label(OUTPUT, OUTPUT_PATH_FOR_GENERATED_IMATRIX),
            imatrix_output_layout,
        )

        self.imatrix_frequency = QSpinBox()
        self.imatrix_frequency.setRange(1, 100)  # Set the range from 1 to 100
        self.imatrix_frequency.setValue(1)  # Set a default value
        imatrix_layout.addRow(
            self.create_label(OUTPUT_FREQUENCY, HOW_OFTEN_TO_SAVE_IMATRIX),
            self.imatrix_frequency,
        )

        # Context size input (now a spinbox)
        self.imatrix_ctx_size = QSpinBox()
        self.imatrix_ctx_size.setRange(1, 1048576)  # Up to one million tokens
        self.imatrix_ctx_size.setValue(512)  # Set a default value
        imatrix_layout.addRow(
            self.create_label(CONTEXT_SIZE, CONTEXT_SIZE_FOR_IMATRIX),
            self.imatrix_ctx_size,
        )

        # Threads input with slider and spinbox
        threads_layout = QHBoxLayout()
        self.threads_slider = QSlider(Qt.Orientation.Horizontal)
        self.threads_slider.setRange(1, 64)
        self.threads_slider.valueChanged.connect(self.update_threads_spinbox)

        self.threads_spinbox = QSpinBox()
        self.threads_spinbox.setRange(1, 128)
        self.threads_spinbox.valueChanged.connect(self.update_threads_slider)
        self.threads_spinbox.setMinimumWidth(75)

        threads_layout.addWidget(self.threads_slider)
        threads_layout.addWidget(self.threads_spinbox)
        imatrix_layout.addRow(
            self.create_label(THREADS, NUMBER_OF_THREADS_FOR_IMATRIX), threads_layout
        )

        # GPU Offload for IMatrix (corrected version)
        gpu_offload_layout = QHBoxLayout()
        self.gpu_offload_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_offload_slider.setRange(0, 200)
        self.gpu_offload_slider.valueChanged.connect(self.update_gpu_offload_spinbox)

        self.gpu_offload_spinbox = QSpinBox()
        self.gpu_offload_spinbox.setRange(0, 1000)
        self.gpu_offload_spinbox.valueChanged.connect(self.update_gpu_offload_slider)
        self.gpu_offload_spinbox.setMinimumWidth(75)

        self.gpu_offload_auto = QCheckBox(AUTO)
        self.gpu_offload_auto.stateChanged.connect(self.toggle_gpu_offload_auto)

        gpu_offload_layout.addWidget(self.gpu_offload_slider)
        gpu_offload_layout.addWidget(self.gpu_offload_spinbox)
        gpu_offload_layout.addWidget(self.gpu_offload_auto)
        imatrix_layout.addRow(
            self.create_label(GPU_OFFLOAD, SET_GPU_OFFLOAD_VALUE), gpu_offload_layout
        )

        imatrix_generate_button = QPushButton(GENERATE_IMATRIX)
        imatrix_generate_button.clicked.connect(self.generate_imatrix)
        imatrix_layout.addRow(imatrix_generate_button)

        imatrix_group.setLayout(imatrix_layout)
        right_layout.addWidget(imatrix_group)

        # LoRA Conversion Section
        lora_group = QGroupBox(LORA_CONVERSION)
        lora_layout = QFormLayout()

        self.lora_input = QLineEdit()
        lora_input_button = QPushButton(BROWSE)
        lora_input_button.clicked.connect(self.browse_lora_input)
        lora_input_layout = QHBoxLayout()
        lora_input_layout.addWidget(self.lora_input)
        lora_input_layout.addWidget(lora_input_button)
        lora_layout.addRow(
            self.create_label(LORA_INPUT_PATH, SELECT_LORA_INPUT_DIRECTORY),
            lora_input_layout,
        )

        self.lora_output = QLineEdit()
        lora_output_button = QPushButton(BROWSE)
        lora_output_button.clicked.connect(self.browse_lora_output)
        lora_output_layout = QHBoxLayout()
        lora_output_layout.addWidget(self.lora_output)
        lora_output_layout.addWidget(lora_output_button)
        lora_layout.addRow(
            self.create_label(LORA_OUTPUT_PATH, SELECT_LORA_OUTPUT_FILE),
            lora_output_layout,
        )

        # Output Type Dropdown
        self.lora_output_type_combo = QComboBox()
        self.lora_output_type_combo.addItems(["GGML", "GGUF"])
        self.lora_output_type_combo.currentIndexChanged.connect(
            self.update_base_model_visibility
        )
        lora_layout.addRow(
            self.create_label(OUTPUT_TYPE, SELECT_OUTPUT_TYPE),
            self.lora_output_type_combo,
        )

        # Base Model Path (initially hidden)
        self.base_model_label = self.create_label(BASE_MODEL, SELECT_BASE_MODEL_FILE)
        self.base_model_path = QLineEdit()
        base_model_button = QPushButton(BROWSE)
        base_model_button.clicked.connect(self.browse_base_model)
        base_model_layout = QHBoxLayout()
        base_model_layout.addWidget(self.base_model_path, 1)  # Give it a stretch factor
        base_model_layout.addWidget(base_model_button)
        self.base_model_widget = QWidget()
        self.base_model_widget.setLayout(base_model_layout)

        # Create a wrapper widget to hold both label and input
        self.base_model_wrapper = QWidget()
        wrapper_layout = QHBoxLayout(self.base_model_wrapper)
        wrapper_layout.addWidget(self.base_model_label)
        wrapper_layout.addWidget(self.base_model_widget, 1)  # Give it a stretch factor
        wrapper_layout.setContentsMargins(
            0, 0, 0, 0
        )  # Remove margins for better alignment

        # Add the wrapper to the layout
        lora_layout.addRow(self.base_model_wrapper)

        # Set initial visibility
        self.update_base_model_visibility(self.lora_output_type_combo.currentIndex())

        lora_convert_button = QPushButton(CONVERT_LORA)
        lora_convert_button.clicked.connect(self.convert_lora)
        lora_layout.addRow(lora_convert_button)

        lora_group.setLayout(lora_layout)
        right_layout.addWidget(lora_group)

        # Export LoRA
        export_lora_group = QGroupBox(EXPORT_LORA)
        export_lora_layout = QFormLayout()

        self.export_lora_model = QLineEdit()
        export_lora_model_button = QPushButton(BROWSE)
        export_lora_model_button.clicked.connect(self.browse_export_lora_model)
        export_lora_model_layout = QHBoxLayout()
        export_lora_model_layout.addWidget(self.export_lora_model)
        export_lora_model_layout.addWidget(export_lora_model_button)
        export_lora_layout.addRow(
            self.create_label(MODEL, SELECT_MODEL_FILE), export_lora_model_layout
        )

        self.export_lora_output = QLineEdit()
        export_lora_output_button = QPushButton(BROWSE)
        export_lora_output_button.clicked.connect(self.browse_export_lora_output)
        export_lora_output_layout = QHBoxLayout()
        export_lora_output_layout.addWidget(self.export_lora_output)
        export_lora_output_layout.addWidget(export_lora_output_button)
        export_lora_layout.addRow(
            self.create_label(OUTPUT, SELECT_OUTPUT_FILE), export_lora_output_layout
        )

        # GGML LoRA Adapters
        self.export_lora_adapters = QListWidget()
        add_adapter_button = QPushButton(ADD_ADAPTER)
        add_adapter_button.clicked.connect(self.add_lora_adapter)
        adapters_layout = QVBoxLayout()
        adapters_layout.addWidget(self.export_lora_adapters)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(add_adapter_button)
        adapters_layout.addLayout(buttons_layout)
        export_lora_layout.addRow(
            self.create_label(GGML_LORA_ADAPTERS, SELECT_LORA_ADAPTER_FILES),
            adapters_layout,
        )

        # Threads
        self.export_lora_threads = QSpinBox()
        self.export_lora_threads.setRange(1, 64)
        self.export_lora_threads.setValue(8)  # Default value
        export_lora_layout.addRow(
            self.create_label(THREADS, NUMBER_OF_THREADS_FOR_LORA_EXPORT),
            self.export_lora_threads,
        )

        export_lora_button = QPushButton(EXPORT_LORA)
        export_lora_button.clicked.connect(self.export_lora)
        export_lora_layout.addRow(export_lora_button)

        export_lora_group.setLayout(export_lora_layout)
        right_layout.addWidget(
            export_lora_group
        )  # Add the Export LoRA group to the right layout

        # HuggingFace to GGUF Conversion
        hf_to_gguf_group = QGroupBox(HF_TO_GGUF_CONVERSION)
        hf_to_gguf_layout = QFormLayout()

        self.hf_model_input = QLineEdit()
        hf_model_input_button = QPushButton(BROWSE)
        hf_model_input_button.clicked.connect(self.browse_hf_model_input)
        hf_model_input_layout = QHBoxLayout()
        hf_model_input_layout.addWidget(self.hf_model_input)
        hf_model_input_layout.addWidget(hf_model_input_button)
        hf_to_gguf_layout.addRow(MODEL_DIRECTORY, hf_model_input_layout)

        self.hf_outfile = QLineEdit()
        hf_outfile_button = QPushButton(BROWSE)
        hf_outfile_button.clicked.connect(self.browse_hf_outfile)
        hf_outfile_layout = QHBoxLayout()
        hf_outfile_layout.addWidget(self.hf_outfile)
        hf_outfile_layout.addWidget(hf_outfile_button)
        hf_to_gguf_layout.addRow(OUTPUT_FILE, hf_outfile_layout)

        self.hf_outtype = QComboBox()
        self.hf_outtype.addItems(["f32", "f16", "bf16", "q8_0", "auto"])
        hf_to_gguf_layout.addRow(OUTPUT_TYPE, self.hf_outtype)

        self.hf_vocab_only = QCheckBox(VOCAB_ONLY)
        hf_to_gguf_layout.addRow(self.hf_vocab_only)

        self.hf_use_temp_file = QCheckBox(USE_TEMP_FILE)
        hf_to_gguf_layout.addRow(self.hf_use_temp_file)

        self.hf_no_lazy = QCheckBox(NO_LAZY_EVALUATION)
        hf_to_gguf_layout.addRow(self.hf_no_lazy)

        self.hf_model_name = QLineEdit()
        hf_to_gguf_layout.addRow(MODEL_NAME, self.hf_model_name)

        self.hf_verbose = QCheckBox(VERBOSE)
        hf_to_gguf_layout.addRow(self.hf_verbose)

        self.hf_split_max_size = QLineEdit()
        hf_to_gguf_layout.addRow(SPLIT_MAX_SIZE, self.hf_split_max_size)

        self.hf_dry_run = QCheckBox(DRY_RUN)
        hf_to_gguf_layout.addRow(self.hf_dry_run)

        hf_to_gguf_convert_button = QPushButton(CONVERT_HF_TO_GGUF)
        hf_to_gguf_convert_button.clicked.connect(self.convert_hf_to_gguf)
        hf_to_gguf_layout.addRow(hf_to_gguf_convert_button)

        hf_to_gguf_group.setLayout(hf_to_gguf_layout)
        right_layout.addWidget(hf_to_gguf_group)

        # Modify the task list to support right-click menu
        self.task_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.task_list.customContextMenuRequested.connect(self.show_task_context_menu)

        # Set inital state
        self.update_base_model_visibility(self.lora_output_type_combo.currentIndex())

        # Timer for updating system info
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(200)

        # Initialize threads
        self.quant_threads = []

        # Load models
        self.load_models()
        self.logger.info(AUTOGGUF_INITIALIZATION_COMPLETE)

    def refresh_backends(self):
        self.logger.info(REFRESHING_BACKENDS)
        llama_bin = os.path.abspath("llama_bin")
        os.makedirs(llama_bin, exist_ok=True)

        self.backend_combo.clear()
        valid_backends = [
            (item, os.path.join(llama_bin, item))
            for item in os.listdir(llama_bin)
            if os.path.isdir(os.path.join(llama_bin, item))
            and "cudart-llama" not in item.lower()
        ]

        if valid_backends:
            for name, path in valid_backends:
                self.backend_combo.addItem(name, userData=path)
            self.backend_combo.setEnabled(
                True
            )  # Enable the combo box if there are valid backends
        else:
            self.backend_combo.addItem(NO_BACKENDS_AVAILABLE)
            self.backend_combo.setEnabled(False)
        self.logger.info(FOUND_VALID_BACKENDS.format(self.backend_combo.count()))

    def update_base_model_visibility(self, index):
        is_gguf = self.lora_output_type_combo.itemText(index) == "GGUF"
        self.base_model_wrapper.setVisible(is_gguf)

    def save_preset(self):
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
            QMessageBox.information(
                self, PRESET_SAVED, PRESET_SAVED_TO.format(file_name)
            )
        self.logger.info(PRESET_SAVED_TO.format(file_name))

    def load_preset(self):
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
            self.leave_output_tensor.setChecked(
                preset.get("leave_output_tensor", False)
            )
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
            self.token_embedding_type.setCurrentText(
                preset.get("token_embedding_type", "")
            )
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

    def save_task_preset(self, task_item):
        self.logger.info(SAVING_TASK_PRESET.format(task_item.task_name))
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                preset = {
                    "command": thread.command,
                    "backend_path": thread.cwd,
                    "log_file": thread.log_file,
                }
                file_name, _ = QFileDialog.getSaveFileName(
                    self, SAVE_TASK_PRESET, "", JSON_FILES
                )
                if file_name:
                    with open(file_name, "w") as f:
                        json.dump(preset, f, indent=4)
                    QMessageBox.information(
                        self, TASK_PRESET_SAVED, TASK_PRESET_SAVED_TO.format(file_name)
                    )
                break

    def browse_export_lora_model(self):
        self.logger.info(BROWSING_FOR_EXPORT_LORA_MODEL_FILE)
        model_file, _ = QFileDialog.getOpenFileName(
            self, SELECT_MODEL_FILE, "", GGUF_FILES
        )
        if model_file:
            self.export_lora_model.setText(os.path.abspath(model_file))

    def browse_export_lora_output(self):
        self.logger.info(BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE)
        output_file, _ = QFileDialog.getSaveFileName(
            self, SELECT_OUTPUT_FILE, "", GGUF_FILES
        )
        if output_file:
            self.export_lora_output.setText(os.path.abspath(output_file))

    def add_lora_adapter(self):
        self.logger.info(ADDING_LORA_ADAPTER)
        adapter_path, _ = QFileDialog.getOpenFileName(
            self, SELECT_LORA_ADAPTER_FILE, "", LORA_FILES
        )
        if adapter_path:
            # Create a widget to hold the path and scale input
            adapter_widget = QWidget()
            adapter_layout = QHBoxLayout(adapter_widget)

            path_input = QLineEdit(adapter_path)
            path_input.setReadOnly(True)
            adapter_layout.addWidget(path_input)

            scale_input = QLineEdit("1.0")  # Default scale value
            adapter_layout.addWidget(scale_input)

            delete_button = QPushButton(DELETE_ADAPTER)
            delete_button.clicked.connect(
                lambda: self.delete_lora_adapter_item(adapter_widget)
            )
            adapter_layout.addWidget(delete_button)

            # Add the widget to the list
            list_item = QListWidgetItem(self.export_lora_adapters)
            list_item.setSizeHint(adapter_widget.sizeHint())
            self.export_lora_adapters.addItem(list_item)
            self.export_lora_adapters.setItemWidget(list_item, adapter_widget)

    def browse_base_model(self):
        self.logger.info(BROWSING_FOR_BASE_MODEL_FOLDER)  # Updated log message
        base_model_folder = QFileDialog.getExistingDirectory(
            self, SELECT_BASE_MODEL_FOLDER
        )
        if base_model_folder:
            self.base_model_path.setText(os.path.abspath(base_model_folder))

    def delete_lora_adapter_item(self, adapter_widget):
        self.logger.info(DELETING_LORA_ADAPTER)
        # Find the QListWidgetItem containing the adapter_widget
        for i in range(self.export_lora_adapters.count()):
            item = self.export_lora_adapters.item(i)
            if self.export_lora_adapters.itemWidget(item) == adapter_widget:
                self.export_lora_adapters.takeItem(i)  # Remove the item
                break

    def browse_hf_model_input(self):
        self.logger.info(BROWSE_FOR_HF_MODEL_DIRECTORY)
        model_dir = QFileDialog.getExistingDirectory(self, SELECT_HF_MODEL_DIRECTORY)
        if model_dir:
            self.hf_model_input.setText(os.path.abspath(model_dir))

    def browse_hf_outfile(self):
        self.logger.info(BROWSE_FOR_HF_TO_GGUF_OUTPUT)
        outfile, _ = QFileDialog.getSaveFileName(
            self, SELECT_OUTPUT_FILE, "", GGUF_FILES
        )
        if outfile:
            self.hf_outfile.setText(os.path.abspath(outfile))

    def convert_hf_to_gguf(self):
        self.logger.info(STARTING_HF_TO_GGUF_CONVERSION)
        try:
            model_dir = self.hf_model_input.text()
            if not model_dir:
                raise ValueError(MODEL_DIRECTORY_REQUIRED)

            command = ["python", "src/convert_hf_to_gguf.py"]

            if self.hf_vocab_only.isChecked():
                command.append("--vocab-only")

            if self.hf_outfile.text():
                command.extend(["--outfile", self.hf_outfile.text()])

            command.extend(["--outtype", self.hf_outtype.currentText()])

            if self.hf_use_temp_file.isChecked():
                command.append("--use-temp-file")

            if self.hf_no_lazy.isChecked():
                command.append("--no-lazy")

            if self.hf_model_name.text():
                command.extend(["--model-name", self.hf_model_name.text()])

            if self.hf_verbose.isChecked():
                command.append("--verbose")

            if self.hf_split_max_size.text():
                command.extend(["--split-max-size", self.hf_split_max_size.text()])

            if self.hf_dry_run.isChecked():
                command.append("--dry-run")

            command.append(model_dir)

            logs_path = self.logs_input.text()
            ensure_directory(logs_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_path, f"hf_to_gguf_{timestamp}.log")

            # Log command
            command_str = " ".join(command)
            self.logger.info(HF_TO_GGUF_CONVERSION_COMMAND.format(command_str))

            thread = QuantizationThread(command, os.getcwd(), log_file)
            self.quant_threads.append(thread)

            task_name = CONVERTING_TO_GGUF.format(os.path.basename(model_dir))
            task_item = TaskListItem(task_name, log_file, show_progress_bar=False)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)

            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(
                lambda: self.task_finished(thread, task_item)
            )
            thread.error_signal.connect(
                lambda err: handle_error(self.logger, err, task_item)
            )
            thread.start()

        except Exception as e:
            show_error(self.logger, ERROR_STARTING_HF_TO_GGUF_CONVERSION.format(str(e)))
        self.logger.info(HF_TO_GGUF_CONVERSION_TASK_STARTED)

    def export_lora(self):
        self.logger.info(STARTING_LORA_EXPORT)
        try:
            model_path = self.export_lora_model.text()
            output_path = self.export_lora_output.text()
            lora_adapters = []

            for i in range(self.export_lora_adapters.count()):
                item = self.export_lora_adapters.item(i)
                adapter_widget = self.export_lora_adapters.itemWidget(item)
                path_input = adapter_widget.layout().itemAt(0).widget()
                scale_input = adapter_widget.layout().itemAt(1).widget()
                adapter_path = path_input.text()
                adapter_scale = scale_input.text()
                lora_adapters.append((adapter_path, adapter_scale))

            if not model_path:
                raise ValueError(MODEL_PATH_REQUIRED)
            if not output_path:
                raise ValueError(OUTPUT_PATH_REQUIRED)
            if not lora_adapters:
                raise ValueError(AT_LEAST_ONE_LORA_ADAPTER_REQUIRED)

            backend_path = self.backend_combo.currentData()
            if not backend_path:
                raise ValueError(NO_BACKEND_SELECTED)

            command = [
                os.path.join(backend_path, "llama-export-lora"),
                "--model",
                model_path,
                "--output",
                output_path,
            ]

            for adapter_path, adapter_scale in lora_adapters:
                if adapter_path:
                    if adapter_scale:
                        try:
                            scale_value = float(adapter_scale)
                            command.extend(
                                ["--lora-scaled", adapter_path, str(scale_value)]
                            )
                        except ValueError:
                            raise ValueError(INVALID_LORA_SCALE_VALUE)
                    else:
                        command.extend(["--lora", adapter_path])

            threads = self.export_lora_threads.value()
            command.extend(["--threads", str(threads)])

            logs_path = self.logs_input.text()
            ensure_directory(logs_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_path, f"lora_export_{timestamp}.log")

            command_str = " ".join(command)
            self.logger.info(f"{LORA_EXPORT_COMMAND}: {command_str}")

            thread = QuantizationThread(command, backend_path, log_file)
            self.quant_threads.append(thread)

            task_item = TaskListItem(EXPORTING_LORA, log_file, show_progress_bar=False)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)

            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(lambda: self.task_finished(thread))
            thread.error_signal.connect(
                lambda err: handle_error(self.logger, err, task_item)
            )
            thread.start()
            self.logger.info(LORA_EXPORT_TASK_STARTED)
        except ValueError as e:
            show_error(self.logger, str(e))
        except Exception as e:
            show_error(self.logger, ERROR_STARTING_LORA_EXPORT.format(str(e)))

    def restart_task(self, task_item):
        self.logger.info(RESTARTING_TASK.format(task_item.task_name))
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                new_thread = QuantizationThread(
                    thread.command, thread.cwd, thread.log_file
                )
                self.quant_threads.append(new_thread)
                new_thread.status_signal.connect(task_item.update_status)
                new_thread.finished_signal.connect(
                    lambda: self.task_finished(new_thread)
                )
                new_thread.error_signal.connect(
                    lambda err: handle_error(self.logger, err, task_item)
                )
                new_thread.model_info_signal.connect(self.update_model_info)
                new_thread.start()
                task_item.update_status(IN_PROGRESS)
                break

    def browse_lora_input(self):
        self.logger.info(BROWSING_FOR_LORA_INPUT_DIRECTORY)
        lora_input_path = QFileDialog.getExistingDirectory(
            self, SELECT_LORA_INPUT_DIRECTORY
        )
        if lora_input_path:
            self.lora_input.setText(os.path.abspath(lora_input_path))
            ensure_directory(lora_input_path)

    def browse_lora_output(self):
        self.logger.info(BROWSING_FOR_LORA_OUTPUT_FILE)
        lora_output_file, _ = QFileDialog.getSaveFileName(
            self, SELECT_LORA_OUTPUT_FILE, "", GGUF_AND_BIN_FILES
        )
        if lora_output_file:
            self.lora_output.setText(os.path.abspath(lora_output_file))

    def convert_lora(self):
        self.logger.info(STARTING_LORA_CONVERSION)
        try:
            lora_input_path = self.lora_input.text()
            lora_output_path = self.lora_output.text()
            lora_output_type = self.lora_output_type_combo.currentText()

            if not lora_input_path:
                raise ValueError(LORA_INPUT_PATH_REQUIRED)
            if not lora_output_path:
                raise ValueError(LORA_OUTPUT_PATH_REQUIRED)

            if lora_output_type == "GGUF":  # Use new file and parameters for GGUF
                command = [
                    "python",
                    "src/convert_lora_to_gguf.py",
                    "--outfile",
                    lora_output_path,
                    lora_input_path,
                ]
                base_model_path = self.base_model_path.text()
                if not base_model_path:
                    raise ValueError(BASE_MODEL_PATH_REQUIRED)
                command.extend(["--base", base_model_path])
            else:  # Use old GGML parameters for GGML
                command = ["python", "src/convert_lora_to_ggml.py", lora_input_path]

            logs_path = self.logs_input.text()
            ensure_directory(logs_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_path, f"lora_conversion_{timestamp}.log")

            command_str = " ".join(command)
            self.logger.info(f"{LORA_CONVERSION_COMMAND}: {command_str}")

            thread = QuantizationThread(command, os.getcwd(), log_file)
            self.quant_threads.append(thread)

            task_name = LORA_CONVERSION_FROM_TO.format(
                os.path.basename(lora_input_path), os.path.basename(lora_output_path)
            )
            task_item = TaskListItem(task_name, log_file, show_progress_bar=False)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)

            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(
                lambda: self.lora_conversion_finished(
                    thread, lora_input_path, lora_output_path
                )
            )
            thread.error_signal.connect(
                lambda err: handle_error(self.logger, err, task_item)
            )
            thread.start()
            self.logger.info(LORA_CONVERSION_TASK_STARTED)
        except ValueError as e:
            show_error(self.logger, str(e))
        except Exception as e:
            show_error(self.logger, ERROR_STARTING_LORA_CONVERSION.format(str(e)))

    def lora_conversion_finished(self, thread, input_path, output_path):
        self.logger.info(LORA_CONVERSION_FINISHED)
        if thread in self.quant_threads:
            self.quant_threads.remove(thread)
        try:
            # Only move the file if the output type is GGML
            if self.lora_output_type_combo.currentText() == "GGML":
                source_file = os.path.join(input_path, "ggml-adapter-model.bin")
                if os.path.exists(source_file):
                    shutil.move(source_file, output_path)
                    self.logger.info(LORA_FILE_MOVED.format(source_file, output_path))
                else:
                    self.logger.warning(LORA_FILE_NOT_FOUND.format(source_file))
        except Exception as e:
            self.logger.error(ERROR_MOVING_LORA_FILE.format(str(e)))

    def download_finished(self, extract_dir):
        self.logger.info(DOWNLOAD_FINISHED_EXTRACTED_TO.format(extract_dir))
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)

        if (
            self.cuda_extract_checkbox.isChecked()
            and self.cuda_extract_checkbox.isVisible()
        ):
            cuda_backend = self.backend_combo_cuda.currentData()
            if cuda_backend and cuda_backend != NO_SUITABLE_CUDA_BACKENDS:
                self.extract_cuda_files(extract_dir, cuda_backend)
                QMessageBox.information(
                    self,
                    DOWNLOAD_COMPLETE,
                    LLAMACPP_DOWNLOADED_AND_EXTRACTED.format(extract_dir, cuda_backend),
                )
            else:
                QMessageBox.warning(
                    self, CUDA_EXTRACTION_FAILED, NO_SUITABLE_CUDA_BACKEND_FOUND
                )
        else:
            QMessageBox.information(
                self,
                DOWNLOAD_COMPLETE,
                LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED.format(extract_dir),
            )

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
            response = requests.get(
                "https://api.github.com/repos/ggerganov/llama.cpp/releases"
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            releases = response.json()
            self.release_combo.clear()
            for release in releases:
                self.release_combo.addItem(release["tag_name"], userData=release)
            self.release_combo.currentIndexChanged.connect(self.update_assets)
            self.update_assets()
        except requests.exceptions.RequestException as e:
            show_error(self.logger, ERROR_FETCHING_RELEASES.format(str(e)))

    def update_assets(self):
        self.logger.debug(UPDATING_ASSET_LIST)
        self.asset_combo.clear()
        release = self.release_combo.currentData()
        if release:
            if "assets" in release:
                for asset in release["assets"]:
                    self.asset_combo.addItem(asset["name"], userData=asset)
            else:
                show_error(
                    self.logger, NO_ASSETS_FOUND_FOR_RELEASE.format(release["tag_name"])
                )
        self.update_cuda_option()

    def download_llama_cpp(self):
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

    def update_cuda_option(self):
        self.logger.debug(UPDATING_CUDA_OPTIONS)
        asset = self.asset_combo.currentData()

        # Handle the case where asset is None
        if asset is None:
            self.logger.warning(NO_ASSET_SELECTED_FOR_CUDA_CHECK)
            self.cuda_extract_checkbox.setVisible(False)
            self.cuda_backend_label.setVisible(False)
            self.backend_combo_cuda.setVisible(False)
            return  # Exit the function early

        is_cuda = asset and "cudart" in asset["name"].lower()
        self.cuda_extract_checkbox.setVisible(is_cuda)
        self.cuda_backend_label.setVisible(is_cuda)
        self.backend_combo_cuda.setVisible(is_cuda)
        if is_cuda:
            self.update_cuda_backends()

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

        if (
            self.cuda_extract_checkbox.isChecked()
            and self.cuda_extract_checkbox.isVisible()
        ):
            cuda_backend = self.backend_combo_cuda.currentData()
            if cuda_backend:
                self.extract_cuda_files(extract_dir, cuda_backend)
                QMessageBox.information(
                    self,
                    DOWNLOAD_COMPLETE,
                    LLAMACPP_DOWNLOADED_AND_EXTRACTED.format(extract_dir, cuda_backend),
                )
            else:
                QMessageBox.warning(
                    self, CUDA_EXTRACTION_FAILED, NO_CUDA_BACKEND_SELECTED
                )
        else:
            QMessageBox.information(
                self,
                DOWNLOAD_COMPLETE,
                LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED.format(extract_dir),
            )

        self.refresh_backends()

    def extract_cuda_files(self, extract_dir, destination):
        self.logger.info(EXTRACTING_CUDA_FILES.format(extract_dir, destination))
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".dll"):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(destination, file)
                    shutil.copy2(source_path, dest_path)

    def download_error(self, error_message):
        self.logger.error(DOWNLOAD_ERROR.format(error_message))
        self.download_button.setEnabled(True)
        self.download_progress.setValue(0)
        show_error(self.logger, DOWNLOAD_FAILED.format(error_message))

        # Clean up any partially downloaded files
        asset = self.asset_combo.currentData()
        if asset:
            partial_file = os.path.join(os.path.abspath("llama_bin"), asset["name"])
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

    def update_threads_spinbox(self, value):
        self.threads_spinbox.setValue(value)

    def update_threads_slider(self, value):
        self.threads_slider.setValue(value)

    def update_gpu_offload_spinbox(self, value):
        self.gpu_offload_spinbox.setValue(value)

    def update_gpu_offload_slider(self, value):
        self.gpu_offload_slider.setValue(value)

    def toggle_gpu_offload_auto(self, state):
        is_auto = state == Qt.CheckState.Checked
        self.gpu_offload_slider.setEnabled(not is_auto)
        self.gpu_offload_spinbox.setEnabled(not is_auto)

    def cancel_task_by_item(self, item):
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                thread.terminate()
                task_item.update_status(CANCELED)
                self.quant_threads.remove(thread)
                break

    def cancel_task(self, item):
        self.logger.info(CANCELLING_TASK.format(item.text()))
        self.cancel_task_by_item(item)

    def delete_task(self, item):
        self.logger.info(DELETING_TASK.format(item.text()))

        # Cancel the task first
        self.cancel_task_by_item(item)

        reply = QMessageBox.question(
            self,
            CONFIRM_DELETION_TITLE,
            CONFIRM_DELETION,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            task_item = self.task_list.itemWidget(item)
            row = self.task_list.row(item)
            self.task_list.takeItem(row)

            if task_item:
                task_item.deleteLater()

    def create_label(self, text, tooltip):
        label = QLabel(text)
        label.setToolTip(tooltip)
        return label

    def load_models(self):
        self.logger.info(LOADING_MODELS)
        models_dir = self.models_input.text()
        ensure_directory(models_dir)
        self.model_tree.clear()

        sharded_models = {}
        single_models = []

        # Regex pattern to match sharded model filenames
        shard_pattern = re.compile(r"(.*)-(\d+)-of-(\d+)\.gguf$")

        for file in os.listdir(models_dir):
            if file.endswith(".gguf"):
                match = shard_pattern.match(file)
                if match:
                    # This is a sharded model
                    base_name, shard_num, total_shards = match.groups()
                    if base_name not in sharded_models:
                        sharded_models[base_name] = []
                    sharded_models[base_name].append((int(shard_num), file))
                else:
                    single_models.append(file)

        # Add sharded models
        for base_name, shards in sharded_models.items():
            parent_item = QTreeWidgetItem(self.model_tree)
            parent_item.setText(0, f"{base_name} ({SHARDED})")
            # Sort shards by shard number and get the first one
            first_shard = sorted(shards, key=lambda x: x[0])[0][1]
            parent_item.setData(0, Qt.ItemDataRole.UserRole, first_shard)
            for _, shard_file in sorted(shards):
                child_item = QTreeWidgetItem(parent_item)
                child_item.setText(0, shard_file)
                child_item.setData(0, Qt.ItemDataRole.UserRole, shard_file)

        # Add single models
        for model in sorted(single_models):
            item = QTreeWidgetItem(self.model_tree)
            item.setText(0, model)
            item.setData(0, Qt.ItemDataRole.UserRole, model)

        self.model_tree.expandAll()
        self.logger.info(LOADED_MODELS.format(len(single_models) + len(sharded_models)))

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
        imatrix_file, _ = QFileDialog.getOpenFileName(
            self, SELECT_IMATRIX_FILE, "", DAT_FILES
        )
        if imatrix_file:
            self.imatrix.setText(os.path.abspath(imatrix_file))

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
        if not self.model_tree.currentItem():
            errors.append(NO_MODEL_SELECTED)

        if errors:
            raise ValueError("\n".join(errors))

    def update_system_info(self):
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        self.ram_bar.setValue(int(ram.percent))
        self.ram_bar.setFormat(
            RAM_USAGE_FORMAT.format(
                ram.percent, ram.used // 1024 // 1024, ram.total // 1024 // 1024
            )
        )
        self.cpu_label.setText(CPU_USAGE_FORMAT.format(cpu))

    def add_kv_override(self, override_string=None):
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

    def remove_kv_override(self, entry):
        self.kv_override_layout.removeWidget(entry)
        self.kv_override_entries.remove(entry)
        entry.deleteLater()

    def quantize_model(self):
        self.logger.info(STARTING_MODEL_QUANTIZATION)
        try:
            self.validate_quantization_inputs()
            selected_item = self.model_tree.currentItem()
            if not selected_item:
                raise ValueError(NO_MODEL_SELECTED)

            model_file = selected_item.data(0, Qt.ItemDataRole.UserRole)
            model_name = selected_item.text(0).replace(" (sharded)", "")

            backend_path = self.backend_combo.currentData()
            if not backend_path:
                raise ValueError(NO_BACKEND_SELECTED)

            selected_quant_types = [
                item.text() for item in self.quant_type.selectedItems()
            ]
            if not selected_quant_types:
                raise ValueError(NO_QUANTIZATION_TYPE_SELECTED)

            input_path = os.path.join(self.models_input.text(), model_file)
            if not os.path.exists(input_path):
                raise FileNotFoundError(INPUT_FILE_NOT_EXIST.format(input_path))

            tasks = []  # List to keep track of all tasks

            for quant_type in selected_quant_types:
                # Start building the output name
                output_name_parts = [
                    os.path.splitext(model_name)[0],
                    "converted",
                    quant_type,
                ]

                # Check for output tensor options
                if (
                    self.use_output_tensor_type.isChecked()
                    or self.leave_output_tensor.isChecked()
                ):
                    output_tensor_part = "o"
                    if self.use_output_tensor_type.isChecked():
                        output_tensor_part += (
                            "." + self.output_tensor_type.currentText()
                        )
                    output_name_parts.append(output_tensor_part)

                # Check for embedding tensor options
                if self.use_token_embedding_type.isChecked():
                    embd_tensor_part = "t." + self.token_embedding_type.currentText()
                    output_name_parts.append(embd_tensor_part)

                # Check for pure option
                if self.pure.isChecked():
                    output_name_parts.append("pure")

                # Check for requantize option
                if self.allow_requantize.isChecked():
                    output_name_parts.append("rq")

                # Check for KV override
                if any(
                    entry.get_override_string() for entry in self.kv_override_entries
                ):
                    output_name_parts.append("kv")

                # Join all parts with underscores and add .gguf extension
                output_name = "_".join(output_name_parts) + ".gguf"
                output_path = os.path.join(self.output_input.text(), output_name)

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
                    command.extend(
                        [
                            "--output-tensor-type",
                            self.output_tensor_type.currentText().lower(),
                        ]
                    )
                if self.use_token_embedding_type.isChecked():
                    command.extend(
                        [
                            "--token-embedding-type",
                            self.token_embedding_type.currentText().lower(),
                        ]
                    )
                if self.keep_split.isChecked():
                    command.append("--keep-split")
                if self.kv_override_entries:
                    for entry in self.kv_override_entries:
                        override_string = entry.get_override_string(
                            model_name=model_name,
                            quant_type=quant_type,
                            output_path=output_path,
                        )
                        if override_string:
                            command.extend(["--override-kv", override_string])

                command.extend([input_path, output_path, quant_type])

                # Add extra arguments
                if self.extra_arguments.text():
                    command.extend(self.extra_arguments.text().split())

                logs_path = self.logs_input.text()
                ensure_directory(logs_path)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(
                    logs_path, f"{model_name}_{timestamp}_{quant_type}.log"
                )

                # Log quant command
                command_str = " ".join(command)
                self.logger.info(f"{QUANTIZATION_COMMAND}: {command_str}")

                thread = QuantizationThread(command, backend_path, log_file)
                self.quant_threads.append(thread)

                task_item = TaskListItem(
                    QUANTIZING_MODEL_TO.format(model_name, quant_type), log_file
                )
                list_item = QListWidgetItem(self.task_list)
                list_item.setSizeHint(task_item.sizeHint())
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, task_item)

                tasks.append(
                    (thread, task_item)
                )  # Add the thread and task_item to our list

                # Connect the output signal to the new progress parsing function
                thread.output_signal.connect(
                    lambda line, ti=task_item: self.parse_progress(line, ti)
                )
                thread.status_signal.connect(task_item.update_status)
                thread.finished_signal.connect(
                    lambda t=thread, ti=task_item: self.task_finished(t, ti)
                )
                thread.error_signal.connect(
                    lambda err, ti=task_item: handle_error(self.logger, err, ti)
                )
                thread.model_info_signal.connect(self.update_model_info)

            # Start all threads after setting them up
            for thread, _ in tasks:
                thread.start()
                self.logger.info(QUANTIZATION_TASK_STARTED.format(model_name))

        except ValueError as e:
            show_error(self.logger, str(e))
        except FileNotFoundError as e:
            show_error(self.logger, str(e))
        except Exception as e:
            show_error(self.logger, ERROR_STARTING_QUANTIZATION.format(str(e)))

    def update_model_info(self, model_info):
        self.logger.debug(UPDATING_MODEL_INFO.format(model_info))
        pass

    def parse_progress(self, line, task_item):
        # Parses the output line for progress information and updates the task item.
        match = re.search(r"\[(\d+)/(\d+)\]", line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            progress = int((current / total) * 100)
            task_item.update_progress(progress)

    def task_finished(self, thread, task_item):
        self.logger.info(TASK_FINISHED.format(thread.log_file))
        if thread in self.quant_threads:
            self.quant_threads.remove(thread)
        task_item.update_status(COMPLETED)

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
                with open_file_safe(task_item.log_file, "r") as f:
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
        model_file, _ = QFileDialog.getOpenFileName(
            self, SELECT_MODEL_FILE, "", GGUF_FILES
        )
        if model_file:
            self.imatrix_model.setText(os.path.abspath(model_file))

    def browse_imatrix_output(self):
        self.logger.info(BROWSING_FOR_IMATRIX_OUTPUT_FILE)
        output_file, _ = QFileDialog.getSaveFileName(
            self, SELECT_OUTPUT_FILE, "", DAT_FILES
        )
        if output_file:
            self.imatrix_output.setText(os.path.abspath(output_file))

    def get_models_data(self):
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

    def get_tasks_data(self):
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

    def generate_imatrix(self):
        self.logger.info(STARTING_IMATRIX_GENERATION)
        try:
            backend_path = self.backend_combo.currentData()
            if not os.path.exists(backend_path):
                raise FileNotFoundError(BACKEND_PATH_NOT_EXIST.format(backend_path))

            # Check if the Model area is empty
            if not self.imatrix_model.text():
                raise ValueError(MODEL_PATH_REQUIRED_FOR_IMATRIX)

            command = [
                os.path.join(backend_path, "llama-imatrix"),
                "-f",
                self.imatrix_datafile.text(),
                "-m",
                self.imatrix_model.text(),
                "-o",
                self.imatrix_output.text(),
                "--output-frequency",
                str(self.imatrix_frequency.value()),
                "--ctx-size",
                str(self.imatrix_ctx_size.value()),
                "--threads",
                str(self.threads_spinbox.value()),
            ]

            if self.gpu_offload_auto.isChecked():
                command.extend(["-ngl", "99"])
            elif self.gpu_offload_spinbox.value() > 0:
                command.extend(["-ngl", str(self.gpu_offload_spinbox.value())])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.logs_input.text(), f"imatrix_{timestamp}.log")

            # Log command
            command_str = " ".join(command)
            self.logger.info(f"{IMATRIX_GENERATION_COMMAND}: {command_str}")

            thread = QuantizationThread(command, backend_path, log_file)
            self.quant_threads.append(thread)

            task_name = GENERATING_IMATRIX_FOR.format(
                os.path.basename(self.imatrix_model.text())
            )
            task_item = TaskListItem(task_name, log_file, show_progress_bar=False)
            list_item = QListWidgetItem(self.task_list)
            list_item.setSizeHint(task_item.sizeHint())
            self.task_list.addItem(list_item)
            self.task_list.setItemWidget(list_item, task_item)

            thread.status_signal.connect(task_item.update_status)
            thread.finished_signal.connect(lambda: self.task_finished(thread))
            thread.error_signal.connect(
                lambda err: handle_error(self.logger, err, task_item)
            )
            thread.start()
        except Exception as e:
            show_error(self.logger, ERROR_STARTING_IMATRIX_GENERATION.format(str(e)))
        self.logger.info(IMATRIX_GENERATION_TASK_STARTED)

    def closeEvent(self, event: QCloseEvent):
        self.logger.info(APPLICATION_CLOSING)
        if self.quant_threads:
            reply = QMessageBox.question(
                self,
                WARNING,
                TASK_RUNNING_WARNING,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                for thread in self.quant_threads:
                    thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
        self.logger.info(APPLICATION_CLOSED)
