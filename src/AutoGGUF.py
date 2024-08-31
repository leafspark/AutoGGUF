import importlib
import json
import re
import shutil
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Tuple

import requests
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from dotenv import load_dotenv

import lora_conversion
import presets
import ui_update
import utils
from CustomTitleBar import CustomTitleBar
from GPUMonitor import GPUMonitor
from Localizations import *
from Logger import Logger
from QuantizationThread import QuantizationThread
from TaskListItem import TaskListItem
from error_handling import handle_error, show_error
from imports_and_globals import (
    ensure_directory,
    open_file_safe,
    resource_path,
    show_about,
)


class AutoGGUF(QMainWindow):

    def __init__(self, args: List[str]) -> None:
        super().__init__()
        self.logger = Logger("AutoGGUF", "logs")

        width, height = self.parse_resolution()

        self.logger.info(INITIALIZING_AUTOGGUF)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QIcon(resource_path("assets/favicon.ico")))
        self.setGeometry(100, 100, width, height)
        self.setWindowFlag(Qt.FramelessWindowHint)

        load_dotenv()  # Loads the .env file

        # Configuration
        self.model_dir_name = os.environ.get("AUTOGGUF_MODEL_DIR_NAME", "models")
        self.output_dir_name = os.environ.get(
            "AUTOGGUF_OUTPUT_DIR_NAME", "quantized_models"
        )

        self.resize_factor = float(
            os.environ.get("AUTOGGUF_RESIZE_FACTOR", 1.1)
        )  # 10% increase/decrease
        self.default_width, self.default_height = self.parse_resolution()
        self.resize(self.default_width, self.default_height)

        ensure_directory(os.path.abspath(self.output_dir_name))
        ensure_directory(os.path.abspath(self.model_dir_name))

        # References
        self.update_base_model_visibility = partial(
            ui_update.update_base_model_visibility, self
        )
        self.update_assets = ui_update.update_assets.__get__(self)
        self.update_cuda_option = ui_update.update_cuda_option.__get__(self)
        self.update_cuda_backends = ui_update.update_cuda_backends.__get__(self)
        self.download_llama_cpp = utils.download_llama_cpp.__get__(self)
        self.refresh_releases = utils.refresh_releases.__get__(self)
        self.browse_lora_input = utils.browse_lora_input.__get__(self)
        self.browse_lora_output = utils.browse_lora_output.__get__(self)
        self.convert_lora = lora_conversion.convert_lora.__get__(self)
        self.show_about = show_about.__get__(self)
        self.save_preset = presets.save_preset.__get__(self)
        self.load_preset = presets.load_preset.__get__(self)
        self.browse_export_lora_model = (
            lora_conversion.browse_export_lora_model.__get__(self)
        )
        self.browse_export_lora_output = (
            lora_conversion.browse_export_lora_output.__get__(self)
        )
        self.add_lora_adapter = lora_conversion.add_lora_adapter.__get__(self)
        self.export_lora = lora_conversion.export_lora.__get__(self)
        self.browse_models = utils.browse_models.__get__(self)
        self.browse_output = utils.browse_output.__get__(self)
        self.browse_logs = utils.browse_logs.__get__(self)
        self.browse_imatrix = utils.browse_imatrix.__get__(self)
        self.get_models_data = utils.get_models_data.__get__(self)
        self.get_tasks_data = utils.get_tasks_data.__get__(self)
        self.add_kv_override = partial(utils.add_kv_override, self)
        self.remove_kv_override = partial(utils.remove_kv_override, self)
        self.cancel_task = partial(TaskListItem.cancel_task, self)
        self.delete_task = partial(TaskListItem.delete_task, self)
        self.show_task_context_menu = partial(TaskListItem.show_task_context_menu, self)
        self.show_task_properties = partial(TaskListItem.show_task_properties, self)
        self.cancel_task_by_item = partial(TaskListItem.cancel_task_by_item, self)
        self.toggle_gpu_offload_auto = partial(ui_update.toggle_gpu_offload_auto, self)
        self.update_threads_spinbox = partial(ui_update.update_threads_spinbox, self)
        self.update_threads_slider = partial(ui_update.update_threads_slider, self)
        self.update_gpu_offload_spinbox = partial(
            ui_update.update_gpu_offload_spinbox, self
        )
        self.update_gpu_offload_slider = partial(
            ui_update.update_gpu_offload_slider, self
        )
        self.update_model_info = partial(ui_update.update_model_info, self.logger, self)
        self.update_system_info = partial(ui_update.update_system_info, self)
        self.update_download_progress = partial(
            ui_update.update_download_progress, self
        )
        self.delete_lora_adapter_item = partial(
            lora_conversion.delete_lora_adapter_item, self
        )

        # Set up main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Custom title bar
        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)

        # Menu bar
        self.menubar = QMenuBar()
        self.title_bar.layout().insertWidget(1, self.menubar)

        # File menu
        file_menu = self.menubar.addMenu("&File")
        close_action = QAction("&Close", self)
        close_action.setShortcut(QKeySequence("Alt+F4"))
        close_action.triggered.connect(self.close)
        save_preset_action = QAction("&Save Preset", self)
        save_preset_action.setShortcut(QKeySequence("Ctrl+S"))
        save_preset_action.triggered.connect(self.save_preset)
        load_preset_action = QAction("&Load Preset", self)
        load_preset_action.setShortcut(QKeySequence("Ctrl+S"))
        load_preset_action.triggered.connect(self.load_preset)
        file_menu.addAction(close_action)
        file_menu.addAction(save_preset_action)
        file_menu.addAction(load_preset_action)

        # Help menu
        help_menu = self.menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.setShortcut(QKeySequence("Ctrl+Q"))
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Content widget
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)

        # Wrap content in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow content to resize
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        self.setCentralWidget(main_widget)

        # Styling
        self.setStyleSheet(
            """
                AutoGGUF {
                    background-color: #2b2b2b;
                    border-radius: 10px;
                }
            """
        )

        # Initialize threads
        self.quant_threads = []

        # Timer for updating system info
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(200)

        # Add all widgets to content_layout
        left_widget = QWidget()
        right_widget = QWidget()
        left_widget.setMinimumWidth(1100)
        right_widget.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_widget)
        right_layout = QVBoxLayout(right_widget)
        content_layout.addWidget(left_widget)
        content_layout.addWidget(right_widget)

        # System info
        self.ram_bar = QProgressBar()
        self.cpu_bar = QProgressBar()
        self.cpu_label = QLabel()
        self.gpu_monitor = GPUMonitor()
        left_layout.addWidget(QLabel(RAM_USAGE))
        left_layout.addWidget(self.ram_bar)
        left_layout.addWidget(QLabel(CPU_USAGE))
        left_layout.addWidget(self.cpu_bar)
        left_layout.addWidget(QLabel(GPU_USAGE))
        left_layout.addWidget(self.gpu_monitor)

        # Backend selection
        backend_layout = QHBoxLayout()
        self.backend_combo = QComboBox()
        self.refresh_backends_button = QPushButton(REFRESH_BACKENDS)
        self.refresh_backends_button.clicked.connect(self.refresh_backends)
        backend_layout.addWidget(QLabel(BACKEND))
        backend_layout.addWidget(self.backend_combo)
        backend_layout.addWidget(self.refresh_backends_button)
        left_layout.addLayout(backend_layout)

        # Download llama.cpp section
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

        # Models path
        models_layout = QHBoxLayout()
        self.models_input = QLineEdit(os.path.abspath(self.model_dir_name))
        models_button = QPushButton(BROWSE)
        models_button.clicked.connect(self.browse_models)
        models_layout.addWidget(QLabel(MODELS_PATH))
        models_layout.addWidget(self.models_input)
        models_layout.addWidget(models_button)
        left_layout.addLayout(models_layout)

        # Output path
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit(os.path.abspath(self.output_dir_name))
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

        # Import Model button
        import_model_button = QPushButton(IMPORT_MODEL)
        import_model_button.clicked.connect(self.import_model)
        left_layout.addWidget(import_model_button)

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

        self.extra_arguments = QLineEdit()
        quant_options_layout.addRow(
            self.create_label(EXTRA_ARGUMENTS, "Additional command-line arguments"),
            self.extra_arguments,
        )

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
        self.imatrix_frequency.setRange(1, 100)
        self.imatrix_frequency.setValue(1)
        imatrix_layout.addRow(
            self.create_label(OUTPUT_FREQUENCY, HOW_OFTEN_TO_SAVE_IMATRIX),
            self.imatrix_frequency,
        )

        self.imatrix_ctx_size = QSpinBox()
        self.imatrix_ctx_size.setRange(1, 1048576)
        self.imatrix_ctx_size.setValue(512)
        imatrix_layout.addRow(
            self.create_label(CONTEXT_SIZE, CONTEXT_SIZE_FOR_IMATRIX),
            self.imatrix_ctx_size,
        )

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

        self.lora_output_type_combo = QComboBox()
        self.lora_output_type_combo.addItems(["GGML", "GGUF"])
        self.lora_output_type_combo.currentIndexChanged.connect(
            self.update_base_model_visibility
        )
        lora_layout.addRow(
            self.create_label(OUTPUT_TYPE, SELECT_OUTPUT_TYPE),
            self.lora_output_type_combo,
        )

        self.base_model_label = self.create_label(BASE_MODEL, SELECT_BASE_MODEL_FILE)
        self.base_model_path = QLineEdit()
        base_model_button = QPushButton(BROWSE)
        base_model_button.clicked.connect(self.browse_base_model)
        base_model_layout = QHBoxLayout()
        base_model_layout.addWidget(self.base_model_path, 1)
        base_model_layout.addWidget(base_model_button)
        self.base_model_widget = QWidget()
        self.base_model_widget.setLayout(base_model_layout)

        self.base_model_wrapper = QWidget()
        wrapper_layout = QHBoxLayout(self.base_model_wrapper)
        wrapper_layout.addWidget(self.base_model_label)
        wrapper_layout.addWidget(self.base_model_widget, 1)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)

        lora_layout.addRow(self.base_model_wrapper)

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

        self.export_lora_threads = QSpinBox()
        self.export_lora_threads.setRange(1, 64)
        self.export_lora_threads.setValue(8)
        export_lora_layout.addRow(
            self.create_label(THREADS, NUMBER_OF_THREADS_FOR_LORA_EXPORT),
            self.export_lora_threads,
        )

        export_lora_button = QPushButton(EXPORT_LORA)
        export_lora_button.clicked.connect(self.export_lora)
        export_lora_layout.addRow(export_lora_button)

        export_lora_group.setLayout(export_lora_layout)
        right_layout.addWidget(export_lora_group)

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

        # Set initial state
        self.update_base_model_visibility(self.lora_output_type_combo.currentIndex())

        # Initialize releases and backends
        if os.environ.get("AUTOGGUF_CHECK_BACKEND", "").lower() == "enabled":
            self.refresh_releases()
        self.refresh_backends()

        if os.environ.get("AUTOGGUF_CHECK_UPDATE", "").lower() == "enabled":
            self.logger.info(CHECKING_FOR_UPDATES)
            self.check_for_updates()

        # Load theme based on environment variable
        theme_path = os.environ.get("AUTOGGUF_THEME")
        if theme_path:
            try:
                with open(theme_path, "r") as f:
                    theme = f.read()
                self.setStyleSheet(theme)
            except (FileNotFoundError, OSError):
                # If the specified theme file is not found or inaccessible,
                # fall back to the default theme
                with open(resource_path("assets/default.css"), "r") as f:
                    default_theme = f.read()
                self.setStyleSheet(default_theme)
        else:
            # If the environment variable is not set, use the default theme
            with open(resource_path("assets/default.css"), "r") as f:
                default_theme = f.read()
            self.setStyleSheet(default_theme)

        # Load models
        self.load_models()

        # Load plugins
        self.plugins = self.load_plugins()
        self.apply_plugins()

        self.logger.info(AUTOGGUF_INITIALIZATION_COMPLETE)

    def load_plugins(self) -> Dict[str, Dict[str, Any]]:
        plugins = {}
        plugin_dir = "plugins"

        if not os.path.exists(plugin_dir):
            self.logger.info(PLUGINS_DIR_NOT_EXIST.format(plugin_dir))
            return plugins

        if not os.path.isdir(plugin_dir):
            self.logger.warning(PLUGINS_DIR_NOT_DIRECTORY.format(plugin_dir))
            return plugins

        for file in os.listdir(plugin_dir):
            if file.endswith(".py") and not file.endswith(".disabled.py"):
                name = file[:-3]
                path = os.path.join(plugin_dir, file)

                try:
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if isinstance(item, type) and hasattr(item, "__data__"):
                            plugin_instance = item()
                            plugin_data = plugin_instance.__data__()

                            compatible_versions = plugin_data.get(
                                "compatible_versions", []
                            )
                            if (
                                "*" in compatible_versions
                                or AUTOGGUF_VERSION in compatible_versions
                            ):
                                plugins[name] = {
                                    "instance": plugin_instance,
                                    "data": plugin_data,
                                }
                                self.logger.info(
                                    PLUGIN_LOADED.format(
                                        plugin_data["name"], plugin_data["version"]
                                    )
                                )
                            else:
                                self.logger.warning(
                                    PLUGIN_INCOMPATIBLE.format(
                                        plugin_data["name"],
                                        plugin_data["version"],
                                        AUTOGGUF_VERSION,
                                        ", ".join(compatible_versions),
                                    )
                                )
                            break
                except Exception as e:
                    self.logger.error(PLUGIN_LOAD_FAILED.format(name, str(e)))

        return plugins

    def apply_plugins(self) -> None:
        if not self.plugins:
            self.logger.info(NO_PLUGINS_LOADED)
            return

        for plugin_name, plugin_info in self.plugins.items():
            plugin_instance = plugin_info["instance"]
            for attr_name in dir(plugin_instance):
                if not attr_name.startswith("__") and attr_name != "init":
                    attr_value = getattr(plugin_instance, attr_name)
                    setattr(self, attr_name, attr_value)

            if hasattr(plugin_instance, "init") and callable(plugin_instance.init):
                plugin_instance.init(self)

    def check_for_updates(self) -> None:
        try:
            response = requests.get(
                "https://api.github.com/repos/leafspark/AutoGGUF/releases/latest"
            )
            response.raise_for_status()  # Raise an exception for bad status codes

            latest_release = response.json()
            latest_version = latest_release["tag_name"].replace("v", "")

            if latest_version > AUTOGGUF_VERSION.replace("v", ""):
                self.prompt_for_update(latest_release)
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"{ERROR_CHECKING_FOR_UPDATES} {e}")

    def prompt_for_update(self, release) -> None:
        update_message = QMessageBox()
        update_message.setIcon(QMessageBox.Information)
        update_message.setWindowTitle(UPDATE_AVAILABLE)
        update_message.setText(NEW_VERSION_AVAILABLE.format(release["tag_name"]))
        update_message.setInformativeText(DOWNLOAD_NEW_VERSION)
        update_message.addButton(QMessageBox.StandardButton.Yes)
        update_message.addButton(QMessageBox.StandardButton.No)
        update_message.setDefaultButton(QMessageBox.StandardButton.Yes)

        if update_message.exec() == QMessageBox.StandardButton.Yes:
            QDesktopServices.openUrl(QUrl(release["html_url"]))

    def keyPressEvent(self, event) -> None:
        if event.modifiers() == Qt.ControlModifier:
            if (
                event.key() == Qt.Key_Equal
            ):  # Qt.Key_Plus doesn't work on some keyboards
                self.resize_window(larger=True)
            elif event.key() == Qt.Key_Minus:
                self.resize_window(larger=False)
            elif event.key() == Qt.Key_0:
                self.reset_size()
        super().keyPressEvent(event)

    def resize_window(self, larger) -> None:
        factor = 1.1 if larger else 1 / 1.1
        current_width = self.width()
        current_height = self.height()
        new_width = int(current_width * factor)
        new_height = int(current_height * factor)
        self.resize(new_width, new_height)

    def reset_size(self) -> None:
        self.resize(self.default_width, self.default_height)

    def parse_resolution(self) -> Tuple[int, int]:
        res = os.environ.get("AUTOGGUF_RESOLUTION", "1650x1100")
        try:
            width, height = map(int, res.split("x"))
            if width <= 0 or height <= 0:
                raise ValueError
            return width, height
        except (ValueError, AttributeError):
            return 1650, 1100

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 10, 10)
        mask = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(mask)

    def refresh_backends(self) -> None:
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
        self.logger.info(FOUND_VALID_BACKENDS.format(len(valid_backends)))

    def save_task_preset(self, task_item) -> None:
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

    def browse_base_model(self) -> None:
        self.logger.info(BROWSING_FOR_BASE_MODEL_FOLDER)  # Updated log message
        base_model_folder = QFileDialog.getExistingDirectory(
            self, SELECT_BASE_MODEL_FOLDER
        )
        if base_model_folder:
            self.base_model_path.setText(os.path.abspath(base_model_folder))

    def browse_hf_model_input(self) -> None:
        self.logger.info(BROWSE_FOR_HF_MODEL_DIRECTORY)
        model_dir = QFileDialog.getExistingDirectory(self, SELECT_HF_MODEL_DIRECTORY)
        if model_dir:
            self.hf_model_input.setText(os.path.abspath(model_dir))

    def browse_hf_outfile(self) -> None:
        self.logger.info(BROWSE_FOR_HF_TO_GGUF_OUTPUT)
        outfile, _ = QFileDialog.getSaveFileName(
            self, SELECT_OUTPUT_FILE, "", GGUF_FILES
        )
        if outfile:
            self.hf_outfile.setText(os.path.abspath(outfile))

    def convert_hf_to_gguf(self) -> None:
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

    def restart_task(self, task_item) -> None:
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

    def lora_conversion_finished(self, thread) -> None:
        self.logger.info(LORA_CONVERSION_FINISHED)
        if thread in self.quant_threads:
            self.quant_threads.remove(thread)

    def download_finished(self, extract_dir) -> None:
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

    def extract_cuda_files(self, extract_dir, destination) -> None:
        self.logger.info(EXTRACTING_CUDA_FILES.format(extract_dir, destination))
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".dll"):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(destination, file)
                    shutil.copy2(source_path, dest_path)

    def download_error(self, error_message) -> None:
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

    def create_label(self, text, tooltip) -> QLabel:
        label = QLabel(text)
        label.setToolTip(tooltip)
        return label

    def verify_gguf(self, file_path) -> bool:
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                return magic == b"GGUF"
        except Exception:
            return False

    def load_models(self) -> None:
        self.logger.info(LOADING_MODELS)
        models_dir = self.models_input.text()
        ensure_directory(models_dir)
        self.model_tree.clear()

        sharded_models = {}
        single_models = []
        concatenated_models = []

        shard_pattern = re.compile(r"(.*)-(\d+)-of-(\d+)\.gguf$")
        concat_pattern = re.compile(r"(.*)\.gguf\.part(\d+)of(\d+)$")

        for file in os.listdir(models_dir):
            full_path = os.path.join(models_dir, file)
            if file.endswith(".gguf"):
                if not self.verify_gguf(full_path):
                    show_error(self.logger, INVALID_GGUF_FILE.format(file))
                    continue

                match = shard_pattern.match(file)
                if match:
                    base_name, shard_num, total_shards = match.groups()
                    if base_name not in sharded_models:
                        sharded_models[base_name] = []
                    sharded_models[base_name].append((int(shard_num), file))
                else:
                    single_models.append(file)
            else:
                match = concat_pattern.match(file)
                if match:
                    concatenated_models.append(file)

        if hasattr(self, "imported_models"):
            for imported_model in self.imported_models:
                file_name = os.path.basename(imported_model)
                if (
                    file_name not in single_models
                    and file_name not in concatenated_models
                ):
                    if self.verify_gguf(imported_model):
                        single_models.append(file_name)
                    else:
                        show_error(
                            self.logger, INVALID_GGUF_FILE.format(imported_model)
                        )

        for base_name, shards in sharded_models.items():
            parent_item = QTreeWidgetItem(self.model_tree)
            parent_item.setText(0, SHARDED_MODEL_NAME.format(base_name))
            first_shard = sorted(shards, key=lambda x: x[0])[0][1]
            parent_item.setData(0, Qt.ItemDataRole.UserRole, first_shard)
            for _, shard_file in sorted(shards):
                child_item = QTreeWidgetItem(parent_item)
                child_item.setText(0, shard_file)
                child_item.setData(0, Qt.ItemDataRole.UserRole, shard_file)

        for model in sorted(single_models):
            self.add_model_to_tree(model)

        for model in sorted(concatenated_models):
            item = self.add_model_to_tree(model)
            item.setForeground(0, Qt.gray)
            item.setToolTip(0, CONCATENATED_FILE_WARNING)

        self.model_tree.expandAll()
        self.logger.info(
            LOADED_MODELS.format(
                len(single_models) + len(sharded_models) + len(concatenated_models)
            )
        )
        if concatenated_models:
            self.logger.warning(
                CONCATENATED_FILES_FOUND.format(len(concatenated_models))
            )

    def add_model_to_tree(self, model) -> QTreeWidgetItem:
        item = QTreeWidgetItem(self.model_tree)
        item.setText(0, model)
        if hasattr(self, "imported_models") and model in [
            os.path.basename(m) for m in self.imported_models
        ]:
            full_path = next(
                m for m in self.imported_models if os.path.basename(m) == model
            )
            item.setData(0, Qt.ItemDataRole.UserRole, full_path)
            item.setToolTip(0, IMPORTED_MODEL_TOOLTIP.format(full_path))
        else:
            item.setData(0, Qt.ItemDataRole.UserRole, model)
        return item

    def validate_quantization_inputs(self) -> None:
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

    def quantize_model(self) -> None:
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

    def parse_progress(self, line, task_item) -> None:
        # Parses the output line for progress information and updates the task item.
        match = re.search(r"\[\s*(\d+)\s*/\s*(\d+)\s*\].*", line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            progress = int((current / total) * 100)
            task_item.update_progress(progress)

    def task_finished(self, thread, task_item) -> None:
        self.logger.info(TASK_FINISHED.format(thread.log_file))
        if thread in self.quant_threads:
            self.quant_threads.remove(thread)
        task_item.update_status(COMPLETED)

    def show_task_details(self, item) -> None:
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

    def import_model(self) -> None:
        self.logger.info(IMPORTING_MODEL)
        file_path, _ = QFileDialog.getOpenFileName(
            self, SELECT_MODEL_TO_IMPORT, "", GGUF_FILES
        )
        if file_path:
            file_name = os.path.basename(file_path)

            # Verify GGUF file
            if not self.verify_gguf(file_path):
                show_error(self.logger, INVALID_GGUF_FILE.format(file_name))
                return

            reply = QMessageBox.question(
                self,
                CONFIRM_IMPORT,
                IMPORT_MODEL_CONFIRMATION.format(file_name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if not hasattr(self, "imported_models"):
                    self.imported_models = []
                self.imported_models.append(file_path)
                self.load_models()
                self.logger.info(MODEL_IMPORTED_SUCCESSFULLY.format(file_name))

    def browse_imatrix_datafile(self) -> None:
        self.logger.info(BROWSING_FOR_IMATRIX_DATA_FILE)
        datafile, _ = QFileDialog.getOpenFileName(self, SELECT_DATA_FILE, "", ALL_FILES)
        if datafile:
            self.imatrix_datafile.setText(os.path.abspath(datafile))

    def browse_imatrix_model(self) -> None:
        self.logger.info(BROWSING_FOR_IMATRIX_MODEL_FILE)
        model_file, _ = QFileDialog.getOpenFileName(
            self, SELECT_MODEL_FILE, "", GGUF_FILES
        )
        if model_file:
            self.imatrix_model.setText(os.path.abspath(model_file))

    def browse_imatrix_output(self) -> None:
        self.logger.info(BROWSING_FOR_IMATRIX_OUTPUT_FILE)
        output_file, _ = QFileDialog.getSaveFileName(
            self, SELECT_OUTPUT_FILE, "", DAT_FILES
        )
        if output_file:
            self.imatrix_output.setText(os.path.abspath(output_file))

    def generate_imatrix(self) -> None:
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

    def closeEvent(self, event: QCloseEvent) -> None:
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
