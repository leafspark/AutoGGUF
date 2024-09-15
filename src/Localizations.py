import os
import re

AUTOGGUF_VERSION = "v1.9.0"


class _Localization:
    def __init__(self):
        pass


class _English(_Localization):
    def __init__(self):
        super().__init__()

        # General UI
        self.WINDOW_TITLE = "AutoGGUF (automated GGUF model quantizer)"
        self.RAM_USAGE = "RAM Usage:"
        self.CPU_USAGE = "CPU Usage:"
        self.BACKEND = "Llama.cpp Backend:"
        self.REFRESH_BACKENDS = "Refresh Backends"
        self.MODELS_PATH = "Models Path:"
        self.OUTPUT_PATH = "Output Path:"
        self.LOGS_PATH = "Logs Path:"
        self.BROWSE = "Browse"
        self.AVAILABLE_MODELS = "Available Models:"
        self.REFRESH_MODELS = "Refresh Models"
        self.STARTUP_ELASPED_TIME = "Initialization took {0} ms"

        # Usage Graphs
        self.CPU_USAGE_OVER_TIME = "CPU Usage Over Time"
        self.RAM_USAGE_OVER_TIME = "RAM Usage Over Time"

        # Environment variables
        self.DOTENV_FILE_NOT_FOUND = ".env file not found."
        self.COULD_NOT_PARSE_LINE = "Could not parse line: {0}"
        self.ERROR_LOADING_DOTENV = "Error loading .env: {0}"

        # Model Import
        self.IMPORT_MODEL = "Import Model"
        self.SELECT_MODEL_TO_IMPORT = "Select Model to Import"
        self.CONFIRM_IMPORT = "Confirm Import"
        self.IMPORT_MODEL_CONFIRMATION = "Do you want to import the model {}?"
        self.MODEL_IMPORTED_SUCCESSFULLY = "Model {} imported successfully"
        self.IMPORTING_MODEL = "Importing model"
        self.IMPORTED_MODEL_TOOLTIP = "Imported model: {}"

        # AutoFP8 Quantization
        self.AUTOFP8_QUANTIZATION_TASK_STARTED = "AutoFP8 quantization task started"
        self.ERROR_STARTING_AUTOFP8_QUANTIZATION = "Error starting AutoFP8 quantization"
        self.QUANTIZING_WITH_AUTOFP8 = "Quantizing {0} with AutoFP8"
        self.QUANTIZING_TO_WITH_AUTOFP8 = "Quantizing {0} to {1}"
        self.QUANTIZE_TO_FP8_DYNAMIC = "Quantize to FP8 Dynamic"
        self.OPEN_MODEL_FOLDER = "Open Model Folder"
        self.QUANTIZE = "Quantize"
        self.OPEN_MODEL_FOLDER = "Open Model Folder"
        self.INPUT_MODEL = "Input Model:"

        # GGUF Verification
        self.INVALID_GGUF_FILE = "Invalid GGUF file: {}"
        self.SHARDED_MODEL_NAME = "{} (Sharded)"
        self.IMPORTED_MODEL_TOOLTIP = "Imported model: {}"
        self.CONCATENATED_FILE_WARNING = "This is a concatenated file part. It will not work with llama-quantize; please concat the file first."
        self.CONCATENATED_FILES_FOUND = (
            "Found {} concatenated file parts. Please concat the files first."
        )

        # Plugins
        self.PLUGINS_DIR_NOT_EXIST = (
            "Plugins directory '{}' does not exist. No plugins will be loaded."
        )
        self.PLUGINS_DIR_NOT_DIRECTORY = (
            "'{}' exists but is not a directory. No plugins will be loaded."
        )
        self.PLUGIN_LOADED = "Loaded plugin: {} {}"
        self.PLUGIN_INCOMPATIBLE = "Plugin {} {} is not compatible with AutoGGUF version {}. Supported versions: {}"
        self.PLUGIN_LOAD_FAILED = "Failed to load plugin {}: {}"
        self.NO_PLUGINS_LOADED = "No plugins loaded."

        # GPU Monitoring
        self.GPU_USAGE = "GPU Usage:"
        self.GPU_USAGE_FORMAT = "GPU: {:.1f}% | VRAM: {:.1f}% ({} MB / {} MB)"
        self.GPU_DETAILS = "GPU Details"
        self.GPU_USAGE_OVER_TIME = "GPU Usage Over Time"
        self.VRAM_USAGE_OVER_TIME = "VRAM Usage Over Time"
        self.PERCENTAGE = "Percentage"
        self.TIME = "Time (s)"
        self.NO_GPU_DETECTED = "No GPU detected"
        self.SELECT_GPU = "Select GPU"
        self.AMD_GPU_NOT_SUPPORTED = "AMD GPU detected, but not supported"

        # Quantization
        self.QUANTIZATION_TYPE = "Quantization Type:"
        self.ALLOW_REQUANTIZE = "Allow Requantize"
        self.LEAVE_OUTPUT_TENSOR = "Leave Output Tensor"
        self.PURE = "Pure"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Include Weights:"
        self.EXCLUDE_WEIGHTS = "Exclude Weights:"
        self.USE_OUTPUT_TENSOR_TYPE = "Use Output Tensor Type"
        self.USE_TOKEN_EMBEDDING_TYPE = "Use Token Embedding Type"
        self.KEEP_SPLIT = "Keep Split"
        self.KV_OVERRIDES = "KV Overrides:"
        self.ADD_NEW_OVERRIDE = "Add new override"
        self.QUANTIZE_MODEL = "Quantize Model"
        self.EXTRA_ARGUMENTS = "Extra Arguments:"
        self.EXTRA_ARGUMENTS_LABEL = "Additional command-line arguments"
        self.QUANTIZATION_COMMAND = "Quantization command"

        # Presets
        self.SAVE_PRESET = "Save Preset"
        self.LOAD_PRESET = "Load Preset"

        # Tasks
        self.TASKS = "Tasks:"

        # llama.cpp Download
        self.DOWNLOAD_LLAMACPP = "Download llama.cpp"
        self.SELECT_RELEASE = "Select Release:"
        self.SELECT_ASSET = "Select Asset:"
        self.EXTRACT_CUDA_FILES = "Extract CUDA files"
        self.SELECT_CUDA_BACKEND = "Select CUDA Backend:"
        self.DOWNLOAD = "Download"
        self.REFRESH_RELEASES = "Refresh Releases"

        # IMatrix Generation
        self.IMATRIX_GENERATION = "IMatrix Generation"
        self.DATA_FILE = "Data File:"
        self.MODEL = "Model:"
        self.OUTPUT = "Output:"
        self.OUTPUT_FREQUENCY = "Output Frequency:"
        self.GPU_OFFLOAD = "GPU Offload:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Generate IMatrix"
        self.CONTEXT_SIZE = "Context Size:"
        self.CONTEXT_SIZE_FOR_IMATRIX = "Context size for IMatrix generation"
        self.THREADS = "Threads:"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = "Number of threads for IMatrix generation"
        self.IMATRIX_GENERATION_COMMAND = "IMatrix generation command"

        # LoRA Conversion
        self.LORA_CONVERSION = "LoRA Conversion"
        self.LORA_INPUT_PATH = "LoRA Input Path"
        self.LORA_OUTPUT_PATH = "LoRA Output Path"
        self.SELECT_LORA_INPUT_DIRECTORY = "Select LoRA Input Directory"
        self.SELECT_LORA_OUTPUT_FILE = "Select LoRA Output File"
        self.CONVERT_LORA = "Convert LoRA"
        self.LORA_CONVERSION_COMMAND = "LoRA conversion command"

        # LoRA Export
        self.EXPORT_LORA = "Export LoRA"
        self.GGML_LORA_ADAPTERS = "GGML LoRA Adapters"
        self.SELECT_LORA_ADAPTER_FILES = "Select LoRA Adapter Files"
        self.ADD_ADAPTER = "Add Adapter"
        self.DELETE_ADAPTER = "Delete"
        self.LORA_SCALE = "LoRA Scale"
        self.ENTER_LORA_SCALE_VALUE = "Enter LoRA Scale Value (Optional)"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = "Number of Threads for LoRA Export"
        self.LORA_EXPORT_COMMAND = "LoRA export command"

        # HuggingFace to GGUF Conversion
        self.HF_TO_GGUF_CONVERSION = "HuggingFace to GGUF Conversion"
        self.MODEL_DIRECTORY = "Model Directory:"
        self.OUTPUT_FILE = "Output File:"
        self.OUTPUT_TYPE = "Output Type:"
        self.VOCAB_ONLY = "Vocab Only"
        self.USE_TEMP_FILE = "Use Temp File"
        self.NO_LAZY_EVALUATION = "No Lazy Evaluation"
        self.MODEL_NAME = "Model Name:"
        self.VERBOSE = "Verbose"
        self.SPLIT_MAX_SIZE = "Split Max Size:"
        self.DRY_RUN = "Dry Run"
        self.CONVERT_HF_TO_GGUF = "Convert HF to GGUF"
        self.SELECT_HF_MODEL_DIRECTORY = "Select HuggingFace Model Directory"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = "Browsing for HuggingFace model directory"
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = (
            "Browsing for HuggingFace to GGUF output file"
        )

        # Update Checking
        self.UPDATE_AVAILABLE = "Update Avaliable"
        self.NEW_VERSION_AVAILABLE = "A new version is avaliable: {}"
        self.DOWNLOAD_NEW_VERSION = "Download?"
        self.ERROR_CHECKING_FOR_UPDATES = "Error checking for updates:"
        self.CHECKING_FOR_UPDATES = "Checking for updates"

        # General Messages
        self.ERROR = "Error"
        self.WARNING = "Warning"
        self.PROPERTIES = "Properties"
        self.CANCEL = "Cancel"
        self.RESTART = "Restart"
        self.DELETE = "Delete"
        self.RENAME = "Rename"
        self.CONFIRM_DELETION = "Are you sure you want to delete this task?"
        self.TASK_RUNNING_WARNING = (
            "Some tasks are still running. Are you sure you want to quit?"
        )
        self.YES = "Yes"
        self.NO = "No"
        self.COMPLETED = "Completed"

        # File Types
        self.ALL_FILES = "All Files (*)"
        self.GGUF_FILES = "GGUF Files (*.gguf)"
        self.DAT_FILES = "DAT Files (*.dat)"
        self.JSON_FILES = "JSON Files (*.json)"
        self.BIN_FILES = "Binary Files (*.bin)"
        self.LORA_FILES = "LoRA Files (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "GGUF and Binary Files (*.gguf *.bin)"
        self.SHARDED = "sharded"

        # Status Messages
        self.DOWNLOAD_COMPLETE = "Download Complete"
        self.CUDA_EXTRACTION_FAILED = "CUDA Extraction Failed"
        self.PRESET_SAVED = "Preset Saved"
        self.PRESET_LOADED = "Preset Loaded"
        self.NO_ASSET_SELECTED = "No asset selected"
        self.DOWNLOAD_FAILED = "Download failed"
        self.NO_BACKEND_SELECTED = "No backend selected"
        self.NO_MODEL_SELECTED = "No model selected"
        self.NO_SUITABLE_CUDA_BACKENDS = "No suitable CUDA backends found"
        self.IN_PROGRESS = "In Progress"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "llama.cpp binary downloaded and extracted to {0}"
        )
        self.CUDA_FILES_EXTRACTED = "CUDA files extracted to"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "No suitable CUDA backend found for extraction"
        )
        self.ERROR_FETCHING_RELEASES = "Error fetching releases: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirm Deletion"
        self.LOG_FOR = "Log for {0}"
        self.FAILED_TO_LOAD_PRESET = "Failed to load preset: {0}"
        self.INITIALIZING_AUTOGGUF = "Initializing AutoGGUF application"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF initialization complete"
        self.REFRESHING_BACKENDS = "Refreshing backends"
        self.NO_BACKENDS_AVAILABLE = "No backends available"
        self.FOUND_VALID_BACKENDS = "Found {0} valid backends"
        self.SAVING_PRESET = "Saving preset"
        self.PRESET_SAVED_TO = "Preset saved to {0}"
        self.LOADING_PRESET = "Loading preset"
        self.PRESET_LOADED_FROM = "Preset loaded from {0}"
        self.ADDING_KV_OVERRIDE = "Adding KV override: {0}"
        self.SAVING_TASK_PRESET = "Saving task preset for {0}"
        self.TASK_PRESET_SAVED = "Task Preset Saved"
        self.TASK_PRESET_SAVED_TO = "Task preset saved to {0}"
        self.RESTARTING_TASK = "Restarting task: {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download finished. Extracted to: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binary downloaded and extracted to {0}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "No suitable CUDA backend found for extraction"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binary downloaded and extracted to {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Refreshing llama.cpp releases"
        self.UPDATING_ASSET_LIST = "Updating asset list"
        self.UPDATING_CUDA_OPTIONS = "Updating CUDA options"
        self.STARTING_LLAMACPP_DOWNLOAD = "Starting llama.cpp download"
        self.UPDATING_CUDA_BACKENDS = "Updating CUDA backends"
        self.NO_CUDA_BACKEND_SELECTED = "No CUDA backend selected for extraction"
        self.EXTRACTING_CUDA_FILES = "Extracting CUDA files from {0} to {1}"
        self.DOWNLOAD_ERROR = "Download error: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Showing task context menu"
        self.SHOWING_PROPERTIES_FOR_TASK = "Showing properties for task: {0}"
        self.CANCELLING_TASK = "Cancelling task: {0}"
        self.CANCELED = "Canceled"
        self.DELETING_TASK = "Deleting task: {0}"
        self.LOADING_MODELS = "Loading models"
        self.LOADED_MODELS = "Loaded {0} models"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Browsing for models directory"
        self.SELECT_MODELS_DIRECTORY = "Select Models Directory"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Browsing for output directory"
        self.SELECT_OUTPUT_DIRECTORY = "Select Output Directory"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Browsing for logs directory"
        self.SELECT_LOGS_DIRECTORY = "Select Logs Directory"
        self.BROWSING_FOR_IMATRIX_FILE = "Browsing for IMatrix file"
        self.SELECT_IMATRIX_FILE = "Select IMatrix File"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU Usage: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validating quantization inputs"
        self.MODELS_PATH_REQUIRED = "Models path is required"
        self.OUTPUT_PATH_REQUIRED = "Output path is required"
        self.LOGS_PATH_REQUIRED = "Logs path is required"
        self.STARTING_MODEL_QUANTIZATION = "Starting model quantization"
        self.INPUT_FILE_NOT_EXIST = "Input file '{0}' does not exist."
        self.QUANTIZING_MODEL_TO = "Quantizing {0} to {1}"
        self.QUANTIZATION_TASK_STARTED = "Quantization task started for {0}"
        self.ERROR_STARTING_QUANTIZATION = "Error starting quantization: {0}"
        self.UPDATING_MODEL_INFO = "Updating model info: {0}"
        self.TASK_FINISHED = "Task finished: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Showing task details for: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Browsing for IMatrix data file"
        self.SELECT_DATA_FILE = "Select Data File"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Browsing for IMatrix model file"
        self.SELECT_MODEL_FILE = "Select Model File"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Browsing for IMatrix output file"
        self.SELECT_OUTPUT_FILE = "Select Output File"
        self.STARTING_IMATRIX_GENERATION = "Starting IMatrix generation"
        self.BACKEND_PATH_NOT_EXIST = "Backend path does not exist: {0}"
        self.GENERATING_IMATRIX = "Generating IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Error starting IMatrix generation: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix generation task started"
        self.ERROR_MESSAGE = "Error: {0}"
        self.TASK_ERROR = "Task error: {0}"
        self.APPLICATION_CLOSING = "Application closing"
        self.APPLICATION_CLOSED = "Application closed"
        self.SELECT_QUANTIZATION_TYPE = "Select the quantization type"
        self.ALLOWS_REQUANTIZING = (
            "Allows requantizing tensors that have already been quantized"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Will leave output.weight un(re)quantized"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Disable k-quant mixtures and quantize all tensors to the same type"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "Use data in file as importance matrix for quant optimizations"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Use importance matrix for these tensors"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Don't use importance matrix for these tensors"
        )
        self.OUTPUT_TENSOR_TYPE = "Output Tensor Type:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Use this type for the output.weight tensor"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token Embedding Type:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Use this type for the token embeddings tensor"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Will generate quantized model in the same shards as input"
        )
        self.OVERRIDE_MODEL_METADATA = "Override model metadata"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Input data file for IMatrix generation"
        self.MODEL_TO_BE_QUANTIZED = "Model to be quantized"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Output path for the generated IMatrix"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "How often to save the IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Set GPU offload value (-ngl)"
        self.STARTING_LORA_CONVERSION = "Starting LoRA Conversion"
        self.LORA_INPUT_PATH_REQUIRED = "LoRA input path is required."
        self.LORA_OUTPUT_PATH_REQUIRED = "LoRA output path is required."
        self.ERROR_STARTING_LORA_CONVERSION = "Error starting LoRA conversion: {}"
        self.LORA_CONVERSION_TASK_STARTED = "LoRA conversion task started."
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = "Browsing for LoRA input directory..."
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "Browsing for LoRA output file..."
        self.CONVERTING_LORA = "LoRA Conversion"
        self.LORA_CONVERSION_FINISHED = "LoRA conversion finished."
        self.LORA_FILE_MOVED = "LoRA file moved from {} to {}."
        self.LORA_FILE_NOT_FOUND = "LoRA file not found: {}."
        self.ERROR_MOVING_LORA_FILE = "Error moving LoRA file: {}"
        self.MODEL_PATH_REQUIRED = "Model path is required."
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = (
            "At least one LoRA adapter is required."
        )
        self.INVALID_LORA_SCALE_VALUE = "Invalid LoRA scale value."
        self.ERROR_STARTING_LORA_EXPORT = "Error starting LoRA export: {}"
        self.LORA_EXPORT_TASK_STARTED = "LoRA export task started."
        self.EXPORTING_LORA = "Exporting LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = (
            "Browsing for Export LoRA Model File..."
        )
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = (
            "Browsing for Export LoRA Output File..."
        )
        self.ADDING_LORA_ADAPTER = "Adding LoRA Adapter..."
        self.DELETING_LORA_ADAPTER = "Deleting LoRA Adapter..."
        self.SELECT_LORA_ADAPTER_FILE = "Select LoRA Adapter File"
        self.STARTING_LORA_EXPORT = "Starting LoRA export..."
        self.SELECT_OUTPUT_TYPE = "Select Output Type (GGUF or GGML)"
        self.BASE_MODEL = "Base Model"
        self.SELECT_BASE_MODEL_FILE = "Select Base Model File (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = "Base model path is required for GGUF output."
        self.BROWSING_FOR_BASE_MODEL_FILE = "Browsing for base model file..."
        self.SELECT_BASE_MODEL_FOLDER = (
            "Select Base Model Folder (containing safetensors)"
        )
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "Browsing for base model folder..."
        self.LORA_CONVERSION_FROM_TO = "LoRA Conversion from {} to {}"
        self.GENERATING_IMATRIX_FOR = "Generating IMatrix for {}"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = (
            "Model path is required for IMatrix generation."
        )
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = "No asset selected for CUDA check"
        self.NO_QUANTIZATION_TYPE_SELECTED = "No quantization type selected. Please select at least one quantization type."
        self.STARTING_HF_TO_GGUF_CONVERSION = "Starting HuggingFace to GGUF conversion"
        self.MODEL_DIRECTORY_REQUIRED = "Model directory is required"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "HF to GGUF Conversion Command: {}"
        self.CONVERTING_TO_GGUF = "Converting {} to GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "Error starting HuggingFace to GGUF conversion: {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = (
            "HuggingFace to GGUF conversion task started"
        )

        # Split GGUF
        self.SPLIT_GGUF = "Split GGUF"
        self.SPLIT_MAX_SIZE = "Split Max Size"
        self.SPLIT_MAX_TENSORS = "Split Max Tensors"
        self.SPLIT_GGUF_TASK_STARTED = "GGUF Split task started"
        self.SPLIT_GGUF_TASK_FINISHED = "GGUF Split task finished"
        self.SPLIT_GGUF_COMMAND = "GGUF Split Command"
        self.SPLIT_GGUF_ERROR = "Error starting GGUF split"
        self.NUMBER_OF_TENSORS = "Number of tensors"
        self.SIZE_IN_UNITS = "Size in G/M"

        # Model actions
        self.CONFIRM_DELETE = "Confirm Delete"
        self.DELETE_MODEL_WARNING = "Are you sure you want to delete the model: {}?"
        self.MODEL_RENAMED_SUCCESSFULLY = "Model renamed successfully."
        self.MODEL_DELETED_SUCCESSFULLY = "Model deleted successfully."

        # HuggingFace Transfer
        self.ALL_FIELDS_REQUIRED = "All fields are required."
        self.HUGGINGFACE_UPLOAD_COMMAND = "HuggingFace Upload Command: "
        self.UPLOADING = "Uploading"
        self.UPLOADING_FOLDER = "Uploading folder"
        self.HF_TRANSFER_TASK_NAME = "{} {} to {} from {}"
        self.ERROR_STARTING_HF_TRANSFER = "Error starting HF transfer: {}"
        self.STARTED_HUGGINGFACE_TRANSFER = "Started HuggingFace {} operation."
        self.SELECT_FOLDER = "Select Folder"
        self.SELECT_FILE = "Select File"


class _French(_Localization):
    def __init__(self):
        super().__init__()

        # Interface utilisateur générale
        self.WINDOW_TITLE = "AutoGGUF (quantificateur de modèle GGUF automatisé)"
        self.RAM_USAGE = "Utilisation de la RAM :"
        self.CPU_USAGE = "Utilisation du processeur :"
        self.BACKEND = "Backend Llama.cpp :"
        self.REFRESH_BACKENDS = "Actualiser les backends"
        self.MODELS_PATH = "Chemin des modèles :"
        self.OUTPUT_PATH = "Chemin de sortie :"
        self.LOGS_PATH = "Chemin des journaux :"
        self.BROWSE = "Parcourir"
        self.AVAILABLE_MODELS = "Modèles disponibles :"
        self.REFRESH_MODELS = "Actualiser les modèles"
        self.STARTUP_ELASPED_TIME = "L'initialisation a pris {0} ms"

        # Variables d'environnement
        self.DOTENV_FILE_NOT_FOUND = "Fichier .env introuvable."
        self.COULD_NOT_PARSE_LINE = "Impossible d'analyser la ligne : {0}"
        self.ERROR_LOADING_DOTENV = "Erreur lors du chargement de .env : {0}"

        # Importation de modèle
        self.IMPORT_MODEL = "Importer un modèle"
        self.SELECT_MODEL_TO_IMPORT = "Sélectionner le modèle à importer"
        self.CONFIRM_IMPORT = "Confirmer l'importation"
        self.IMPORT_MODEL_CONFIRMATION = "Voulez-vous importer le modèle {} ?"
        self.MODEL_IMPORTED_SUCCESSFULLY = "Modèle {} importé avec succès"
        self.IMPORTING_MODEL = "Importation du modèle"
        self.IMPORTED_MODEL_TOOLTIP = "Modèle importé : {}"

        # Quantification AutoFP8
        self.AUTOFP8_QUANTIZATION_TASK_STARTED = (
            "La tâche de quantification AutoFP8 a démarré"
        )
        self.ERROR_STARTING_AUTOFP8_QUANTIZATION = (
            "Erreur lors du démarrage de la quantification AutoFP8"
        )
        self.QUANTIZING_WITH_AUTOFP8 = "Quantification de {0} avec AutoFP8"
        self.QUANTIZING_TO_WITH_AUTOFP8 = "Quantification de {0} en {1}"
        self.QUANTIZE_TO_FP8_DYNAMIC = "Quantifier en FP8 dynamique"
        self.OPEN_MODEL_FOLDER = "Ouvrir le dossier du modèle"
        self.QUANTIZE = "Quantifier"
        self.OPEN_MODEL_FOLDER = "Ouvrir le dossier du modèle"
        self.INPUT_MODEL = "Modèle d'entrée :"

        # Vérification GGUF
        self.INVALID_GGUF_FILE = "Fichier GGUF invalide : {}"
        self.SHARDED_MODEL_NAME = "{} (Fragmenté)"
        self.IMPORTED_MODEL_TOOLTIP = "Modèle importé : {}"
        self.CONCATENATED_FILE_WARNING = "Il s'agit d'une partie de fichier concaténée. Cela ne fonctionnera pas avec llama-quantize ; veuillez d'abord concaténer le fichier."
        self.CONCATENATED_FILES_FOUND = "{} parties de fichiers concaténées trouvées. Veuillez d'abord concaténer les fichiers."

        # Plugins
        self.PLUGINS_DIR_NOT_EXIST = (
            "Le répertoire des plugins '{}' n'existe pas. Aucun plugin ne sera chargé."
        )
        self.PLUGINS_DIR_NOT_DIRECTORY = (
            "'{}' existe mais n'est pas un répertoire. Aucun plugin ne sera chargé."
        )
        self.PLUGIN_LOADED = "Plugin chargé : {} {}"
        self.PLUGIN_INCOMPATIBLE = "Le plugin {} {} n'est pas compatible avec la version {} d'AutoGGUF. Versions prises en charge : {}"
        self.PLUGIN_LOAD_FAILED = "Échec du chargement du plugin {} : {}"
        self.NO_PLUGINS_LOADED = "Aucun plugin chargé."

        # Surveillance du GPU
        self.GPU_USAGE = "Utilisation du GPU :"
        self.GPU_USAGE_FORMAT = "GPU: {:.1f}% | VRAM: {:.1f}% ({} Mo / {} Mo)"
        self.GPU_DETAILS = "Détails du GPU"
        self.GPU_USAGE_OVER_TIME = "Utilisation du GPU au fil du temps"
        self.VRAM_USAGE_OVER_TIME = "Utilisation de la VRAM au fil du temps"
        self.PERCENTAGE = "Pourcentage"
        self.TIME = "Temps (s)"
        self.NO_GPU_DETECTED = "Aucun GPU détecté"
        self.SELECT_GPU = "Sélectionner le GPU"
        self.AMD_GPU_NOT_SUPPORTED = "GPU AMD détecté, mais non pris en charge"

        # Quantification
        self.QUANTIZATION_TYPE = "Type de quantification :"
        self.ALLOW_REQUANTIZE = "Autoriser la requantification"
        self.LEAVE_OUTPUT_TENSOR = "Laisser le tenseur de sortie"
        self.PURE = "Pur"
        self.IMATRIX = "IMatrix :"
        self.INCLUDE_WEIGHTS = "Inclure les poids :"
        self.EXCLUDE_WEIGHTS = "Exclure les poids :"
        self.USE_OUTPUT_TENSOR_TYPE = "Utiliser le type de tenseur de sortie"
        self.USE_TOKEN_EMBEDDING_TYPE = "Utiliser le type d'intégration de jetons"
        self.KEEP_SPLIT = "Conserver la division"
        self.KV_OVERRIDES = "Remplacements KV :"
        self.ADD_NEW_OVERRIDE = "Ajouter un nouveau remplacement"
        self.QUANTIZE_MODEL = "Quantifier le modèle"
        self.EXTRA_ARGUMENTS = "Arguments supplémentaires :"
        self.EXTRA_ARGUMENTS_LABEL = "Arguments de ligne de commande supplémentaires"
        self.QUANTIZATION_COMMAND = "Commande de quantification"

        # Préréglages
        self.SAVE_PRESET = "Enregistrer le préréglage"
        self.LOAD_PRESET = "Charger le préréglage"

        # Tâches
        self.TASKS = "Tâches :"

        # Téléchargement de llama.cpp
        self.DOWNLOAD_LLAMACPP = "Télécharger llama.cpp"
        self.SELECT_RELEASE = "Sélectionner la version :"
        self.SELECT_ASSET = "Sélectionner l'actif :"
        self.EXTRACT_CUDA_FILES = "Extraire les fichiers CUDA"
        self.SELECT_CUDA_BACKEND = "Sélectionner le backend CUDA :"
        self.DOWNLOAD = "Télécharger"
        self.REFRESH_RELEASES = "Actualiser les versions"

        # Génération d'IMatrix
        self.IMATRIX_GENERATION = "Génération d'IMatrix"
        self.DATA_FILE = "Fichier de données :"
        self.MODEL = "Modèle :"
        self.OUTPUT = "Sortie :"
        self.OUTPUT_FREQUENCY = "Fréquence de sortie :"
        self.GPU_OFFLOAD = "Déchargement GPU :"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Générer IMatrix"
        self.CONTEXT_SIZE = "Taille du contexte :"
        self.CONTEXT_SIZE_FOR_IMATRIX = (
            "Taille du contexte pour la génération d'IMatrix"
        )
        self.THREADS = "Threads :"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = (
            "Nombre de threads pour la génération d'IMatrix"
        )
        self.IMATRIX_GENERATION_COMMAND = "Commande de génération d'IMatrix"

        # Conversion LoRA
        self.LORA_CONVERSION = "Conversion LoRA"
        self.LORA_INPUT_PATH = "Chemin d'entrée LoRA"
        self.LORA_OUTPUT_PATH = "Chemin de sortie LoRA"
        self.SELECT_LORA_INPUT_DIRECTORY = "Sélectionner le répertoire d'entrée LoRA"
        self.SELECT_LORA_OUTPUT_FILE = "Sélectionner le fichier de sortie LoRA"
        self.CONVERT_LORA = "Convertir LoRA"
        self.LORA_CONVERSION_COMMAND = "Commande de conversion LoRA"

        # Exportation LoRA
        self.EXPORT_LORA = "Exporter LoRA"
        self.GGML_LORA_ADAPTERS = "Adaptateurs GGML LoRA"
        self.SELECT_LORA_ADAPTER_FILES = "Sélectionner les fichiers d'adaptateur LoRA"
        self.ADD_ADAPTER = "Ajouter un adaptateur"
        self.DELETE_ADAPTER = "Supprimer"
        self.LORA_SCALE = "Échelle LoRA"
        self.ENTER_LORA_SCALE_VALUE = "Saisir la valeur de l'échelle LoRA (facultatif)"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = (
            "Nombre de threads pour l'exportation LoRA"
        )
        self.LORA_EXPORT_COMMAND = "Commande d'exportation LoRA"

        # Conversion HuggingFace vers GGUF
        self.HF_TO_GGUF_CONVERSION = "Conversion HuggingFace vers GGUF"
        self.MODEL_DIRECTORY = "Répertoire du modèle :"
        self.OUTPUT_FILE = "Fichier de sortie :"
        self.OUTPUT_TYPE = "Type de sortie :"
        self.VOCAB_ONLY = "Vocab uniquement"
        self.USE_TEMP_FILE = "Utiliser un fichier temporaire"
        self.NO_LAZY_EVALUATION = "Pas d'évaluation paresseuse"
        self.MODEL_NAME = "Nom du modèle :"
        self.VERBOSE = "Verbeux"
        self.SPLIT_MAX_SIZE = "Taille maximale de la division :"
        self.DRY_RUN = "Exécution à sec"
        self.CONVERT_HF_TO_GGUF = "Convertir HF en GGUF"
        self.SELECT_HF_MODEL_DIRECTORY = (
            "Sélectionner le répertoire du modèle HuggingFace"
        )
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = (
            "Recherche du répertoire du modèle HuggingFace"
        )
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = (
            "Recherche du fichier de sortie HuggingFace vers GGUF"
        )

        # Vérification des mises à jour
        self.UPDATE_AVAILABLE = "Mise à jour disponible"
        self.NEW_VERSION_AVAILABLE = "Une nouvelle version est disponible : {}"
        self.DOWNLOAD_NEW_VERSION = "Télécharger ?"
        self.ERROR_CHECKING_FOR_UPDATES = (
            "Erreur lors de la vérification des mises à jour :"
        )
        self.CHECKING_FOR_UPDATES = "Vérification des mises à jour"

        # Messages généraux
        self.ERROR = "Erreur"
        self.WARNING = "Avertissement"
        self.PROPERTIES = "Propriétés"
        self.CANCEL = "Annuler"
        self.RESTART = "Redémarrer"
        self.DELETE = "Supprimer"
        self.CONFIRM_DELETION = "Êtes-vous sûr de vouloir supprimer cette tâche ?"
        self.TASK_RUNNING_WARNING = "Certaines tâches sont encore en cours d'exécution. Êtes-vous sûr de vouloir quitter ?"
        self.YES = "Oui"
        self.NO = "Non"
        self.COMPLETED = "Terminé"

        # Types de fichiers
        self.ALL_FILES = "Tous les fichiers (*)"
        self.GGUF_FILES = "Fichiers GGUF (*.gguf)"
        self.DAT_FILES = "Fichiers DAT (*.dat)"
        self.JSON_FILES = "Fichiers JSON (*.json)"
        self.BIN_FILES = "Fichiers binaires (*.bin)"
        self.LORA_FILES = "Fichiers LoRA (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "Fichiers GGUF et binaires (*.gguf *.bin)"
        self.SHARDED = "fragmenté"

        # Messages d'état
        self.DOWNLOAD_COMPLETE = "Téléchargement terminé"
        self.CUDA_EXTRACTION_FAILED = "Échec de l'extraction CUDA"
        self.PRESET_SAVED = "Préréglage enregistré"
        self.PRESET_LOADED = "Préréglage chargé"
        self.NO_ASSET_SELECTED = "Aucun actif sélectionné"
        self.DOWNLOAD_FAILED = "Échec du téléchargement"
        self.NO_BACKEND_SELECTED = "Aucun backend sélectionné"
        self.NO_MODEL_SELECTED = "Aucun modèle sélectionné"
        self.NO_SUITABLE_CUDA_BACKENDS = "Aucun backend CUDA approprié trouvé"
        self.IN_PROGRESS = "En cours"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "Binaire llama.cpp téléchargé et extrait vers {0}"
        )
        self.CUDA_FILES_EXTRACTED = "Fichiers CUDA extraits vers"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Aucun backend CUDA approprié trouvé pour l'extraction"
        )
        self.ERROR_FETCHING_RELEASES = (
            "Erreur lors de la récupération des versions : {0}"
        )
        self.CONFIRM_DELETION_TITLE = "Confirmer la suppression"
        self.LOG_FOR = "Journal pour {0}"
        self.FAILED_TO_LOAD_PRESET = "Échec du chargement du préréglage : {0}"
        self.INITIALIZING_AUTOGGUF = "Initialisation de l'application AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Initialisation d'AutoGGUF terminée"
        self.REFRESHING_BACKENDS = "Actualisation des backends"
        self.NO_BACKENDS_AVAILABLE = "Aucun backend disponible"
        self.FOUND_VALID_BACKENDS = "{0} backends valides trouvés"
        self.SAVING_PRESET = "Enregistrement du préréglage"
        self.PRESET_SAVED_TO = "Préréglage enregistré vers {0}"
        self.LOADING_PRESET = "Chargement du préréglage"
        self.PRESET_LOADED_FROM = "Préréglage chargé depuis {0}"
        self.ADDING_KV_OVERRIDE = "Ajout du remplacement KV : {0}"
        self.SAVING_TASK_PRESET = "Enregistrement du préréglage de tâche pour {0}"
        self.TASK_PRESET_SAVED = "Préréglage de tâche enregistré"
        self.TASK_PRESET_SAVED_TO = "Préréglage de tâche enregistré vers {0}"
        self.RESTARTING_TASK = "Redémarrage de la tâche : {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "Téléchargement terminé. Extrait vers : {0}"
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "Binaire llama.cpp téléchargé et extrait vers {0}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Aucun backend CUDA approprié trouvé pour l'extraction"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Binaire llama.cpp téléchargé et extrait vers {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Actualisation des versions de llama.cpp"
        self.UPDATING_ASSET_LIST = "Mise à jour de la liste des actifs"
        self.UPDATING_CUDA_OPTIONS = "Mise à jour des options CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Démarrage du téléchargement de llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Mise à jour des backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = (
            "Aucun backend CUDA sélectionné pour l'extraction"
        )
        self.EXTRACTING_CUDA_FILES = "Extraction des fichiers CUDA de {0} vers {1}"
        self.DOWNLOAD_ERROR = "Erreur de téléchargement : {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Affichage du menu contextuel de la tâche"
        self.SHOWING_PROPERTIES_FOR_TASK = "Affichage des propriétés de la tâche : {0}"
        self.CANCELLING_TASK = "Annulation de la tâche : {0}"
        self.CANCELED = "Annulé"
        self.DELETING_TASK = "Suppression de la tâche : {0}"
        self.LOADING_MODELS = "Chargement des modèles"
        self.LOADED_MODELS = "{0} modèles chargés"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Recherche du répertoire des modèles"
        self.SELECT_MODELS_DIRECTORY = "Sélectionner le répertoire des modèles"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Recherche du répertoire de sortie"
        self.SELECT_OUTPUT_DIRECTORY = "Sélectionner le répertoire de sortie"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Recherche du répertoire des journaux"
        self.SELECT_LOGS_DIRECTORY = "Sélectionner le répertoire des journaux"
        self.BROWSING_FOR_IMATRIX_FILE = "Recherche du fichier IMatrix"
        self.SELECT_IMATRIX_FILE = "Sélectionner le fichier IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} Mo / {2} Mo)"
        self.CPU_USAGE_FORMAT = "Utilisation du processeur : {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validation des entrées de quantification"
        self.MODELS_PATH_REQUIRED = "Le chemin des modèles est requis"
        self.OUTPUT_PATH_REQUIRED = "Le chemin de sortie est requis"
        self.LOGS_PATH_REQUIRED = "Le chemin des journaux est requis"
        self.STARTING_MODEL_QUANTIZATION = "Démarrage de la quantification du modèle"
        self.INPUT_FILE_NOT_EXIST = "Le fichier d'entrée '{0}' n'existe pas."
        self.QUANTIZING_MODEL_TO = "Quantification de {0} en {1}"
        self.QUANTIZATION_TASK_STARTED = "La tâche de quantification a démarré pour {0}"
        self.ERROR_STARTING_QUANTIZATION = (
            "Erreur lors du démarrage de la quantification : {0}"
        )
        self.UPDATING_MODEL_INFO = "Mise à jour des informations du modèle : {0}"
        self.TASK_FINISHED = "Tâche terminée : {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Affichage des détails de la tâche pour : {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Recherche du fichier de données IMatrix"
        self.SELECT_DATA_FILE = "Sélectionner le fichier de données"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Recherche du fichier de modèle IMatrix"
        self.SELECT_MODEL_FILE = "Sélectionner le fichier de modèle"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Recherche du fichier de sortie IMatrix"
        self.SELECT_OUTPUT_FILE = "Sélectionner le fichier de sortie"
        self.STARTING_IMATRIX_GENERATION = "Démarrage de la génération d'IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Le chemin du backend n'existe pas : {0}"
        self.GENERATING_IMATRIX = "Génération d'IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Erreur lors du démarrage de la génération d'IMatrix : {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = (
            "La tâche de génération d'IMatrix a démarré"
        )
        self.ERROR_MESSAGE = "Erreur : {0}"
        self.TASK_ERROR = "Erreur de tâche : {0}"
        self.APPLICATION_CLOSING = "Fermeture de l'application"
        self.APPLICATION_CLOSED = "Application fermée"
        self.SELECT_QUANTIZATION_TYPE = "Sélectionnez le type de quantification"
        self.ALLOWS_REQUANTIZING = (
            "Permet de requantifier les tenseurs qui ont déjà été quantifiés"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Laissera output.weight non (re)quantifié"
        self.DISABLE_K_QUANT_MIXTURES = "Désactiver les mélanges k-quant et quantifier tous les tenseurs au même type"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Utiliser les données du fichier comme matrice d'importance pour les optimisations de quantification"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Utiliser la matrice d'importance pour ces tenseurs"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Ne pas utiliser la matrice d'importance pour ces tenseurs"
        )
        self.OUTPUT_TENSOR_TYPE = "Type de tenseur de sortie :"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Utiliser ce type pour le tenseur output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Type d'intégration de jetons :"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Utiliser ce type pour le tenseur d'intégration de jetons"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Générera le modèle quantifié dans les mêmes fragments que l'entrée"
        )
        self.OVERRIDE_MODEL_METADATA = "Remplacer les métadonnées du modèle"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Fichier de données d'entrée pour la génération d'IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modèle à quantifier"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Chemin de sortie pour l'IMatrix généré"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Fréquence d'enregistrement de l'IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Définir la valeur de déchargement GPU (-ngl)"
        self.STARTING_LORA_CONVERSION = "Démarrage de la conversion LoRA"
        self.LORA_INPUT_PATH_REQUIRED = "Le chemin d'entrée LoRA est requis."
        self.LORA_OUTPUT_PATH_REQUIRED = "Le chemin de sortie LoRA est requis."
        self.ERROR_STARTING_LORA_CONVERSION = (
            "Erreur lors du démarrage de la conversion LoRA : {}"
        )
        self.LORA_CONVERSION_TASK_STARTED = "La tâche de conversion LoRA a démarré."
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = (
            "Recherche du répertoire d'entrée LoRA..."
        )
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "Recherche du fichier de sortie LoRA..."
        self.CONVERTING_LORA = "Conversion LoRA"
        self.LORA_CONVERSION_FINISHED = "Conversion LoRA terminée."
        self.LORA_FILE_MOVED = "Fichier LoRA déplacé de {} vers {}."
        self.LORA_FILE_NOT_FOUND = "Fichier LoRA introuvable : {}."
        self.ERROR_MOVING_LORA_FILE = "Erreur lors du déplacement du fichier LoRA : {}"
        self.MODEL_PATH_REQUIRED = "Le chemin du modèle est requis."
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = (
            "Au moins un adaptateur LoRA est requis."
        )
        self.INVALID_LORA_SCALE_VALUE = "Valeur d'échelle LoRA invalide."
        self.ERROR_STARTING_LORA_EXPORT = (
            "Erreur lors du démarrage de l'exportation LoRA : {}"
        )
        self.LORA_EXPORT_TASK_STARTED = "La tâche d'exportation LoRA a démarré."
        self.EXPORTING_LORA = "Exportation de LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = (
            "Recherche du fichier de modèle LoRA à exporter..."
        )
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = (
            "Recherche du fichier de sortie LoRA à exporter..."
        )
        self.ADDING_LORA_ADAPTER = "Ajout d'un adaptateur LoRA..."
        self.DELETING_LORA_ADAPTER = "Suppression d'un adaptateur LoRA..."
        self.SELECT_LORA_ADAPTER_FILE = "Sélectionner le fichier d'adaptateur LoRA"
        self.STARTING_LORA_EXPORT = "Démarrage de l'exportation LoRA..."
        self.SELECT_OUTPUT_TYPE = "Sélectionner le type de sortie (GGUF ou GGML)"
        self.BASE_MODEL = "Modèle de base"
        self.SELECT_BASE_MODEL_FILE = "Sélectionner le fichier de modèle de base (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = (
            "Le chemin du modèle de base est requis pour la sortie GGUF."
        )
        self.BROWSING_FOR_BASE_MODEL_FILE = "Recherche du fichier de modèle de base..."
        self.SELECT_BASE_MODEL_FOLDER = (
            "Sélectionner le dossier du modèle de base (contenant safetensors)"
        )
        self.BROWSING_FOR_BASE_MODEL_FOLDER = (
            "Recherche du dossier du modèle de base..."
        )
        self.LORA_CONVERSION_FROM_TO = "Conversion LoRA de {} vers {}"
        self.GENERATING_IMATRIX_FOR = "Génération d'IMatrix pour {}"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = (
            "Le chemin du modèle est requis pour la génération d'IMatrix."
        )
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = (
            "Aucun actif sélectionné pour la vérification CUDA"
        )
        self.NO_QUANTIZATION_TYPE_SELECTED = "Aucun type de quantification sélectionné. Veuillez sélectionner au moins un type de quantification."
        self.STARTING_HF_TO_GGUF_CONVERSION = (
            "Démarrage de la conversion HuggingFace vers GGUF"
        )
        self.MODEL_DIRECTORY_REQUIRED = "Le répertoire du modèle est requis"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "Commande de conversion HF vers GGUF : {}"
        self.CONVERTING_TO_GGUF = "Conversion de {} en GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "Erreur lors du démarrage de la conversion HuggingFace vers GGUF : {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = (
            "La tâche de conversion HuggingFace vers GGUF a démarré"
        )


class _SimplifiedChinese(_Localization):
    def __init__(self):
        super().__init__()

        # 通用界面
        self.WINDOW_TITLE = "AutoGGUF（自动GGUF模型量化器）"
        self.RAM_USAGE = "内存使用率："
        self.CPU_USAGE = "CPU使用率："
        self.BACKEND = "Llama.cpp后端："
        self.REFRESH_BACKENDS = "刷新后端"
        self.MODELS_PATH = "模型路径："
        self.OUTPUT_PATH = "输出路径："
        self.LOGS_PATH = "日志路径："
        self.BROWSE = "浏览"
        self.AVAILABLE_MODELS = "可用模型："
        self.REFRESH_MODELS = "刷新模型"

        # 量化
        self.QUANTIZATION_TYPE = "量化类型："
        self.ALLOW_REQUANTIZE = "允许重新量化"
        self.LEAVE_OUTPUT_TENSOR = "保留输出张量"
        self.PURE = "纯净"
        self.IMATRIX = "重要性矩阵："
        self.INCLUDE_WEIGHTS = "包含权重："
        self.EXCLUDE_WEIGHTS = "排除权重："
        self.USE_OUTPUT_TENSOR_TYPE = "使用输出张量类型"
        self.USE_TOKEN_EMBEDDING_TYPE = "使用词元嵌入类型"
        self.KEEP_SPLIT = "保持分割"
        self.KV_OVERRIDES = "KV覆盖："
        self.ADD_NEW_OVERRIDE = "添加新覆盖"
        self.QUANTIZE_MODEL = "量化模型"
        self.EXTRA_ARGUMENTS = "额外参数："
        self.EXTRA_ARGUMENTS_LABEL = "附加命令行参数"
        self.QUANTIZATION_COMMAND = "量化命令"

        # 预设
        self.SAVE_PRESET = "保存预设"
        self.LOAD_PRESET = "加载预设"

        # 任务
        self.TASKS = "任务："

        # llama.cpp下载
        self.DOWNLOAD_LLAMACPP = "下载llama.cpp"
        self.SELECT_RELEASE = "选择版本："
        self.SELECT_ASSET = "选择资源："
        self.EXTRACT_CUDA_FILES = "提取CUDA文件"
        self.SELECT_CUDA_BACKEND = "选择CUDA后端："
        self.DOWNLOAD = "下载"
        self.REFRESH_RELEASES = "刷新版本"

        # IMatrix生成
        self.IMATRIX_GENERATION = "IMatrix生成"
        self.DATA_FILE = "数据文件："
        self.MODEL = "模型："
        self.OUTPUT = "输出："
        self.OUTPUT_FREQUENCY = "输出频率："
        self.GPU_OFFLOAD = "GPU卸载："
        self.AUTO = "自动"
        self.GENERATE_IMATRIX = "生成IMatrix"
        self.CONTEXT_SIZE = "上下文大小："
        self.CONTEXT_SIZE_FOR_IMATRIX = "IMatrix生成的上下文大小"
        self.THREADS = "线程数："
        self.NUMBER_OF_THREADS_FOR_IMATRIX = "IMatrix生成的线程数"
        self.IMATRIX_GENERATION_COMMAND = "IMatrix生成命令"

        # LoRA转换
        self.LORA_CONVERSION = "LoRA转换"
        self.LORA_INPUT_PATH = "LoRA输入路径"
        self.LORA_OUTPUT_PATH = "LoRA输出路径"
        self.SELECT_LORA_INPUT_DIRECTORY = "选择LoRA输入目录"
        self.SELECT_LORA_OUTPUT_FILE = "选择LoRA输出文件"
        self.CONVERT_LORA = "转换LoRA"
        self.LORA_CONVERSION_COMMAND = "LoRA转换命令"

        # LoRA导出
        self.EXPORT_LORA = "导出LoRA"
        self.GGML_LORA_ADAPTERS = "GGML LoRA适配器"
        self.SELECT_LORA_ADAPTER_FILES = "选择LoRA适配器文件"
        self.ADD_ADAPTER = "添加适配器"
        self.DELETE_ADAPTER = "删除"
        self.LORA_SCALE = "LoRA比例"
        self.ENTER_LORA_SCALE_VALUE = "输入LoRA比例值（可选）"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = "LoRA导出的线程数"
        self.LORA_EXPORT_COMMAND = "LoRA导出命令"

        # HuggingFace到GGUF转换
        self.HF_TO_GGUF_CONVERSION = "HuggingFace到GGUF转换"
        self.MODEL_DIRECTORY = "模型目录："
        self.OUTPUT_FILE = "输出文件："
        self.OUTPUT_TYPE = "输出类型："
        self.VOCAB_ONLY = "仅词汇表"
        self.USE_TEMP_FILE = "使用临时文件"
        self.NO_LAZY_EVALUATION = "不使用延迟评估"
        self.MODEL_NAME = "模型名称："
        self.VERBOSE = "详细模式"
        self.SPLIT_MAX_SIZE = "最大分割大小："
        self.DRY_RUN = "试运行"
        self.CONVERT_HF_TO_GGUF = "转换HF到GGUF"
        self.SELECT_HF_MODEL_DIRECTORY = "选择HuggingFace模型目录"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = "浏览HuggingFace模型目录"
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = "浏览HuggingFace到GGUF输出文件"

        # 通用消息
        self.ERROR = "错误"
        self.WARNING = "警告"
        self.PROPERTIES = "属性"
        self.CANCEL = "取消"
        self.RESTART = "重启"
        self.DELETE = "删除"
        self.CONFIRM_DELETION = "您确定要删除此任务吗？"
        self.TASK_RUNNING_WARNING = "一些任务仍在运行。您确定要退出吗？"
        self.YES = "是"
        self.NO = "否"
        self.COMPLETED = "已完成"

        # 文件类型
        self.ALL_FILES = "所有文件 (*)"
        self.GGUF_FILES = "GGUF文件 (*.gguf)"
        self.DAT_FILES = "DAT文件 (*.dat)"
        self.JSON_FILES = "JSON文件 (*.json)"
        self.BIN_FILES = "二进制文件 (*.bin)"
        self.LORA_FILES = "LoRA文件 (*.bin)"
        self.GGUF_AND_BIN_FILES = "GGUF和二进制文件 (*.gguf *.bin)"
        self.SHARDED = "分片"

        # 状态消息
        self.DOWNLOAD_COMPLETE = "下载完成"
        self.CUDA_EXTRACTION_FAILED = "CUDA提取失败"
        self.PRESET_SAVED = "预设已保存"
        self.PRESET_LOADED = "预设已加载"
        self.NO_ASSET_SELECTED = "未选择资源"
        self.DOWNLOAD_FAILED = "下载失败"
        self.NO_BACKEND_SELECTED = "未选择后端"
        self.NO_MODEL_SELECTED = "未选择模型"
        self.NO_SUITABLE_CUDA_BACKENDS = "未找到合适的CUDA后端"
        self.IN_PROGRESS = "进行中"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp二进制文件已下载并解压到{0}"
        self.CUDA_FILES_EXTRACTED = "CUDA文件已提取到"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = "未找到合适的CUDA后端进行提取"
        self.ERROR_FETCHING_RELEASES = "获取版本时出错：{0}"
        self.CONFIRM_DELETION_TITLE = "确认删除"
        self.LOG_FOR = "{0}的日志"
        self.FAILED_LOAD_PRESET = "加载预设失败：{0}"
        self.INITIALIZING_AUTOGGUF = "初始化AutoGGUF应用程序"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF初始化完成"
        self.REFRESHING_BACKENDS = "刷新后端"
        self.NO_BACKENDS_AVAILABLE = "没有可用的后端"
        self.FOUND_VALID_BACKENDS = "找到{0}个有效后端"
        self.SAVING_PRESET = "保存预设"
        self.PRESET_SAVED_TO = "预设已保存到{0}"
        self.LOADING_PRESET = "加载预设"
        self.PRESET_LOADED_FROM = "预设已从{0}加载"
        self.ADDING_KV_OVERRIDE = "添加KV覆盖：{0}"
        self.SAVING_TASK_PRESET = "保存{0}的任务预设"
        self.TASK_PRESET_SAVED = "任务预设已保存"
        self.TASK_PRESET_SAVED_TO = "任务预设已保存到{0}"
        self.RESTARTING_TASK = "重启任务：{0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "下载完成。已解压到：{0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp二进制文件已下载并解压到{0}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = "未找到合适的CUDA后端进行提取"
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp二进制文件已下载并解压到{0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "刷新llama.cpp版本"
        self.UPDATING_ASSET_LIST = "更新资源列表"
        self.UPDATING_CUDA_OPTIONS = "更新CUDA选项"
        self.STARTING_LLAMACPP_DOWNLOAD = "开始下载llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "更新CUDA后端"
        self.NO_CUDA_BACKEND_SELECTED = "未选择CUDA后端进行提取"
        self.EXTRACTING_CUDA_FILES = "从{0}提取CUDA文件到{1}"
        self.DOWNLOAD_ERROR = "下载错误：{0}"
        self.SHOWING_TASK_CONTEXT_MENU = "显示任务上下文菜单"
        self.SHOWING_PROPERTIES_FOR_TASK = "显示任务属性：{0}"
        self.CANCELLING_TASK = "取消任务：{0}"
        self.CANCELED = "已取消"
        self.DELETING_TASK = "删除任务：{0}"
        self.LOADING_MODELS = "加载模型"
        self.LOADED_MODELS = "已加载{0}个模型"
        self.BROWSING_FOR_MODELS_DIRECTORY = "浏览模型目录"
        self.SELECT_MODELS_DIRECTORY = "选择模型目录"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "浏览输出目录"
        self.SELECT_OUTPUT_DIRECTORY = "选择输出目录"
        self.BROWSING_FOR_LOGS_DIRECTORY = "浏览日志目录"
        self.SELECT_LOGS_DIRECTORY = "选择日志目录"
        self.BROWSING_FOR_IMATRIX_FILE = "浏览IMatrix文件"
        self.SELECT_IMATRIX_FILE = "选择IMatrix文件"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU使用率：{0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "验证量化输入"
        self.MODELS_PATH_REQUIRED = "需要模型路径"
        self.OUTPUT_PATH_REQUIRED = "需要输出路径"
        self.LOGS_PATH_REQUIRED = "需要日志路径"
        self.STARTING_MODEL_QUANTIZATION = "开始模型量化"
        self.INPUT_FILE_NOT_EXIST = "输入文件'{0}'不存在。"
        self.QUANTIZING_MODEL_TO = "将{0}量化为{1}"
        self.QUANTIZATION_TASK_STARTED = "已开始{0}的量化任务"
        self.ERROR_STARTING_QUANTIZATION = "启动量化时出错：{0}"
        self.UPDATING_MODEL_INFO = "更新模型信息：{0}"
        self.TASK_FINISHED = "任务完成：{0}"
        self.SHOWING_TASK_DETAILS_FOR = "显示任务详情：{0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "浏览IMatrix数据文件"
        self.SELECT_DATA_FILE = "选择数据文件"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "浏览IMatrix模型文件"
        self.SELECT_MODEL_FILE = "选择模型文件"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "浏览IMatrix输出文件"
        self.SELECT_OUTPUT_FILE = "选择输出文件"
        self.STARTING_IMATRIX_GENERATION = "开始IMatrix生成"
        self.BACKEND_PATH_NOT_EXIST = "后端路径不存在：{0}"
        self.GENERATING_IMATRIX = "生成IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = "启动IMatrix生成时出错：{0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix生成任务已开始"
        self.ERROR_MESSAGE = "错误：{0}"
        self.TASK_ERROR = "任务错误：{0}"
        self.APPLICATION_CLOSING = "应用程序正在关闭"
        self.APPLICATION_CLOSED = "应用程序已关闭"
        self.SELECT_QUANTIZATION_TYPE = "选择量化类型"
        self.ALLOWS_REQUANTIZING = "允许重新量化已量化的张量"
        self.LEAVE_OUTPUT_WEIGHT = "将保持output.weight不被（重新）量化"
        self.DISABLE_K_QUANT_MIXTURES = "禁用k-quant混合并将所有张量量化为相同类型"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "使用文件中的数据作为量化优化的重要性矩阵"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = "对这些张量使用重要性矩阵"
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = "不对这些张量使用重要性矩阵"
        self.OUTPUT_TENSOR_TYPE = "输出张量类型："
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = "对output.weight张量使用此类型"
        self.TOKEN_EMBEDDING_TYPE = "词元嵌入类型："
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = "对词元嵌入张量使用此类型"
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "将生成与输入相同分片的量化模型"
        )
        self.OVERRIDE_MODEL_METADATA = "覆盖模型元数据"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix生成的输入数据文件"
        self.MODEL_TO_BE_QUANTIZED = "要量化的模型"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "生成的IMatrix的输出路径"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "保存IMatrix的频率"
        self.SET_GPU_OFFLOAD_VALUE = "设置GPU卸载值（-ngl）"
        self.STARTING_LORA_CONVERSION = "开始LoRA转换"
        self.LORA_INPUT_PATH_REQUIRED = "需要LoRA输入路径。"
        self.LORA_OUTPUT_PATH_REQUIRED = "需要LoRA输出路径。"
        self.ERROR_STARTING_LORA_CONVERSION = "启动LoRA转换时出错：{}"
        self.LORA_CONVERSION_TASK_STARTED = "LoRA转换任务已开始。"
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = "浏览LoRA输入目录..."
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "浏览LoRA输出文件..."
        self.CONVERTING_LORA = "LoRA转换"
        self.LORA_CONVERSION_FINISHED = "LoRA转换完成。"
        self.LORA_FILE_MOVED = "LoRA文件已从{}移动到{}。"
        self.LORA_FILE_NOT_FOUND = "未找到LoRA文件：{}。"
        self.ERROR_MOVING_LORA_FILE = "移动LoRA文件时出错：{}"
        self.MODEL_PATH_REQUIRED = "需要模型路径。"
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = "至少需要一个LoRA适配器。"
        self.INVALID_LORA_SCALE_VALUE = "无效的LoRA比例值。"
        self.ERROR_STARTING_LORA_EXPORT = "启动LoRA导出时出错：{}"
        self.LORA_EXPORT_TASK_STARTED = "LoRA导出任务已开始。"
        self.EXPORTING_LORA = "导出LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = "浏览导出LoRA模型文件..."
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = "浏览导出LoRA输出文件..."
        self.ADDING_LORA_ADAPTER = "添加LoRA适配器..."
        self.DELETING_LORA_ADAPTER = "删除LoRA适配器..."
        self.SELECT_LORA_ADAPTER_FILE = "选择LoRA适配器文件"
        self.STARTING_LORA_EXPORT = "开始LoRA导出..."
        self.SELECT_OUTPUT_TYPE = "选择输出类型（GGUF或GGML）"
        self.BASE_MODEL = "基础模型"
        self.SELECT_BASE_MODEL_FILE = "选择基础模型文件（GGUF）"
        self.BASE_MODEL_PATH_REQUIRED = "GGUF输出需要基础模型路径。"
        self.BROWSING_FOR_BASE_MODEL_FILE = "浏览基础模型文件..."
        self.SELECT_BASE_MODEL_FOLDER = "选择基础模型文件夹（包含safetensors）"
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "浏览基础模型文件夹..."
        self.LORA_CONVERSION_FROM_TO = "LoRA转换从{}到{}"
        self.GENERATING_IMATRIX_FOR = "为{}生成IMatrix"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = "IMatrix生成需要模型路径。"
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = "未选择用于CUDA检查的资源"
        self.NO_QUANTIZATION_TYPE_SELECTED = "未选择量化类型。请至少选择一种量化类型。"
        self.STARTING_HF_TO_GGUF_CONVERSION = "开始HuggingFace到GGUF转换"
        self.MODEL_DIRECTORY_REQUIRED = "需要模型目录"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "HF到GGUF转换命令：{}"
        self.CONVERTING_TO_GGUF = "将{}转换为GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "启动HuggingFace到GGUF转换时出错：{}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = "HuggingFace到GGUF转换任务已开始"


class _Spanish(_Localization):
    def __init__(self):
        super().__init__()

        # Interfaz de usuario general
        self.WINDOW_TITLE = "AutoGGUF (cuantificador de modelos GGUF automatizado)"
        self.RAM_USAGE = "Uso de RAM:"
        self.CPU_USAGE = "Uso de CPU:"
        self.BACKEND = "Backend de Llama.cpp:"
        self.REFRESH_BACKENDS = "Actualizar Backends"
        self.MODELS_PATH = "Ruta de los Modelos:"
        self.OUTPUT_PATH = "Ruta de Salida:"
        self.LOGS_PATH = "Ruta de Registros:"
        self.BROWSE = "Examinar"
        self.AVAILABLE_MODELS = "Modelos Disponibles:"
        self.REFRESH_MODELS = "Actualizar Modelos"

        # Importación del Modelo
        self.IMPORT_MODEL = "Importar Modelo"
        self.SELECT_MODEL_TO_IMPORT = "Seleccionar Modelo para Importar"
        self.CONFIRM_IMPORT = "Confirmar Importación"
        self.IMPORT_MODEL_CONFIRMATION = "¿Desea importar el modelo {}?"
        self.MODEL_IMPORTED_SUCCESSFULLY = "Modelo {} importado correctamente"
        self.IMPORTING_MODEL = "Importando modelo"
        self.IMPORTED_MODEL_TOOLTIP = "Modelo importado: {}"

        # Verificación GGUF
        self.INVALID_GGUF_FILE = "Archivo GGUF inválido: {}"
        self.SHARDED_MODEL_NAME = "{} (Fragmentado)"
        self.IMPORTED_MODEL_TOOLTIP = "Modelo importado: {}"
        self.CONCATENATED_FILE_WARNING = "Esta es una parte de un archivo concatenado. No funcionará con llama-quantize; por favor, concatena el archivo primero."
        self.CONCATENATED_FILES_FOUND = "Se encontraron {} partes de archivos concatenados. Por favor, concatena los archivos primero."

        # Plugins
        self.PLUGINS_DIR_NOT_EXIST = (
            "El directorio de plugins '{}' no existe. No se cargarán plugins."
        )
        self.PLUGINS_DIR_NOT_DIRECTORY = (
            "'{}' existe pero no es un directorio. No se cargarán plugins."
        )
        self.PLUGIN_LOADED = "Plugin cargado: {} {}"
        self.PLUGIN_INCOMPATIBLE = "El plugin {} {} no es compatible con la versión {} de AutoGGUF. Versiones compatibles: {}"
        self.PLUGIN_LOAD_FAILED = "Error al cargar el plugin {}: {}"
        self.NO_PLUGINS_LOADED = "No se han cargado plugins."

        # Monitoreo de GPU
        self.GPU_USAGE = "Uso de GPU:"
        self.GPU_USAGE_FORMAT = "GPU: {:.1f}% | VRAM: {:.1f}% ({} MB / {} MB)"
        self.GPU_DETAILS = "Detalles de la GPU"
        self.GPU_USAGE_OVER_TIME = "Uso de la GPU a lo largo del tiempo"
        self.VRAM_USAGE_OVER_TIME = "Uso de la VRAM a lo largo del tiempo"
        self.PERCENTAGE = "Porcentaje"
        self.TIME = "Tiempo (s)"
        self.NO_GPU_DETECTED = "No se ha detectado ninguna GPU"
        self.SELECT_GPU = "Seleccionar GPU"
        self.AMD_GPU_NOT_SUPPORTED = "GPU AMD detectada, pero no compatible"

        # Cuantización
        self.QUANTIZATION_TYPE = "Tipo de Cuantización:"
        self.ALLOW_REQUANTIZE = "Permitir Recuantización"
        self.LEAVE_OUTPUT_TENSOR = "Dejar Tensor de Salida"
        self.PURE = "Puro"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Incluir Pesos:"
        self.EXCLUDE_WEIGHTS = "Excluir Pesos:"
        self.USE_OUTPUT_TENSOR_TYPE = "Usar Tipo de Tensor de Salida"
        self.USE_TOKEN_EMBEDDING_TYPE = "Usar Tipo de Incrustación de Tokens"
        self.KEEP_SPLIT = "Mantener División"
        self.KV_OVERRIDES = "Anulaciones de KV:"
        self.ADD_NEW_OVERRIDE = "Agregar nueva anulación"
        self.QUANTIZE_MODEL = "Cuantizar Modelo"
        self.EXTRA_ARGUMENTS = "Argumentos Adicionales:"
        self.EXTRA_ARGUMENTS_LABEL = "Argumentos de línea de comandos adicionales"
        self.QUANTIZATION_COMMAND = "Comando de cuantización"

        # Ajustes Preestablecidos
        self.SAVE_PRESET = "Guardar Ajuste Preestablecido"
        self.LOAD_PRESET = "Cargar Ajuste Preestablecido"

        # Tareas
        self.TASKS = "Tareas:"

        # Descarga de llama.cpp
        self.DOWNLOAD_LLAMACPP = "Descargar llama.cpp"
        self.SELECT_RELEASE = "Seleccionar Versión:"
        self.SELECT_ASSET = "Seleccionar Activo:"
        self.EXTRACT_CUDA_FILES = "Extraer archivos CUDA"
        self.SELECT_CUDA_BACKEND = "Seleccionar Backend CUDA:"
        self.DOWNLOAD = "Descargar"
        self.REFRESH_RELEASES = "Actualizar Versiones"

        # Generación de IMatrix
        self.IMATRIX_GENERATION = "Generación de IMatrix"
        self.DATA_FILE = "Archivo de Datos:"
        self.MODEL = "Modelo:"
        self.OUTPUT = "Salida:"
        self.OUTPUT_FREQUENCY = "Frecuencia de Salida:"
        self.GPU_OFFLOAD = "Descarga de GPU:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Generar IMatrix"
        self.CONTEXT_SIZE = "Tamaño del Contexto:"
        self.CONTEXT_SIZE_FOR_IMATRIX = (
            "Tamaño del contexto para la generación de IMatrix"
        )
        self.THREADS = "Hilos:"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = (
            "Número de hilos para la generación de IMatrix"
        )
        self.IMATRIX_GENERATION_COMMAND = "Comando de generación de IMatrix"

        # Conversión de LoRA
        self.LORA_CONVERSION = "Conversión de LoRA"
        self.LORA_INPUT_PATH = "Ruta de Entrada de LoRA"
        self.LORA_OUTPUT_PATH = "Ruta de Salida de LoRA"
        self.SELECT_LORA_INPUT_DIRECTORY = "Seleccionar Directorio de Entrada de LoRA"
        self.SELECT_LORA_OUTPUT_FILE = "Seleccionar Archivo de Salida de LoRA"
        self.CONVERT_LORA = "Convertir LoRA"
        self.LORA_CONVERSION_COMMAND = "Comando de conversión de LoRA"

        # Exportación de LoRA
        self.EXPORT_LORA = "Exportar LoRA"
        self.GGML_LORA_ADAPTERS = "Adaptadores LoRA GGML"
        self.SELECT_LORA_ADAPTER_FILES = "Seleccionar Archivos de Adaptador LoRA"
        self.ADD_ADAPTER = "Agregar Adaptador"
        self.DELETE_ADAPTER = "Eliminar"
        self.LORA_SCALE = "Escala de LoRA"
        self.ENTER_LORA_SCALE_VALUE = (
            "Introducir el valor de la escala de LoRA (opcional)"
        )
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = (
            "Número de hilos para la exportación de LoRA"
        )
        self.LORA_EXPORT_COMMAND = "Comando de exportación de LoRA"

        # Conversión de HuggingFace a GGUF
        self.HF_TO_GGUF_CONVERSION = "Conversión de HuggingFace a GGUF"
        self.MODEL_DIRECTORY = "Directorio del Modelo:"
        self.OUTPUT_FILE = "Archivo de Salida:"
        self.OUTPUT_TYPE = "Tipo de Salida:"
        self.VOCAB_ONLY = "Sólo Vocabulario"
        self.USE_TEMP_FILE = "Usar Archivo Temporal"
        self.NO_LAZY_EVALUATION = "Sin Evaluación Perezosa"
        self.MODEL_NAME = "Nombre del Modelo:"
        self.VERBOSE = "Verboso"
        self.SPLIT_MAX_SIZE = "Tamaño Máximo de División:"
        self.DRY_RUN = "Ejecución de Prueba"
        self.CONVERT_HF_TO_GGUF = "Convertir HF a GGUF"
        self.SELECT_HF_MODEL_DIRECTORY = "Seleccionar Directorio del Modelo HuggingFace"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = (
            "Buscando el directorio del modelo HuggingFace"
        )
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = (
            "Buscando el archivo de salida de HuggingFace a GGUF"
        )

        # Comprobación de Actualizaciones
        self.UPDATE_AVAILABLE = "Actualización Disponible"
        self.NEW_VERSION_AVAILABLE = "Hay una nueva versión disponible: {}"
        self.DOWNLOAD_NEW_VERSION = "¿Descargar?"
        self.ERROR_CHECKING_FOR_UPDATES = "Error al buscar actualizaciones:"
        self.CHECKING_FOR_UPDATES = "Buscando actualizaciones"

        # Mensajes Generales
        self.ERROR = "Error"
        self.WARNING = "Advertencia"
        self.PROPERTIES = "Propiedades"
        self.CANCEL = "Cancelar"
        self.RESTART = "Reiniciar"
        self.DELETE = "Eliminar"
        self.CONFIRM_DELETION = "¿Está seguro de que desea eliminar esta tarea?"
        self.TASK_RUNNING_WARNING = (
            "Algunas tareas siguen en ejecución. ¿Está seguro de que desea salir?"
        )
        self.YES = "Sí"
        self.NO = "No"
        self.COMPLETED = "Completado"

        # Tipos de Archivos
        self.ALL_FILES = "Todos los Archivos (*)"
        self.GGUF_FILES = "Archivos GGUF (*.gguf)"
        self.DAT_FILES = "Archivos DAT (*.dat)"
        self.JSON_FILES = "Archivos JSON (*.json)"
        self.BIN_FILES = "Archivos Binarios (*.bin)"
        self.LORA_FILES = "Archivos LoRA (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "Archivos GGUF y Binarios (*.gguf *.bin)"
        self.SHARDED = "fragmentado"

        # Mensajes de Estado
        self.DOWNLOAD_COMPLETE = "Descarga Completa"
        self.CUDA_EXTRACTION_FAILED = "Error en la Extracción de CUDA"
        self.PRESET_SAVED = "Ajuste Preestablecido Guardado"
        self.PRESET_LOADED = "Ajuste Preestablecido Cargado"
        self.NO_ASSET_SELECTED = "No se ha seleccionado ningún activo"
        self.DOWNLOAD_FAILED = "Error en la descarga"
        self.NO_BACKEND_SELECTED = "No se ha seleccionado ningún backend"
        self.NO_MODEL_SELECTED = "No se ha seleccionado ningún modelo"
        self.NO_SUITABLE_CUDA_BACKENDS = "No se han encontrado backends CUDA adecuados"
        self.IN_PROGRESS = "En Progreso"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "Binario de llama.cpp descargado y extraído a {0}"
        )
        self.CUDA_FILES_EXTRACTED = "Archivos CUDA extraídos a"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "No se ha encontrado ningún backend CUDA adecuado para la extracción"
        )
        self.ERROR_FETCHING_RELEASES = "Error al obtener las versiones: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirmar Eliminación"
        self.LOG_FOR = "Registro para {0}"
        self.FAILED_TO_LOAD_PRESET = "Error al cargar el ajuste preestablecido: {0}"
        self.INITIALIZING_AUTOGGUF = "Inicializando la aplicación AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Inicialización de AutoGGUF completada"
        self.REFRESHING_BACKENDS = "Actualizando backends"
        self.NO_BACKENDS_AVAILABLE = "No hay backends disponibles"
        self.FOUND_VALID_BACKENDS = "Se han encontrado {0} backends válidos"
        self.SAVING_PRESET = "Guardando ajuste preestablecido"
        self.PRESET_SAVED_TO = "Ajuste preestablecido guardado en {0}"
        self.LOADING_PRESET = "Cargando ajuste preestablecido"
        self.PRESET_LOADED_FROM = "Ajuste preestablecido cargado desde {0}"
        self.ADDING_KV_OVERRIDE = "Añadiendo anulación de KV: {0}"
        self.SAVING_TASK_PRESET = "Guardando ajuste preestablecido de tarea para {0}"
        self.TASK_PRESET_SAVED = "Ajuste Preestablecido de Tarea Guardado"
        self.TASK_PRESET_SAVED_TO = "Ajuste preestablecido de tarea guardado en {0}"
        self.RESTARTING_TASK = "Reiniciando tarea: {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Descarga finalizada. Extraído a: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "Binario de llama.cpp descargado y extraído a {0}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "No se ha encontrado ningún backend CUDA adecuado para la extracción"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Binario de llama.cpp descargado y extraído a {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Actualizando versiones de llama.cpp"
        self.UPDATING_ASSET_LIST = "Actualizando lista de activos"
        self.UPDATING_CUDA_OPTIONS = "Actualizando opciones de CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Iniciando descarga de llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Actualizando backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = (
            "No se ha seleccionado ningún backend CUDA para la extracción"
        )
        self.EXTRACTING_CUDA_FILES = "Extrayendo archivos CUDA de {0} a {1}"
        self.DOWNLOAD_ERROR = "Error de descarga: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Mostrando menú contextual de la tarea"
        self.SHOWING_PROPERTIES_FOR_TASK = "Mostrando propiedades para la tarea: {0}"
        self.CANCELLING_TASK = "Cancelando tarea: {0}"
        self.CANCELED = "Cancelado"
        self.DELETING_TASK = "Eliminando tarea: {0}"
        self.LOADING_MODELS = "Cargando modelos"
        self.LOADED_MODELS = "Cargados {0} modelos"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Buscando el directorio de modelos"
        self.SELECT_MODELS_DIRECTORY = "Seleccionar Directorio de Modelos"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Buscando el directorio de salida"
        self.SELECT_OUTPUT_DIRECTORY = "Seleccionar Directorio de Salida"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Buscando el directorio de registros"
        self.SELECT_LOGS_DIRECTORY = "Seleccionar Directorio de Registros"
        self.BROWSING_FOR_IMATRIX_FILE = "Buscando el archivo IMatrix"
        self.SELECT_IMATRIX_FILE = "Seleccionar Archivo IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Uso de CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validando entradas de cuantización"
        self.MODELS_PATH_REQUIRED = "Se requiere la ruta de los modelos"
        self.OUTPUT_PATH_REQUIRED = "Se requiere la ruta de salida"
        self.LOGS_PATH_REQUIRED = "Se requiere la ruta de registros"
        self.STARTING_MODEL_QUANTIZATION = "Iniciando la cuantización del modelo"
        self.INPUT_FILE_NOT_EXIST = "El archivo de entrada '{0}' no existe."
        self.QUANTIZING_MODEL_TO = "Cuantizando {0} a {1}"
        self.QUANTIZATION_TASK_STARTED = "Tarea de cuantización iniciada para {0}"
        self.ERROR_STARTING_QUANTIZATION = "Error al iniciar la cuantización: {0}"
        self.UPDATING_MODEL_INFO = "Actualizando información del modelo: {0}"
        self.TASK_FINISHED = "Tarea finalizada: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Mostrando detalles de la tarea para: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Buscando el archivo de datos de IMatrix"
        self.SELECT_DATA_FILE = "Seleccionar Archivo de Datos"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = (
            "Buscando el archivo de modelo de IMatrix"
        )
        self.SELECT_MODEL_FILE = "Seleccionar Archivo de Modelo"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = (
            "Buscando el archivo de salida de IMatrix"
        )
        self.SELECT_OUTPUT_FILE = "Seleccionar Archivo de Salida"
        self.STARTING_IMATRIX_GENERATION = "Iniciando la generación de IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "La ruta del backend no existe: {0}"
        self.GENERATING_IMATRIX = "Generando IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Error al iniciar la generación de IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Tarea de generación de IMatrix iniciada"
        self.ERROR_MESSAGE = "Error: {0}"
        self.TASK_ERROR = "Error de tarea: {0}"
        self.APPLICATION_CLOSING = "Cerrando la aplicación"
        self.APPLICATION_CLOSED = "Aplicación cerrada"
        self.SELECT_QUANTIZATION_TYPE = "Seleccione el tipo de cuantización"
        self.ALLOWS_REQUANTIZING = (
            "Permite recuantizar tensores que ya han sido cuantizados"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Dejará output.weight sin (re)cuantizar"
        self.DISABLE_K_QUANT_MIXTURES = "Desactivar las mezclas k-quant y cuantizar todos los tensores al mismo tipo"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Utilizar los datos del archivo como matriz de importancia para las optimizaciones de cuantización"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Utilizar la matriz de importancia para estos tensores"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "No utilizar la matriz de importancia para estos tensores"
        )
        self.OUTPUT_TENSOR_TYPE = "Tipo de Tensor de Salida:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Utilizar este tipo para el tensor output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Tipo de Incrustación de Tokens:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Utilizar este tipo para el tensor de incrustaciones de tokens"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Generará el modelo cuantizado en los mismos fragmentos que la entrada"
        )
        self.OVERRIDE_MODEL_METADATA = "Sobrescribir los metadatos del modelo"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Archivo de datos de entrada para la generación de IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modelo a cuantizar"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Ruta de salida para la IMatrix generada"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Con qué frecuencia guardar la IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Establecer el valor de descarga de la GPU (-ngl)"
        self.STARTING_LORA_CONVERSION = "Iniciando la conversión de LoRA"
        self.LORA_INPUT_PATH_REQUIRED = "Se requiere la ruta de entrada de LoRA."
        self.LORA_OUTPUT_PATH_REQUIRED = "Se requiere la ruta de salida de LoRA."
        self.ERROR_STARTING_LORA_CONVERSION = (
            "Error al iniciar la conversión de LoRA: {}"
        )
        self.LORA_CONVERSION_TASK_STARTED = "Tarea de conversión de LoRA iniciada."
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = (
            "Buscando el directorio de entrada de LoRA..."
        )
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "Buscando el archivo de salida de LoRA..."
        self.CONVERTING_LORA = "Conversión de LoRA"
        self.LORA_CONVERSION_FINISHED = "Conversión de LoRA finalizada."
        self.LORA_FILE_MOVED = "Archivo LoRA movido de {} a {}."
        self.LORA_FILE_NOT_FOUND = "Archivo LoRA no encontrado: {}."
        self.ERROR_MOVING_LORA_FILE = "Error al mover el archivo LoRA: {}"
        self.MODEL_PATH_REQUIRED = "Se requiere la ruta del modelo."
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = (
            "Se requiere al menos un adaptador LoRA."
        )
        self.INVALID_LORA_SCALE_VALUE = "Valor de escala de LoRA no válido."
        self.ERROR_STARTING_LORA_EXPORT = "Error al iniciar la exportación de LoRA: {}"
        self.LORA_EXPORT_TASK_STARTED = "Tarea de exportación de LoRA iniciada."
        self.EXPORTING_LORA = "Exportando LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = (
            "Buscando el archivo de modelo de exportación de LoRA..."
        )
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = (
            "Buscando el archivo de salida de exportación de LoRA..."
        )
        self.ADDING_LORA_ADAPTER = "Añadiendo adaptador LoRA..."
        self.DELETING_LORA_ADAPTER = "Eliminando adaptador LoRA..."
        self.SELECT_LORA_ADAPTER_FILE = "Seleccionar archivo de adaptador LoRA"
        self.STARTING_LORA_EXPORT = "Iniciando la exportación de LoRA..."
        self.SELECT_OUTPUT_TYPE = "Seleccionar tipo de salida (GGUF o GGML)"
        self.BASE_MODEL = "Modelo base"
        self.SELECT_BASE_MODEL_FILE = "Seleccionar archivo de modelo base (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = (
            "Se requiere la ruta del modelo base para la salida GGUF."
        )
        self.BROWSING_FOR_BASE_MODEL_FILE = "Buscando el archivo de modelo base..."
        self.SELECT_BASE_MODEL_FOLDER = (
            "Seleccionar carpeta de modelo base (que contenga safetensors)"
        )
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "Buscando la carpeta de modelo base..."
        self.LORA_CONVERSION_FROM_TO = "Conversión de LoRA de {} a {}"
        self.GENERATING_IMATRIX_FOR = "Generando IMatrix para {}"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = (
            "Se requiere la ruta del modelo para la generación de IMatrix."
        )
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = (
            "No se ha seleccionado ningún activo para la comprobación de CUDA"
        )
        self.NO_QUANTIZATION_TYPE_SELECTED = "No se ha seleccionado ningún tipo de cuantización. Por favor, seleccione al menos un tipo de cuantización."
        self.STARTING_HF_TO_GGUF_CONVERSION = (
            "Iniciando la conversión de HuggingFace a GGUF"
        )
        self.MODEL_DIRECTORY_REQUIRED = "Se requiere el directorio del modelo"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "Comando de conversión de HF a GGUF: {}"
        self.CONVERTING_TO_GGUF = "Convirtiendo {} a GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "Error al iniciar la conversión de HuggingFace a GGUF: {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = (
            "Tarea de conversión de HuggingFace a GGUF iniciada"
        )


class _Hindi(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (स्वचालित GGUF मॉडल क्वांटाइज़र)"
        self.RAM_USAGE = "RAM उपयोग:"
        self.CPU_USAGE = "CPU उपयोग:"
        self.BACKEND = "Llama.cpp बैकएंड:"
        self.REFRESH_BACKENDS = "बैकएंड रीफ्रेश करें"
        self.MODELS_PATH = "मॉडल पथ:"
        self.OUTPUT_PATH = "आउटपुट पथ:"
        self.LOGS_PATH = "लॉग पथ:"
        self.BROWSE = "ब्राउज़ करें"
        self.AVAILABLE_MODELS = "उपलब्ध मॉडल:"
        self.QUANTIZATION_TYPE = "क्वांटाइजेशन प्रकार:"
        self.ALLOW_REQUANTIZE = "पुनः क्वांटाइज़ करने की अनुमति दें"
        self.LEAVE_OUTPUT_TENSOR = "आउटपुट टेंसर छोड़ें"
        self.PURE = "शुद्ध"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "वेट शामिल करें:"
        self.EXCLUDE_WEIGHTS = "वेट बाहर रखें:"
        self.USE_OUTPUT_TENSOR_TYPE = "आउटपुट टेंसर प्रकार का उपयोग करें"
        self.USE_TOKEN_EMBEDDING_TYPE = "टोकन एम्बेडिंग प्रकार का उपयोग करें"
        self.KEEP_SPLIT = "विभाजन रखें"
        self.KV_OVERRIDES = "KV ओवरराइड:"
        self.ADD_NEW_OVERRIDE = "नया ओवरराइड जोड़ें"
        self.QUANTIZE_MODEL = "मॉडल क्वांटाइज़ करें"
        self.SAVE_PRESET = "प्रीसेट सहेजें"
        self.LOAD_PRESET = "प्रीसेट लोड करें"
        self.TASKS = "कार्य:"
        self.DOWNLOAD_LLAMACPP = "llama.cpp डाउनलोड करें"
        self.SELECT_RELEASE = "रिलीज़ चुनें:"
        self.SELECT_ASSET = "एसेट चुनें:"
        self.EXTRACT_CUDA_FILES = "CUDA फ़ाइलें निकालें"
        self.SELECT_CUDA_BACKEND = "CUDA बैकएंड चुनें:"
        self.DOWNLOAD = "डाउनलोड करें"
        self.IMATRIX_GENERATION = "IMatrix उत्पादन"
        self.DATA_FILE = "डेटा फ़ाइल:"
        self.MODEL = "मॉडल:"
        self.OUTPUT = "आउटपुट:"
        self.OUTPUT_FREQUENCY = "आउटपुट आवृत्ति:"
        self.GPU_OFFLOAD = "GPU ऑफलोड:"
        self.AUTO = "स्वचालित"
        self.GENERATE_IMATRIX = "IMatrix उत्पन्न करें"
        self.ERROR = "त्रुटि"
        self.WARNING = "चेतावनी"
        self.PROPERTIES = "गुण"
        self.WINDOW_TITLE = "AutoGGUF (स्वचालित GGUF मॉडल क्वांटाइज़र)"
        self.RAM_USAGE = "RAM उपयोग:"
        self.CPU_USAGE = "CPU उपयोग:"
        self.BACKEND = "Llama.cpp बैकएंड:"
        self.REFRESH_BACKENDS = "बैकएंड रीफ्रेश करें"
        self.MODELS_PATH = "मॉडल पथ:"
        self.OUTPUT_PATH = "आउटपुट पथ:"
        self.LOGS_PATH = "लॉग पथ:"
        self.BROWSE = "ब्राउज़ करें"
        self.AVAILABLE_MODELS = "उपलब्ध मॉडल:"
        self.QUANTIZATION_TYPE = "क्वांटाइजेशन प्रकार:"
        self.ALLOW_REQUANTIZE = "पुनः क्वांटाइज़ करने की अनुमति दें"
        self.LEAVE_OUTPUT_TENSOR = "आउटपुट टेंसर छोड़ें"
        self.PURE = "शुद्ध"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "वेट शामिल करें:"
        self.EXCLUDE_WEIGHTS = "वेट बाहर रखें:"
        self.USE_OUTPUT_TENSOR_TYPE = "आउटपुट टेंसर प्रकार का उपयोग करें"
        self.USE_TOKEN_EMBEDDING_TYPE = "टोकन एम्बेडिंग प्रकार का उपयोग करें"
        self.KEEP_SPLIT = "विभाजन रखें"
        self.KV_OVERRIDES = "KV ओवरराइड:"
        self.ADD_NEW_OVERRIDE = "नया ओवरराइड जोड़ें"
        self.QUANTIZE_MODEL = "मॉडल क्वांटाइज़ करें"
        self.SAVE_PRESET = "प्रीसेट सहेजें"
        self.LOAD_PRESET = "प्रीसेट लोड करें"
        self.TASKS = "कार्य:"
        self.DOWNLOAD_LLAMACPP = "llama.cpp डाउनलोड करें"
        self.SELECT_RELEASE = "रिलीज़ चुनें:"
        self.SELECT_ASSET = "एसेट चुनें:"
        self.EXTRACT_CUDA_FILES = "CUDA फ़ाइलें निकालें"
        self.SELECT_CUDA_BACKEND = "CUDA बैकएंड चुनें:"
        self.DOWNLOAD = "डाउनलोड करें"
        self.IMATRIX_GENERATION = "IMatrix उत्पादन"
        self.DATA_FILE = "डेटा फ़ाइल:"
        self.MODEL = "मॉडल:"
        self.OUTPUT = "आउटपुट:"
        self.OUTPUT_FREQUENCY = "आउटपुट आवृत्ति:"
        self.GPU_OFFLOAD = "GPU ऑफलोड:"
        self.AUTO = "स्वचालित"
        self.GENERATE_IMATRIX = "IMatrix उत्पन्न करें"
        self.ERROR = "त्रुटि"
        self.WARNING = "चेतावनी"
        self.PROPERTIES = "गुण"
        self.CANCEL = "रद्द करें"
        self.RESTART = "पुनः आरंभ करें"
        self.DELETE = "हटाएं"
        self.CONFIRM_DELETION = "क्या आप वाकई इस कार्य को हटाना चाहते हैं?"
        self.TASK_RUNNING_WARNING = (
            "कुछ कार्य अभी भी चल रहे हैं। क्या आप वाकई बाहर निकलना चाहते हैं?"
        )
        self.YES = "हां"
        self.NO = "नहीं"
        self.DOWNLOAD_COMPLETE = "डाउनलोड पूरा हुआ"
        self.CUDA_EXTRACTION_FAILED = "CUDA निष्कर्षण विफल"
        self.PRESET_SAVED = "प्रीसेट सहेजा गया"
        self.PRESET_LOADED = "प्रीसेट लोड किया गया"
        self.NO_ASSET_SELECTED = "कोई एसेट चयनित नहीं"
        self.DOWNLOAD_FAILED = "डाउनलोड विफल"
        self.NO_BACKEND_SELECTED = "कोई बैकएंड चयनित नहीं"
        self.NO_MODEL_SELECTED = "कोई मॉडल चयनित नहीं"
        self.REFRESH_RELEASES = "रिलीज़ रीफ्रेश करें"
        self.NO_SUITABLE_CUDA_BACKENDS = "कोई उपयुक्त CUDA बैकएंड नहीं मिला"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "llama.cpp बाइनरी डाउनलोड और {0} में निकाली गई\nCUDA फ़ाइलें {1} में निकाली गईं"
        )
        self.CUDA_FILES_EXTRACTED = "CUDA फ़ाइलें निकाली गईं"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "निष्कर्षण के लिए कोई उपयुक्त CUDA बैकएंड नहीं मिला"
        )
        self.ERROR_FETCHING_RELEASES = "रिलीज़ प्राप्त करने में त्रुटि: {0}"
        self.CONFIRM_DELETION_TITLE = "हटाने की पुष्टि करें"
        self.LOG_FOR = "{0} के लिए लॉग"
        self.ALL_FILES = "सभी फ़ाइलें (*)"
        self.GGUF_FILES = "GGUF फ़ाइलें (*.gguf)"
        self.DAT_FILES = "DAT फ़ाइलें (*.dat)"
        self.JSON_FILES = "JSON फ़ाइलें (*.json)"
        self.FAILED_LOAD_PRESET = "प्रीसेट लोड करने में विफल: {0}"
        self.INITIALIZING_AUTOGGUF = "AutoGGUF एप्लिकेशन प्रारंभ हो रहा है"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF प्रारंभीकरण पूरा हुआ"
        self.REFRESHING_BACKENDS = "बैकएंड रीफ्रेश हो रहे हैं"
        self.NO_BACKENDS_AVAILABLE = "कोई बैकएंड उपलब्ध नहीं"
        self.FOUND_VALID_BACKENDS = "{0} मान्य बैकएंड मिले"
        self.SAVING_PRESET = "प्रीसेट सहेजा जा रहा है"
        self.PRESET_SAVED_TO = "प्रीसेट {0} में सहेजा गया"
        self.LOADING_PRESET = "प्रीसेट लोड हो रहा है"
        self.PRESET_LOADED_FROM = "{0} से प्रीसेट लोड किया गया"
        self.ADDING_KV_OVERRIDE = "KV ओवरराइड जोड़ा जा रहा है: {0}"
        self.SAVING_TASK_PRESET = "{0} के लिए कार्य प्रीसेट सहेजा जा रहा है"
        self.TASK_PRESET_SAVED = "कार्य प्रीसेट सहेजा गया"
        self.TASK_PRESET_SAVED_TO = "कार्य प्रीसेट {0} में सहेजा गया"
        self.RESTARTING_TASK = "कार्य पुनः आरंभ हो रहा है: {0}"
        self.IN_PROGRESS = "प्रगति में"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "डाउनलोड समाप्त। निकाला गया: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp बाइनरी डाउनलोड और {0} में निकाली गई\nCUDA फ़ाइलें {1} में निकाली गईं"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "निष्कर्षण के लिए कोई उपयुक्त CUDA बैकएंड नहीं मिला"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp बाइनरी डाउनलोड और {0} में निकाली गई"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "llama.cpp रिलीज़ रीफ्रेश हो रही हैं"
        self.UPDATING_ASSET_LIST = "एसेट सूची अपडेट हो रही है"
        self.UPDATING_CUDA_OPTIONS = "CUDA विकल्प अपडेट हो रहे हैं"
        self.STARTING_LLAMACPP_DOWNLOAD = "llama.cpp डाउनलोड शुरू हो रहा है"
        self.UPDATING_CUDA_BACKENDS = "CUDA बैकएंड अपडेट हो रहे हैं"
        self.NO_CUDA_BACKEND_SELECTED = "निष्कर्षण के लिए कोई CUDA बैकएंड चयनित नहीं"
        self.EXTRACTING_CUDA_FILES = "{0} से {1} में CUDA फ़ाइलें निकाली जा रही हैं"
        self.DOWNLOAD_ERROR = "डाउनलोड त्रुटि: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "कार्य संदर्भ मेनू दिखाया जा रहा है"
        self.SHOWING_PROPERTIES_FOR_TASK = "कार्य के लिए गुण दिखाए जा रहे हैं: {0}"
        self.CANCELLING_TASK = "कार्य रद्द किया जा रहा है: {0}"
        self.CANCELED = "रद्द किया गया"
        self.DELETING_TASK = "कार्य हटाया जा रहा है: {0}"
        self.LOADING_MODELS = "मॉडल लोड हो रहे हैं"
        self.LOADED_MODELS = "{0} मॉडल लोड किए गए"
        self.BROWSING_FOR_MODELS_DIRECTORY = "मॉडल निर्देशिका के लिए ब्राउज़ किया जा रहा है"
        self.SELECT_MODELS_DIRECTORY = "मॉडल निर्देशिका चुनें"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "आउटपुट निर्देशिका के लिए ब्राउज़ किया जा रहा है"
        self.SELECT_OUTPUT_DIRECTORY = "आउटपुट निर्देशिका चुनें"
        self.BROWSING_FOR_LOGS_DIRECTORY = "लॉग निर्देशिका के लिए ब्राउज़ किया जा रहा है"
        self.SELECT_LOGS_DIRECTORY = "लॉग निर्देशिका चुनें"
        self.BROWSING_FOR_IMATRIX_FILE = "IMatrix फ़ाइल के लिए ब्राउज़ किया जा रहा है"
        self.SELECT_IMATRIX_FILE = "IMatrix फ़ाइल चुनें"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU उपयोग: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "क्वांटाइजेशन इनपुट सत्यापित किए जा रहे हैं"
        self.MODELS_PATH_REQUIRED = "मॉडल पथ आवश्यक है"
        self.OUTPUT_PATH_REQUIRED = "आउटपुट पथ आवश्यक है"
        self.LOGS_PATH_REQUIRED = "लॉग पथ आवश्यक है"
        self.STARTING_MODEL_QUANTIZATION = "मॉडल क्वांटाइजेशन शुरू हो रहा है"
        self.INPUT_FILE_NOT_EXIST = "इनपुट फ़ाइल '{0}' मौजूद नहीं है।"
        self.QUANTIZING_MODEL_TO = "{0} को {1} में क्वांटाइज़ किया जा रहा है"
        self.QUANTIZATION_TASK_STARTED = "{0} के लिए क्वांटाइजेशन कार्य शुरू हुआ"
        self.ERROR_STARTING_QUANTIZATION = "क्वांटाइजेशन शुरू करने में त्रुटि: {0}"
        self.UPDATING_MODEL_INFO = "मॉडल जानकारी अपडेट हो रही है: {0}"
        self.TASK_FINISHED = "कार्य समाप्त: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "कार्य विवरण दिखाए जा रहे हैं: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = (
            "IMatrix डेटा फ़ाइल के लिए ब्राउज़ किया जा रहा है"
        )
        self.SELECT_DATA_FILE = "डेटा फ़ाइल चुनें"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = (
            "IMatrix मॉडल फ़ाइल के लिए ब्राउज़ किया जा रहा है"
        )
        self.SELECT_MODEL_FILE = "मॉडल फ़ाइल चुनें"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = (
            "IMatrix आउटपुट फ़ाइल के लिए ब्राउज़ किया जा रहा है"
        )
        self.SELECT_OUTPUT_FILE = "आउटपुट फ़ाइल चुनें"
        self.STARTING_IMATRIX_GENERATION = "IMatrix उत्पादन शुरू हो रहा है"
        self.BACKEND_PATH_NOT_EXIST = "बैकएंड पथ मौजूद नहीं है: {0}"
        self.GENERATING_IMATRIX = "IMatrix उत्पन्न किया जा रहा है"
        self.ERROR_STARTING_IMATRIX_GENERATION = "IMatrix उत्पादन शुरू करने में त्रुटि: {0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix उत्पादन कार्य शुरू हुआ"
        self.ERROR_MESSAGE = "त्रुटि: {0}"
        self.TASK_ERROR = "कार्य त्रुटि: {0}"
        self.APPLICATION_CLOSING = "एप्लिकेशन बंद हो रहा है"
        self.APPLICATION_CLOSED = "एप्लिकेशन बंद हो गया"
        self.SELECT_QUANTIZATION_TYPE = "क्वांटाइजेशन प्रकार चुनें"
        self.ALLOWS_REQUANTIZING = (
            "पहले से क्वांटाइज़ किए गए टेंसर को पुनः क्वांटाइज़ करने की अनुमति देता है"
        )
        self.LEAVE_OUTPUT_WEIGHT = "output.weight को अक्वांटाइज़ (या पुनः क्वांटाइज़) छोड़ देगा"
        self.DISABLE_K_QUANT_MIXTURES = (
            "k-quant मिश्रण को अक्षम करें और सभी टेंसर को एक ही प्रकार में क्वांटाइज़ करें"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "क्वांट अनुकूलन के लिए फ़ाइल में डेटा को महत्व मैट्रिक्स के रूप में उपयोग करें"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = "इन टेंसर के लिए महत्व मैट्रिक्स का उपयोग करें"
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "इन टेंसर के लिए महत्व मैट्रिक्स का उपयोग न करें"
        )
        self.OUTPUT_TENSOR_TYPE = "आउटपुट टेंसर प्रकार:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "output.weight टेंसर के लिए इस प्रकार का उपयोग करें"
        )
        self.TOKEN_EMBEDDING_TYPE = "टोकन एम्बेडिंग प्रकार:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "टोकन एम्बेडिंग टेंसर के लिए इस प्रकार का उपयोग करें"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "इनपुट के समान शार्ड्स में क्वांटाइज़ किए गए मॉडल को उत्पन्न करेगा"
        )
        self.OVERRIDE_MODEL_METADATA = "मॉडल मेटाडेटा को ओवरराइड करें"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix उत्पादन के लिए इनपुट डेटा फ़ाइल"
        self.MODEL_TO_BE_QUANTIZED = "क्वांटाइज़ किए जाने वाला मॉडल"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "उत्पन्न IMatrix के लिए आउटपुट पथ"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "IMatrix को कितनी बार सहेजना है"
        self.SET_GPU_OFFLOAD_VALUE = "GPU ऑफलोड मान सेट करें (-ngl)"
        self.COMPLETED = "पूरा हुआ"
        self.REFRESH_MODELS = "मॉडल रीफ्रेश करें"


class _Russian(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (автоматический квантователь моделей GGUF)"
        self.RAM_USAGE = "Использование ОЗУ:"
        self.CPU_USAGE = "Использование ЦП:"
        self.BACKEND = "Бэкенд Llama.cpp:"
        self.REFRESH_BACKENDS = "Обновить бэкенды"
        self.MODELS_PATH = "Путь к моделям:"
        self.OUTPUT_PATH = "Путь вывода:"
        self.LOGS_PATH = "Путь к логам:"
        self.BROWSE = "Обзор"
        self.AVAILABLE_MODELS = "Доступные модели:"
        self.QUANTIZATION_TYPE = "Тип квантования:"
        self.ALLOW_REQUANTIZE = "Разрешить переквантование"
        self.LEAVE_OUTPUT_TENSOR = "Оставить выходной тензор"
        self.PURE = "Чистый"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Включить веса:"
        self.EXCLUDE_WEIGHTS = "Исключить веса:"
        self.USE_OUTPUT_TENSOR_TYPE = "Использовать тип выходного тензора"
        self.USE_TOKEN_EMBEDDING_TYPE = "Использовать тип встраивания токенов"
        self.KEEP_SPLIT = "Сохранить разделение"
        self.KV_OVERRIDES = "KV переопределения:"
        self.ADD_NEW_OVERRIDE = "Добавить новое переопределение"
        self.QUANTIZE_MODEL = "Квантовать модель"
        self.SAVE_PRESET = "Сохранить пресет"
        self.LOAD_PRESET = "Загрузить пресет"
        self.TASKS = "Задачи:"
        self.DOWNLOAD_LLAMACPP = "Скачать llama.cpp"
        self.SELECT_RELEASE = "Выбрать релиз:"
        self.SELECT_ASSET = "Выбрать актив:"
        self.EXTRACT_CUDA_FILES = "Извлечь файлы CUDA"
        self.SELECT_CUDA_BACKEND = "Выбрать бэкенд CUDA:"
        self.DOWNLOAD = "Скачать"
        self.IMATRIX_GENERATION = "Генерация IMatrix"
        self.DATA_FILE = "Файл данных:"
        self.MODEL = "Модель:"
        self.OUTPUT = "Вывод:"
        self.OUTPUT_FREQUENCY = "Частота вывода:"
        self.GPU_OFFLOAD = "Разгрузка GPU:"
        self.AUTO = "Авто"
        self.GENERATE_IMATRIX = "Сгенерировать IMatrix"
        self.ERROR = "Ошибка"
        self.WARNING = "Предупреждение"
        self.PROPERTIES = "Свойства"
        self.CANCEL = "Отмена"
        self.RESTART = "Перезапуск"
        self.DELETE = "Удалить"
        self.CONFIRM_DELETION = "Вы уверены, что хотите удалить эту задачу?"
        self.TASK_RUNNING_WARNING = (
            "Некоторые задачи все еще выполняются. Вы уверены, что хотите выйти?"
        )
        self.YES = "Да"
        self.NO = "Нет"
        self.DOWNLOAD_COMPLETE = "Загрузка завершена"
        self.CUDA_EXTRACTION_FAILED = "Извлечение CUDA не удалось"
        self.PRESET_SAVED = "Пресет сохранен"
        self.PRESET_LOADED = "Пресет загружен"
        self.NO_ASSET_SELECTED = "Актив не выбран"
        self.DOWNLOAD_FAILED = "Загрузка не удалась"
        self.NO_BACKEND_SELECTED = "Бэкенд не выбран"
        self.NO_MODEL_SELECTED = "Модель не выбрана"
        self.REFRESH_RELEASES = "Обновить релизы"
        self.NO_SUITABLE_CUDA_BACKENDS = "Подходящие бэкенды CUDA не найдены"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Бинарный файл llama.cpp загружен и извлечен в {0}\nФайлы CUDA извлечены в {1}"
        self.CUDA_FILES_EXTRACTED = "Файлы CUDA извлечены в"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Подходящий бэкенд CUDA для извлечения не найден"
        )
        self.ERROR_FETCHING_RELEASES = "Ошибка получения релизов: {0}"
        self.CONFIRM_DELETION_TITLE = "Подтвердить удаление"
        self.LOG_FOR = "Лог для {0}"
        self.ALL_FILES = "Все файлы (*)"
        self.GGUF_FILES = "Файлы GGUF (*.gguf)"
        self.DAT_FILES = "Файлы DAT (*.dat)"
        self.JSON_FILES = "Файлы JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Не удалось загрузить пресет: {0}"
        self.INITIALIZING_AUTOGGUF = "Инициализация приложения AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Инициализация AutoGGUF завершена"
        self.REFRESHING_BACKENDS = "Обновление бэкендов"
        self.NO_BACKENDS_AVAILABLE = "Бэкенды недоступны"
        self.FOUND_VALID_BACKENDS = "Найдено {0} действительных бэкендов"
        self.SAVING_PRESET = "Сохранение пресета"
        self.PRESET_SAVED_TO = "Пресет сохранен в {0}"
        self.LOADING_PRESET = "Загрузка пресета"
        self.PRESET_LOADED_FROM = "Пресет загружен из {0}"
        self.ADDING_KV_OVERRIDE = "Добавление KV переопределения: {0}"
        self.SAVING_TASK_PRESET = "Сохранение пресета задачи для {0}"
        self.TASK_PRESET_SAVED = "Пресет задачи сохранен"
        self.TASK_PRESET_SAVED_TO = "Пресет задачи сохранен в {0}"
        self.RESTARTING_TASK = "Перезапуск задачи: {0}"
        self.IN_PROGRESS = "В процессе"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Загрузка завершена. Извлечено в: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Бинарный файл llama.cpp загружен и извлечен в {0}\nФайлы CUDA извлечены в {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Подходящий бэкенд CUDA для извлечения не найден"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Бинарный файл llama.cpp загружен и извлечен в {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Обновление релизов llama.cpp"
        self.UPDATING_ASSET_LIST = "Обновление списка активов"
        self.UPDATING_CUDA_OPTIONS = "Обновление параметров CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Начало загрузки llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Обновление бэкендов CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Бэкенд CUDA для извлечения не выбран"
        self.EXTRACTING_CUDA_FILES = "Извлечение файлов CUDA из {0} в {1}"
        self.DOWNLOAD_ERROR = "Ошибка загрузки: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Отображение контекстного меню задачи"
        self.SHOWING_PROPERTIES_FOR_TASK = "Отображение свойств задачи: {0}"
        self.CANCELLING_TASK = "Отмена задачи: {0}"
        self.CANCELED = "Отменено"
        self.DELETING_TASK = "Удаление задачи: {0}"
        self.LOADING_MODELS = "Загрузка моделей"
        self.LOADED_MODELS = "Загружено {0} моделей"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Поиск каталога моделей"
        self.SELECT_MODELS_DIRECTORY = "Выберите каталог моделей"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Поиск выходного каталога"
        self.SELECT_OUTPUT_DIRECTORY = "Выберите выходной каталог"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Поиск каталога логов"
        self.SELECT_LOGS_DIRECTORY = "Выберите каталог логов"
        self.BROWSING_FOR_IMATRIX_FILE = "Поиск файла IMatrix"
        self.SELECT_IMATRIX_FILE = "Выберите файл IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} МБ / {2} МБ)"
        self.CPU_USAGE_FORMAT = "Использование ЦП: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Проверка входных данных квантования"
        self.MODELS_PATH_REQUIRED = "Требуется путь к моделям"
        self.OUTPUT_PATH_REQUIRED = "Требуется путь вывода"
        self.LOGS_PATH_REQUIRED = "Требуется путь к логам"
        self.STARTING_MODEL_QUANTIZATION = "Начало квантования модели"
        self.INPUT_FILE_NOT_EXIST = "Входной файл '{0}' не существует."
        self.QUANTIZING_MODEL_TO = "Квантование {0} в {1}"
        self.QUANTIZATION_TASK_STARTED = "Задача квантования запущена для {0}"
        self.ERROR_STARTING_QUANTIZATION = "Ошибка запуска квантования: {0}"
        self.UPDATING_MODEL_INFO = "Обновление информации о модели: {0}"
        self.TASK_FINISHED = "Задача завершена: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Отображение сведений о задаче для: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Поиск файла данных IMatrix"
        self.SELECT_DATA_FILE = "Выберите файл данных"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Поиск файла модели IMatrix"
        self.SELECT_MODEL_FILE = "Выберите файл модели"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Поиск выходного файла IMatrix"
        self.SELECT_OUTPUT_FILE = "Выберите выходной файл"
        self.STARTING_IMATRIX_GENERATION = "Начало генерации IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Путь бэкенда не существует: {0}"
        self.GENERATING_IMATRIX = "Генерация IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = "Ошибка запуска генерации IMatrix: {0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "Задача генерации IMatrix запущена"
        self.ERROR_MESSAGE = "Ошибка: {0}"
        self.TASK_ERROR = "Ошибка задачи: {0}"
        self.APPLICATION_CLOSING = "Закрытие приложения"
        self.APPLICATION_CLOSED = "Приложение закрыто"
        self.SELECT_QUANTIZATION_TYPE = "Выберите тип квантования"
        self.ALLOWS_REQUANTIZING = (
            "Позволяет переквантовать тензоры, которые уже были квантованы"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Оставит output.weight не (пере)квантованным"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Отключить k-квантовые смеси и квантовать все тензоры к одному типу"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Использовать данные в файле как матрицу важности для оптимизации квантования"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Использовать матрицу важности для этих тензоров"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Не использовать матрицу важности для этих тензоров"
        )
        self.OUTPUT_TENSOR_TYPE = "Тип выходного тензора:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Использовать этот тип для тензора output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Тип встраивания токенов:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Использовать этот тип для тензора встраивания токенов"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = "Будет генерировать квантованную модель в тех же шардах, что и входные данные"
        self.OVERRIDE_MODEL_METADATA = "Переопределить метаданные модели"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Входной файл данных для генерации IMatrix"
        self.MODEL_TO_BE_QUANTIZED = "Модель для квантования"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Выходной путь для сгенерированного IMatrix"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Как часто сохранять IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Установить значение разгрузки GPU (-ngl)"
        self.COMPLETED = "Завершено"
        self.REFRESH_MODELS = "Обновить модели"


class _Ukrainian(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (автоматичний квантувальник моделей GGUF)"
        self.RAM_USAGE = "Використання ОЗУ:"
        self.CPU_USAGE = "Використання ЦП:"
        self.BACKEND = "Бекенд Llama.cpp:"
        self.REFRESH_BACKENDS = "Оновити бекенди"
        self.MODELS_PATH = "Шлях до моделей:"
        self.OUTPUT_PATH = "Шлях виводу:"
        self.LOGS_PATH = "Шлях до логів:"
        self.BROWSE = "Огляд"
        self.AVAILABLE_MODELS = "Доступні моделі:"
        self.QUANTIZATION_TYPE = "Тип квантування:"
        self.ALLOW_REQUANTIZE = "Дозволити переквантування"
        self.LEAVE_OUTPUT_TENSOR = "Залишити вихідний тензор"
        self.PURE = "Чистий"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Включити ваги:"
        self.EXCLUDE_WEIGHTS = "Виключити ваги:"
        self.USE_OUTPUT_TENSOR_TYPE = "Використовувати тип вихідного тензора"
        self.USE_TOKEN_EMBEDDING_TYPE = "Використовувати тип вбудовування токенів"
        self.KEEP_SPLIT = "Зберегти розділення"
        self.KV_OVERRIDES = "KV перевизначення:"
        self.ADD_NEW_OVERRIDE = "Додати нове перевизначення"
        self.QUANTIZE_MODEL = "Квантувати модель"
        self.SAVE_PRESET = "Зберегти пресет"
        self.LOAD_PRESET = "Завантажити пресет"
        self.TASKS = "Завдання:"
        self.DOWNLOAD_LLAMACPP = "Завантажити llama.cpp"
        self.SELECT_RELEASE = "Вибрати реліз:"
        self.SELECT_ASSET = "Вибрати актив:"
        self.EXTRACT_CUDA_FILES = "Витягнути файли CUDA"
        self.SELECT_CUDA_BACKEND = "Вибрати бекенд CUDA:"
        self.DOWNLOAD = "Завантажити"
        self.IMATRIX_GENERATION = "Генерація IMatrix"
        self.DATA_FILE = "Файл даних:"
        self.MODEL = "Модель:"
        self.OUTPUT = "Вивід:"
        self.OUTPUT_FREQUENCY = "Частота виводу:"
        self.GPU_OFFLOAD = "Розвантаження GPU:"
        self.AUTO = "Авто"
        self.GENERATE_IMATRIX = "Згенерувати IMatrix"
        self.ERROR = "Помилка"
        self.WARNING = "Попередження"
        self.PROPERTIES = "Властивості"
        self.CANCEL = "Скасувати"
        self.RESTART = "Перезапустити"
        self.DELETE = "Видалити"
        self.CONFIRM_DELETION = "Ви впевнені, що хочете видалити це завдання?"
        self.TASK_RUNNING_WARNING = (
            "Деякі завдання все ще виконуються. Ви впевнені, що хочете вийти?"
        )
        self.YES = "Так"
        self.NO = "Ні"
        self.DOWNLOAD_COMPLETE = "Завантаження завершено"
        self.CUDA_EXTRACTION_FAILED = "Витягнення CUDA не вдалося"
        self.PRESET_SAVED = "Пресет збережено"
        self.PRESET_LOADED = "Пресет завантажено"
        self.NO_ASSET_SELECTED = "Актив не вибрано"
        self.DOWNLOAD_FAILED = "Завантаження не вдалося"
        self.NO_BACKEND_SELECTED = "Бекенд не вибрано"
        self.NO_MODEL_SELECTED = "Модель не вибрано"
        self.REFRESH_RELEASES = "Оновити релізи"
        self.NO_SUITABLE_CUDA_BACKENDS = "Підходящі бекенди CUDA не знайдено"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Бінарний файл llama.cpp завантажено та витягнуто в {0}\nФайли CUDA витягнуто в {1}"
        self.CUDA_FILES_EXTRACTED = "Файли CUDA витягнуто в"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Підходящий бекенд CUDA для витягнення не знайдено"
        )
        self.ERROR_FETCHING_RELEASES = "Помилка отримання релізів: {0}"
        self.CONFIRM_DELETION_TITLE = "Підтвердити видалення"
        self.LOG_FOR = "Лог для {0}"
        self.ALL_FILES = "Всі файли (*)"
        self.GGUF_FILES = "Файли GGUF (*.gguf)"
        self.DAT_FILES = "Файли DAT (*.dat)"
        self.JSON_FILES = "Файли JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Не вдалося завантажити пресет: {0}"
        self.INITIALIZING_AUTOGGUF = "Ініціалізація програми AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Ініціалізація AutoGGUF завершена"
        self.REFRESHING_BACKENDS = "Оновлення бекендів"
        self.NO_BACKENDS_AVAILABLE = "Бекенди недоступні"
        self.FOUND_VALID_BACKENDS = "Знайдено {0} дійсних бекендів"
        self.SAVING_PRESET = "Збереження пресета"
        self.PRESET_SAVED_TO = "Пресет збережено в {0}"
        self.LOADING_PRESET = "Завантаження пресета"
        self.PRESET_LOADED_FROM = "Пресет завантажено з {0}"
        self.ADDING_KV_OVERRIDE = "Додавання KV перевизначення: {0}"
        self.SAVING_TASK_PRESET = "Збереження пресета завдання для {0}"
        self.TASK_PRESET_SAVED = "Пресет завдання збережено"
        self.TASK_PRESET_SAVED_TO = "Пресет завдання збережено в {0}"
        self.RESTARTING_TASK = "Перезапуск завдання: {0}"
        self.IN_PROGRESS = "В процесі"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Завантаження завершено. Витягнуто в: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Бінарний файл llama.cpp завантажено та витягнуто в {0}\nФайли CUDA витягнуто в {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Підходящий бекенд CUDA для витягнення не знайдено"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Бінарний файл llama.cpp завантажено та витягнуто в {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Оновлення релізів llama.cpp"
        self.UPDATING_ASSET_LIST = "Оновлення списку активів"
        self.UPDATING_CUDA_OPTIONS = "Оновлення параметрів CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Початок завантаження llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Оновлення бекендів CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Бекенд CUDA для витягнення не вибрано"
        self.EXTRACTING_CUDA_FILES = "Витягнення файлів CUDA з {0} в {1}"
        self.DOWNLOAD_ERROR = "Помилка завантаження: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Відображення контекстного меню завдання"
        self.SHOWING_PROPERTIES_FOR_TASK = "Відображення властивостей завдання: {0}"
        self.CANCELLING_TASK = "Скасування завдання: {0}"
        self.CANCELED = "Скасовано"
        self.DELETING_TASK = "Видалення завдання: {0}"
        self.LOADING_MODELS = "Завантаження моделей"
        self.LOADED_MODELS = "Завантажено {0} моделей"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Пошук каталогу моделей"
        self.SELECT_MODELS_DIRECTORY = "Виберіть каталог моделей"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Пошук вихідного каталогу"
        self.SELECT_OUTPUT_DIRECTORY = "Виберіть вихідний каталог"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Пошук каталогу логів"
        self.SELECT_LOGS_DIRECTORY = "Виберіть каталог логів"
        self.BROWSING_FOR_IMATRIX_FILE = "Пошук файлу IMatrix"
        self.SELECT_IMATRIX_FILE = "Виберіть файл IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} МБ / {2} МБ)"
        self.CPU_USAGE_FORMAT = "Використання ЦП: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Перевірка вхідних даних квантування"
        self.MODELS_PATH_REQUIRED = "Потрібен шлях до моделей"
        self.OUTPUT_PATH_REQUIRED = "Потрібен шлях виводу"
        self.LOGS_PATH_REQUIRED = "Потрібен шлях до логів"
        self.STARTING_MODEL_QUANTIZATION = "Початок квантування моделі"
        self.INPUT_FILE_NOT_EXIST = "Вхідний файл '{0}' не існує."
        self.QUANTIZING_MODEL_TO = "Квантування {0} в {1}"
        self.QUANTIZATION_TASK_STARTED = "Завдання квантування запущено для {0}"
        self.ERROR_STARTING_QUANTIZATION = "Помилка запуску квантування: {0}"
        self.UPDATING_MODEL_INFO = "Оновлення інформації про модель: {0}"
        self.TASK_FINISHED = "Завдання завершено: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Відображення відомостей про завдання для: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Пошук файлу даних IMatrix"
        self.SELECT_DATA_FILE = "Виберіть файл даних"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Пошук файлу моделі IMatrix"
        self.SELECT_MODEL_FILE = "Виберіть файл моделі"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Пошук вихідного файлу IMatrix"
        self.SELECT_OUTPUT_FILE = "Виберіть вихідний файл"
        self.STARTING_IMATRIX_GENERATION = "Початок генерації IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Шлях бекенда не існує: {0}"
        self.GENERATING_IMATRIX = "Генерація IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Помилка запуску генерації IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Завдання генерації IMatrix запущено"
        self.ERROR_MESSAGE = "Помилка: {0}"
        self.TASK_ERROR = "Помилка завдання: {0}"
        self.APPLICATION_CLOSING = "Закриття програми"
        self.APPLICATION_CLOSED = "Програма закрита"
        self.SELECT_QUANTIZATION_TYPE = "Виберіть тип квантування"
        self.ALLOWS_REQUANTIZING = (
            "Дозволяє переквантувати тензори, які вже були квантовані"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Залишить output.weight не (пере)квантованим"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Вимкнути k-квантові суміші та квантувати всі тензори до одного типу"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Використовувати дані у файлі як матрицю важливості для оптимізації квантування"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Використовувати матрицю важливості для цих тензорів"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Не використовувати матрицю важливості для цих тензорів"
        )
        self.OUTPUT_TENSOR_TYPE = "Тип вихідного тензора:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Використовувати цей тип для тензора output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Тип вбудовування токенів:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Використовувати цей тип для тензора вбудовування токенів"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Генеруватиме квантовану модель у тих самих шардах, що й вхідні дані"
        )
        self.OVERRIDE_MODEL_METADATA = "Перевизначити метадані моделі"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Вхідний файл даних для генерації IMatrix"
        self.MODEL_TO_BE_QUANTIZED = "Модель для квантування"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Вихідний шлях для згенерованого IMatrix"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Як часто зберігати IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Встановити значення розвантаження GPU (-ngl)"
        self.COMPLETED = "Завершено"
        self.REFRESH_MODELS = "Оновити моделі"


class _Japanese(_Localization):
    def __init__(self):
        super().__init__()

        # General UI
        self.WINDOW_TITLE = "AutoGGUF (自動GGUFモデル量子化ツール)"
        self.RAM_USAGE = "RAM使用量:"
        self.CPU_USAGE = "CPU使用量:"
        self.BACKEND = "Llama.cpp バックエンド:"
        self.REFRESH_BACKENDS = "バックエンドを更新"
        self.MODELS_PATH = "モデルパス:"
        self.OUTPUT_PATH = "出力パス:"
        self.LOGS_PATH = "ログパス:"
        self.BROWSE = "参照"
        self.AVAILABLE_MODELS = "利用可能なモデル:"
        self.REFRESH_MODELS = "モデルを更新"
        self.STARTUP_ELASPED_TIME = "初期化に{0}ミリ秒かかりました"

        # Usage Graphs
        self.CPU_USAGE_OVER_TIME = "時間経過によるCPU使用量"
        self.RAM_USAGE_OVER_TIME = "時間経過によるRAM使用量"

        # Environment variables
        self.DOTENV_FILE_NOT_FOUND = ".envファイルが見つかりません。"
        self.COULD_NOT_PARSE_LINE = "行を解析できませんでした: {0}"
        self.ERROR_LOADING_DOTENV = ".envの読み込みエラー: {0}"

        # Model Import
        self.IMPORT_MODEL = "モデルをインポート"
        self.SELECT_MODEL_TO_IMPORT = "インポートするモデルを選択"
        self.CONFIRM_IMPORT = "インポートの確認"
        self.IMPORT_MODEL_CONFIRMATION = "モデル{}をインポートしますか？"
        self.MODEL_IMPORTED_SUCCESSFULLY = "モデル{}が正常にインポートされました"
        self.IMPORTING_MODEL = "モデルをインポート中"
        self.IMPORTED_MODEL_TOOLTIP = "インポートされたモデル: {}"

        # AutoFP8 Quantization
        self.AUTOFP8_QUANTIZATION_TASK_STARTED = "AutoFP8量子化タスクが開始されました"
        self.ERROR_STARTING_AUTOFP8_QUANTIZATION = "AutoFP8量子化の開始エラー"
        self.QUANTIZING_WITH_AUTOFP8 = "{0}をAutoFP8で量子化中"
        self.QUANTIZING_TO_WITH_AUTOFP8 = "{0}を{1}にAutoFP8で量子化中"
        self.QUANTIZE_TO_FP8_DYNAMIC = "FP8 Dynamicに量子化"
        self.OPEN_MODEL_FOLDER = "モデルフォルダを開く"
        self.QUANTIZE = "量子化"
        self.OPEN_MODEL_FOLDER = "モデルフォルダを開く"
        self.INPUT_MODEL = "入力モデル:"

        # GGUF Verification
        self.INVALID_GGUF_FILE = "無効なGGUFファイル: {}"
        self.SHARDED_MODEL_NAME = "{} (シャード)"
        self.IMPORTED_MODEL_TOOLTIP = "インポートされたモデル: {}"
        self.CONCATENATED_FILE_WARNING = "これは連結されたファイル部分です。llama-quantizeでは動作しません。先にファイルを連結してください。"
        self.CONCATENATED_FILES_FOUND = "{}個の連結されたファイル部分が見つかりました。先にファイルを連結してください。"

        # Plugins
        self.PLUGINS_DIR_NOT_EXIST = (
            "プラグインディレクトリ '{}'が存在しません。プラグインはロードされません。"
        )
        self.PLUGINS_DIR_NOT_DIRECTORY = "'{}'は存在しますが、ディレクトリではありません。プラグインはロードされません。"
        self.PLUGIN_LOADED = "プラグインをロードしました: {} {}"
        self.PLUGIN_INCOMPATIBLE = "プラグイン {} {} はAutoGGUFバージョン {} と互換性がありません。サポートされているバージョン: {}"
        self.PLUGIN_LOAD_FAILED = "プラグイン {} のロードに失敗しました: {}"
        self.NO_PLUGINS_LOADED = "プラグインがロードされていません。"

        # GPU Monitoring
        self.GPU_USAGE = "GPU使用量:"
        self.GPU_USAGE_FORMAT = "GPU: {:.1f}% | VRAM: {:.1f}% ({} MB / {} MB)"
        self.GPU_DETAILS = "GPU詳細"
        self.GPU_USAGE_OVER_TIME = "時間経過によるGPU使用量"
        self.VRAM_USAGE_OVER_TIME = "時間経過によるVRAM使用量"
        self.PERCENTAGE = "パーセンテージ"
        self.TIME = "時間 (秒)"
        self.NO_GPU_DETECTED = "GPUが検出されません"
        self.SELECT_GPU = "GPUを選択"
        self.AMD_GPU_NOT_SUPPORTED = "AMD GPUが検出されましたが、サポートされていません"

        # Quantization
        self.QUANTIZATION_TYPE = "量子化タイプ:"
        self.ALLOW_REQUANTIZE = "再量子化を許可"
        self.LEAVE_OUTPUT_TENSOR = "出力テンソルを残す"
        self.PURE = "純粋"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "重みを含む:"
        self.EXCLUDE_WEIGHTS = "重みを除外:"
        self.USE_OUTPUT_TENSOR_TYPE = "出力テンソルタイプを使用"
        self.USE_TOKEN_EMBEDDING_TYPE = "トークン埋め込みタイプを使用"
        self.KEEP_SPLIT = "分割を維持"
        self.KV_OVERRIDES = "KVオーバーライド:"
        self.ADD_NEW_OVERRIDE = "新しいオーバーライドを追加"
        self.QUANTIZE_MODEL = "モデルを量子化"
        self.EXTRA_ARGUMENTS = "追加引数:"
        self.EXTRA_ARGUMENTS_LABEL = "追加のコマンドライン引数"
        self.QUANTIZATION_COMMAND = "量子化コマンド"

        # Presets
        self.SAVE_PRESET = "プリセットを保存"
        self.LOAD_PRESET = "プリセットを読み込み"

        # Tasks
        self.TASKS = "タスク:"

        # llama.cpp Download
        self.DOWNLOAD_LLAMACPP = "llama.cppをダウンロード"
        self.SELECT_RELEASE = "リリースを選択:"
        self.SELECT_ASSET = "アセットを選択:"
        self.EXTRACT_CUDA_FILES = "CUDAファイルを抽出"
        self.SELECT_CUDA_BACKEND = "CUDAバックエンドを選択:"
        self.DOWNLOAD = "ダウンロード"
        self.REFRESH_RELEASES = "リリースを更新"

        # IMatrix Generation
        self.IMATRIX_GENERATION = "IMatrix生成"
        self.DATA_FILE = "データファイル:"
        self.MODEL = "モデル:"
        self.OUTPUT = "出力:"
        self.OUTPUT_FREQUENCY = "出力頻度:"
        self.GPU_OFFLOAD = "GPUオフロード:"
        self.AUTO = "自動"
        self.GENERATE_IMATRIX = "IMatrixを生成"
        self.CONTEXT_SIZE = "コンテキストサイズ:"
        self.CONTEXT_SIZE_FOR_IMATRIX = "IMatrix生成のコンテキストサイズ"
        self.THREADS = "スレッド数:"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = "IMatrix生成のスレッド数"
        self.IMATRIX_GENERATION_COMMAND = "IMatrix生成コマンド"

        # LoRA Conversion
        self.LORA_CONVERSION = "LoRA変換"
        self.LORA_INPUT_PATH = "LoRA入力パス"
        self.LORA_OUTPUT_PATH = "LoRA出力パス"
        self.SELECT_LORA_INPUT_DIRECTORY = "LoRA入力ディレクトリを選択"
        self.SELECT_LORA_OUTPUT_FILE = "LoRA出力ファイルを選択"
        self.CONVERT_LORA = "LoRAを変換"
        self.LORA_CONVERSION_COMMAND = "LoRA変換コマンド"

        # LoRA Export
        self.EXPORT_LORA = "LoRAをエクスポート"
        self.GGML_LORA_ADAPTERS = "GGML LoRAアダプター"
        self.SELECT_LORA_ADAPTER_FILES = "LoRAアダプターファイルを選択"
        self.ADD_ADAPTER = "アダプターを追加"
        self.DELETE_ADAPTER = "削除"
        self.LORA_SCALE = "LoRAスケール"
        self.ENTER_LORA_SCALE_VALUE = "LoRAスケール値を入力 (オプション)"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = "LoRAエクスポートのスレッド数"
        self.LORA_EXPORT_COMMAND = "LoRAエクスポートコマンド"

        # HuggingFace to GGUF Conversion
        self.HF_TO_GGUF_CONVERSION = "HuggingFaceからGGUFへの変換"
        self.MODEL_DIRECTORY = "モデルディレクトリ:"
        self.OUTPUT_FILE = "出力ファイル:"
        self.OUTPUT_TYPE = "出力タイプ:"
        self.VOCAB_ONLY = "語彙のみ"
        self.USE_TEMP_FILE = "一時ファイルを使用"
        self.NO_LAZY_EVALUATION = "遅延評価なし"
        self.MODEL_NAME = "モデル名:"
        self.VERBOSE = "詳細"
        self.SPLIT_MAX_SIZE = "分割最大サイズ:"
        self.DRY_RUN = "ドライラン"
        self.CONVERT_HF_TO_GGUF = "HFをGGUFに変換"
        self.SELECT_HF_MODEL_DIRECTORY = "HuggingFaceモデルディレクトリを選択"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = "HuggingFaceモデルディレクトリを参照"
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = "HuggingFaceからGGUFへの出力ファイルを参照"

        # Update Checking
        self.UPDATE_AVAILABLE = "アップデートが利用可能"
        self.NEW_VERSION_AVAILABLE = "新しいバージョンが利用可能です: {}"
        self.DOWNLOAD_NEW_VERSION = "ダウンロードしますか？"
        self.ERROR_CHECKING_FOR_UPDATES = "アップデートの確認中にエラーが発生しました:"
        self.CHECKING_FOR_UPDATES = "アップデートを確認中"

        # General Messages
        self.ERROR = "エラー"
        self.WARNING = "警告"
        self.PROPERTIES = "プロパティ"
        self.CANCEL = "キャンセル"
        self.RESTART = "再起動"
        self.DELETE = "削除"
        self.RENAME = "名前変更"
        self.CONFIRM_DELETION = "このタスクを削除してもよろしいですか？"
        self.TASK_RUNNING_WARNING = "タスクがまだ実行中です。終了してもよろしいですか？"
        self.YES = "はい"
        self.NO = "いいえ"
        self.COMPLETED = "完了"

        # File Types
        self.ALL_FILES = "すべてのファイル (*)"
        self.GGUF_FILES = "GGUFファイル (*.gguf)"
        self.DAT_FILES = "DATファイル (*.dat)"
        self.JSON_FILES = "JSONファイル (*.json)"
        self.BIN_FILES = "バイナリファイル (*.bin)"
        self.LORA_FILES = "LoRAファイル (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "GGUFおよびバイナリファイル (*.gguf *.bin)"
        self.SHARDED = "シャード"

        # Status Messages
        self.DOWNLOAD_COMPLETE = "ダウンロード完了"
        self.CUDA_EXTRACTION_FAILED = "CUDA抽出失敗"
        self.PRESET_SAVED = "プリセットが保存されました"
        self.PRESET_LOADED = "プリセットが読み込まれました"
        self.NO_ASSET_SELECTED = "アセットが選択されていません"
        self.DOWNLOAD_FAILED = "ダウンロード失敗"
        self.NO_BACKEND_SELECTED = "バックエンドが選択されていません"
        self.NO_MODEL_SELECTED = "モデルが選択されていません"
        self.NO_SUITABLE_CUDA_BACKENDS = "適切なCUDAバックエンドが見つかりません"
        self.IN_PROGRESS = "進行中"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "llama.cppバイナリがダウンロードされ、{0}に抽出されました"
        )
        self.CUDA_FILES_EXTRACTED = "CUDAファイルが抽出されました:"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "抽出に適したCUDAバックエンドが見つかりません"
        )
        self.ERROR_FETCHING_RELEASES = "リリースの取得中にエラーが発生しました: {0}"
        self.CONFIRM_DELETION_TITLE = "削除の確認"
        self.LOG_FOR = "{0}のログ"
        self.FAILED_TO_LOAD_PRESET = "プリセットの読み込みに失敗しました: {0}"
        self.INITIALIZING_AUTOGGUF = "AutoGGUFアプリケーションを初期化中"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF初期化完了"
        self.REFRESHING_BACKENDS = "バックエンドを更新中"
        self.NO_BACKENDS_AVAILABLE = "利用可能なバックエンドがありません"
        self.FOUND_VALID_BACKENDS = "{0}個の有効なバックエンドが見つかりました"
        self.SAVING_PRESET = "プリセットを保存中"
        self.PRESET_SAVED_TO = "プリセットが{0}に保存されました"
        self.LOADING_PRESET = "プリセットを読み込み中"
        self.PRESET_LOADED_FROM = "プリセットが{0}から読み込まれました"
        self.ADDING_KV_OVERRIDE = "KVオーバーライドを追加中: {0}"
        self.SAVING_TASK_PRESET = "{0}のタスクプリセットを保存中"
        self.TASK_PRESET_SAVED = "タスクプリセットが保存されました"
        self.TASK_PRESET_SAVED_TO = "タスクプリセットが{0}に保存されました"
        self.RESTARTING_TASK = "タスクを再起動中: {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "ダウンロードが完了しました。抽出先: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "llama.cppバイナリがダウンロードされ、{0}に抽出されました"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "抽出に適したCUDAバックエンドが見つかりません"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cppバイナリがダウンロードされ、{0}に抽出されました"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "llama.cppリリースを更新中"
        self.UPDATING_ASSET_LIST = "アセットリストを更新中"
        self.UPDATING_CUDA_OPTIONS = "CUDAオプションを更新中"
        self.STARTING_LLAMACPP_DOWNLOAD = "llama.cppのダウンロードを開始中"
        self.UPDATING_CUDA_BACKENDS = "CUDAバックエンドを更新中"
        self.NO_CUDA_BACKEND_SELECTED = "抽出用のCUDAバックエンドが選択されていません"
        self.EXTRACTING_CUDA_FILES = "{0}からCUDAファイルを{1}に抽出中"
        self.DOWNLOAD_ERROR = "ダウンロードエラー: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "タスクコンテキストメニューを表示中"
        self.SHOWING_PROPERTIES_FOR_TASK = "タスクのプロパティを表示中: {0}"
        self.CANCELLING_TASK = "タスクをキャンセル中: {0}"
        self.CANCELED = "キャンセルされました"
        self.DELETING_TASK = "タスクを削除中: {0}"
        self.LOADING_MODELS = "モデルを読み込み中"
        self.LOADED_MODELS = "{0}個のモデルが読み込まれました"
        self.BROWSING_FOR_MODELS_DIRECTORY = "モデルディレクトリを参照中"
        self.SELECT_MODELS_DIRECTORY = "モデルディレクトリを選択"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "出力ディレクトリを参照中"
        self.SELECT_OUTPUT_DIRECTORY = "出力ディレクトリを選択"
        self.BROWSING_FOR_LOGS_DIRECTORY = "ログディレクトリを参照中"
        self.SELECT_LOGS_DIRECTORY = "ログディレクトリを選択"
        self.BROWSING_FOR_IMATRIX_FILE = "IMatrixファイルを参照中"
        self.SELECT_IMATRIX_FILE = "IMatrixファイルを選択"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU使用量: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "量子化入力を検証中"
        self.MODELS_PATH_REQUIRED = "モデルパスが必要です"
        self.OUTPUT_PATH_REQUIRED = "出力パスが必要です"
        self.LOGS_PATH_REQUIRED = "ログパスが必要です"
        self.STARTING_MODEL_QUANTIZATION = "モデル量子化を開始中"
        self.INPUT_FILE_NOT_EXIST = "入力ファイル '{0}' が存在しません。"
        self.QUANTIZING_MODEL_TO = "{0}を{1}に量子化中"
        self.QUANTIZATION_TASK_STARTED = "{0}の量子化タスクが開始されました"
        self.ERROR_STARTING_QUANTIZATION = "量子化の開始エラー: {0}"
        self.UPDATING_MODEL_INFO = "モデル情報を更新中: {0}"
        self.TASK_FINISHED = "タスクが完了しました: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "タスクの詳細を表示中: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "IMatrixデータファイルを参照中"
        self.SELECT_DATA_FILE = "データファイルを選択"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "IMatrixモデルファイルを参照中"
        self.SELECT_MODEL_FILE = "モデルファイルを選択"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "IMatrix出力ファイルを参照中"
        self.SELECT_OUTPUT_FILE = "出力ファイルを選択"
        self.STARTING_IMATRIX_GENERATION = "IMatrix生成を開始中"
        self.BACKEND_PATH_NOT_EXIST = "バックエンドパスが存在しません: {0}"
        self.GENERATING_IMATRIX = "IMatrixを生成中"
        self.ERROR_STARTING_IMATRIX_GENERATION = "IMatrix生成の開始エラー: {0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix生成タスクが開始されました"
        self.ERROR_MESSAGE = "エラー: {0}"
        self.TASK_ERROR = "タスクエラー: {0}"
        self.APPLICATION_CLOSING = "アプリケーションを終了中"
        self.APPLICATION_CLOSED = "アプリケーションが終了しました"
        self.SELECT_QUANTIZATION_TYPE = "量子化タイプを選択してください"
        self.ALLOWS_REQUANTIZING = "すでに量子化されたテンソルの再量子化を許可します"
        self.LEAVE_OUTPUT_WEIGHT = "output.weightを量子化（再量子化）せずに残します"
        self.DISABLE_K_QUANT_MIXTURES = (
            "k-quant混合を無効にし、すべてのテンソルを同じタイプに量子化します"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "ファイル内のデータを量子化最適化の重要度行列として使用します"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "これらのテンソルに重要度行列を使用します"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "これらのテンソルに重要度行列を使用しません"
        )
        self.OUTPUT_TENSOR_TYPE = "出力テンソルタイプ:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "output.weightテンソルにこのタイプを使用します"
        )
        self.TOKEN_EMBEDDING_TYPE = "トークン埋め込みタイプ:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "トークン埋め込みテンソルにこのタイプを使用します"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "入力と同じシャードで量子化モデルを生成します"
        )
        self.OVERRIDE_MODEL_METADATA = "モデルメタデータをオーバーライドします"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix生成の入力データファイル"
        self.MODEL_TO_BE_QUANTIZED = "量子化するモデル"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "生成されたIMatrixの出力パス"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "IMatrixを保存する頻度"
        self.SET_GPU_OFFLOAD_VALUE = "GPUオフロード値を設定 (-ngl)"
        self.STARTING_LORA_CONVERSION = "LoRA変換を開始中"
        self.LORA_INPUT_PATH_REQUIRED = "LoRA入力パスが必要です。"
        self.LORA_OUTPUT_PATH_REQUIRED = "LoRA出力パスが必要です。"
        self.ERROR_STARTING_LORA_CONVERSION = "LoRA変換の開始エラー: {}"
        self.LORA_CONVERSION_TASK_STARTED = "LoRA変換タスクが開始されました。"
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = "LoRA入力ディレクトリを参照中..."
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "LoRA出力ファイルを参照中..."
        self.CONVERTING_LORA = "LoRA変換"
        self.LORA_CONVERSION_FINISHED = "LoRA変換が完了しました。"
        self.LORA_FILE_MOVED = "LoRAファイルが{}から{}に移動されました。"
        self.LORA_FILE_NOT_FOUND = "LoRAファイルが見つかりません: {}。"
        self.ERROR_MOVING_LORA_FILE = "LoRAファイルの移動エラー: {}"
        self.MODEL_PATH_REQUIRED = "モデルパスが必要です。"
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = (
            "少なくとも1つのLoRAアダプターが必要です。"
        )
        self.INVALID_LORA_SCALE_VALUE = "無効なLoRAスケール値です。"
        self.ERROR_STARTING_LORA_EXPORT = "LoRAエクスポートの開始エラー: {}"
        self.LORA_EXPORT_TASK_STARTED = "LoRAエクスポートタスクが開始されました。"
        self.EXPORTING_LORA = "LoRAをエクスポート中..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = (
            "エクスポートLoRAモデルファイルを参照中..."
        )
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = (
            "エクスポートLoRA出力ファイルを参照中..."
        )
        self.ADDING_LORA_ADAPTER = "LoRAアダプターを追加中..."
        self.DELETING_LORA_ADAPTER = "LoRAアダプターを削除中..."
        self.SELECT_LORA_ADAPTER_FILE = "LoRAアダプターファイルを選択"
        self.STARTING_LORA_EXPORT = "LoRAエクスポートを開始中..."
        self.SELECT_OUTPUT_TYPE = "出力タイプを選択 (GGUFまたはGGML)"
        self.BASE_MODEL = "ベースモデル"
        self.SELECT_BASE_MODEL_FILE = "ベースモデルファイルを選択 (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = "GGUF出力にはベースモデルパスが必要です。"
        self.BROWSING_FOR_BASE_MODEL_FILE = "ベースモデルファイルを参照中..."
        self.SELECT_BASE_MODEL_FOLDER = "ベースモデルフォルダを選択 (safetensorsを含む)"
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "ベースモデルフォルダを参照中..."
        self.LORA_CONVERSION_FROM_TO = "{}から{}へのLoRA変換"
        self.GENERATING_IMATRIX_FOR = "{}のIMatrixを生成中"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = "IMatrix生成にはモデルパスが必要です。"
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = (
            "CUDA確認用のアセットが選択されていません"
        )
        self.NO_QUANTIZATION_TYPE_SELECTED = "量子化タイプが選択されていません。少なくとも1つの量子化タイプを選択してください。"
        self.STARTING_HF_TO_GGUF_CONVERSION = "HuggingFaceからGGUFへの変換を開始中"
        self.MODEL_DIRECTORY_REQUIRED = "モデルディレクトリが必要です"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "HFからGGUFへの変換コマンド: {}"
        self.CONVERTING_TO_GGUF = "{}をGGUFに変換中"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "HuggingFaceからGGUFへの変換開始エラー: {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = (
            "HuggingFaceからGGUFへの変換タスクが開始されました"
        )

        # Split GGUF
        self.SPLIT_GGUF = "GGUFを分割"
        self.SPLIT_MAX_SIZE = "分割最大サイズ"
        self.SPLIT_MAX_TENSORS = "分割最大テンソル"
        self.SPLIT_GGUF_TASK_STARTED = "GGUF分割タスクが開始されました"
        self.SPLIT_GGUF_TASK_FINISHED = "GGUF分割タスクが完了しました"
        self.SPLIT_GGUF_COMMAND = "GGUF分割コマンド"
        self.SPLIT_GGUF_ERROR = "GGUF分割の開始エラー"
        self.NUMBER_OF_TENSORS = "テンソル数"
        self.SIZE_IN_UNITS = "サイズ（G/M）"

        # Model actions
        self.CONFIRM_DELETE = "削除の確認"
        self.DELETE_MODEL_WARNING = "モデル{}を削除してもよろしいですか？"
        self.MODEL_RENAMED_SUCCESSFULLY = "モデルの名前が正常に変更されました。"
        self.MODEL_DELETED_SUCCESSFULLY = "モデルが正常に削除されました。"

        # HuggingFace Transfer
        self.ALL_FIELDS_REQUIRED = "すべてのフィールドが必須です。"
        self.HUGGINGFACE_UPLOAD_COMMAND = "HuggingFaceアップロードコマンド: "
        self.UPLOADING = "アップロード中"
        self.UPLOADING_FOLDER = "フォルダをアップロード中"
        self.HF_TRANSFER_TASK_NAME = "{} {}を{}から{}"
        self.ERROR_STARTING_HF_TRANSFER = "HF転送の開始エラー: {}"
        self.STARTED_HUGGINGFACE_TRANSFER = "HuggingFace{}操作を開始しました。"
        self.SELECT_FOLDER = "フォルダを選択"
        self.SELECT_FILE = "ファイルを選択"


class _German(_Localization):
    def __init__(self):
        super().__init__()

        # General UI
        self.WINDOW_TITLE = "AutoGGUF (automatischer GGUF-Modell-Quantisierer)"
        self.RAM_USAGE = "RAM-Nutzung:"
        self.CPU_USAGE = "CPU-Nutzung:"
        self.BACKEND = "Llama.cpp Backend:"
        self.REFRESH_BACKENDS = "Backends aktualisieren"
        self.MODELS_PATH = "Modellpfad:"
        self.OUTPUT_PATH = "Ausgabepfad:"
        self.LOGS_PATH = "Logpfad:"
        self.BROWSE = "Durchsuchen"
        self.AVAILABLE_MODELS = "Verfügbare Modelle:"
        self.REFRESH_MODELS = "Modelle aktualisieren"
        self.STARTUP_ELASPED_TIME = "Initialisierung dauerte {0} ms"

        # Usage Graphs
        self.CPU_USAGE_OVER_TIME = "CPU-Nutzung über Zeit"
        self.RAM_USAGE_OVER_TIME = "RAM-Nutzung über Zeit"

        # Environment variables
        self.DOTENV_FILE_NOT_FOUND = ".env-Datei nicht gefunden."
        self.COULD_NOT_PARSE_LINE = "Zeile konnte nicht geparst werden: {0}"
        self.ERROR_LOADING_DOTENV = "Fehler beim Laden von .env: {0}"

        # Model Import
        self.IMPORT_MODEL = "Modell importieren"
        self.SELECT_MODEL_TO_IMPORT = "Zu importierendes Modell auswählen"
        self.CONFIRM_IMPORT = "Import bestätigen"
        self.IMPORT_MODEL_CONFIRMATION = "Möchten Sie das Modell {} importieren?"
        self.MODEL_IMPORTED_SUCCESSFULLY = "Modell {} erfolgreich importiert"
        self.IMPORTING_MODEL = "Modell wird importiert"
        self.IMPORTED_MODEL_TOOLTIP = "Importiertes Modell: {}"

        # AutoFP8 Quantization
        self.AUTOFP8_QUANTIZATION_TASK_STARTED = (
            "AutoFP8-Quantisierungsaufgabe gestartet"
        )
        self.ERROR_STARTING_AUTOFP8_QUANTIZATION = (
            "Fehler beim Starten der AutoFP8-Quantisierung"
        )
        self.QUANTIZING_WITH_AUTOFP8 = "Quantisiere {0} mit AutoFP8"
        self.QUANTIZING_TO_WITH_AUTOFP8 = "Quantisiere {0} zu {1}"
        self.QUANTIZE_TO_FP8_DYNAMIC = "Zu FP8 Dynamic quantisieren"
        self.OPEN_MODEL_FOLDER = "Modellordner öffnen"
        self.QUANTIZE = "Quantisieren"
        self.OPEN_MODEL_FOLDER = "Modellordner öffnen"
        self.INPUT_MODEL = "Eingabemodell:"

        # GGUF Verification
        self.INVALID_GGUF_FILE = "Ungültige GGUF-Datei: {}"
        self.SHARDED_MODEL_NAME = "{} (Geteilt)"
        self.IMPORTED_MODEL_TOOLTIP = "Importiertes Modell: {}"
        self.CONCATENATED_FILE_WARNING = "Dies ist ein verketteter Dateiteil. Es funktioniert nicht mit llama-quantize; bitte verketten Sie die Datei zuerst."
        self.CONCATENATED_FILES_FOUND = (
            "{} verkettete Dateiteile gefunden. Bitte verketten Sie die Dateien zuerst."
        )

        # Plugins
        self.PLUGINS_DIR_NOT_EXIST = (
            "Plugins-Verzeichnis '{}' existiert nicht. Es werden keine Plugins geladen."
        )
        self.PLUGINS_DIR_NOT_DIRECTORY = "'{}' existiert, ist aber kein Verzeichnis. Es werden keine Plugins geladen."
        self.PLUGIN_LOADED = "Plugin geladen: {} {}"
        self.PLUGIN_INCOMPATIBLE = "Plugin {} {} ist nicht kompatibel mit AutoGGUF Version {}. Unterstützte Versionen: {}"
        self.PLUGIN_LOAD_FAILED = "Fehler beim Laden des Plugins {}: {}"
        self.NO_PLUGINS_LOADED = "Keine Plugins geladen."

        # GPU Monitoring
        self.GPU_USAGE = "GPU-Nutzung:"
        self.GPU_USAGE_FORMAT = "GPU: {:.1f}% | VRAM: {:.1f}% ({} MB / {} MB)"
        self.GPU_DETAILS = "GPU-Details"
        self.GPU_USAGE_OVER_TIME = "GPU-Nutzung über Zeit"
        self.VRAM_USAGE_OVER_TIME = "VRAM-Nutzung über Zeit"
        self.PERCENTAGE = "Prozentsatz"
        self.TIME = "Zeit (s)"
        self.NO_GPU_DETECTED = "Keine GPU erkannt"
        self.SELECT_GPU = "GPU auswählen"
        self.AMD_GPU_NOT_SUPPORTED = "AMD GPU erkannt, aber nicht unterstützt"

        # Quantization
        self.QUANTIZATION_TYPE = "Quantisierungstyp:"
        self.ALLOW_REQUANTIZE = "Requantisierung erlauben"
        self.LEAVE_OUTPUT_TENSOR = "Ausgabetensor belassen"
        self.PURE = "Rein"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Gewichte einschließen:"
        self.EXCLUDE_WEIGHTS = "Gewichte ausschließen:"
        self.USE_OUTPUT_TENSOR_TYPE = "Ausgabetensortyp verwenden"
        self.USE_TOKEN_EMBEDDING_TYPE = "Token-Einbettungstyp verwenden"
        self.KEEP_SPLIT = "Teilung beibehalten"
        self.KV_OVERRIDES = "KV-Überschreibungen:"
        self.ADD_NEW_OVERRIDE = "Neue Überschreibung hinzufügen"
        self.QUANTIZE_MODEL = "Modell quantisieren"
        self.EXTRA_ARGUMENTS = "Zusätzliche Argumente:"
        self.EXTRA_ARGUMENTS_LABEL = "Zusätzliche Kommandozeilenargumente"
        self.QUANTIZATION_COMMAND = "Quantisierungsbefehl"

        # Presets
        self.SAVE_PRESET = "Voreinstellung speichern"
        self.LOAD_PRESET = "Voreinstellung laden"

        # Tasks
        self.TASKS = "Aufgaben:"

        # llama.cpp Download
        self.DOWNLOAD_LLAMACPP = "llama.cpp herunterladen"
        self.SELECT_RELEASE = "Release auswählen:"
        self.SELECT_ASSET = "Asset auswählen:"
        self.EXTRACT_CUDA_FILES = "CUDA-Dateien extrahieren"
        self.SELECT_CUDA_BACKEND = "CUDA-Backend auswählen:"
        self.DOWNLOAD = "Herunterladen"
        self.REFRESH_RELEASES = "Releases aktualisieren"

        # IMatrix Generation
        self.IMATRIX_GENERATION = "IMatrix-Generierung"
        self.DATA_FILE = "Datendatei:"
        self.MODEL = "Modell:"
        self.OUTPUT = "Ausgabe:"
        self.OUTPUT_FREQUENCY = "Ausgabehäufigkeit:"
        self.GPU_OFFLOAD = "GPU-Auslagerung:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "IMatrix generieren"
        self.CONTEXT_SIZE = "Kontextgröße:"
        self.CONTEXT_SIZE_FOR_IMATRIX = "Kontextgröße für IMatrix-Generierung"
        self.THREADS = "Threads:"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = (
            "Anzahl der Threads für IMatrix-Generierung"
        )
        self.IMATRIX_GENERATION_COMMAND = "IMatrix-Generierungsbefehl"

        # LoRA Conversion
        self.LORA_CONVERSION = "LoRA-Konvertierung"
        self.LORA_INPUT_PATH = "LoRA-Eingabepfad"
        self.LORA_OUTPUT_PATH = "LoRA-Ausgabepfad"
        self.SELECT_LORA_INPUT_DIRECTORY = "LoRA-Eingabeverzeichnis auswählen"
        self.SELECT_LORA_OUTPUT_FILE = "LoRA-Ausgabedatei auswählen"
        self.CONVERT_LORA = "LoRA konvertieren"
        self.LORA_CONVERSION_COMMAND = "LoRA-Konvertierungsbefehl"

        # LoRA Export
        self.EXPORT_LORA = "LoRA exportieren"
        self.GGML_LORA_ADAPTERS = "GGML LoRA-Adapter"
        self.SELECT_LORA_ADAPTER_FILES = "LoRA-Adapterdateien auswählen"
        self.ADD_ADAPTER = "Adapter hinzufügen"
        self.DELETE_ADAPTER = "Löschen"
        self.LORA_SCALE = "LoRA-Skala"
        self.ENTER_LORA_SCALE_VALUE = "LoRA-Skalenwert eingeben (Optional)"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = "Anzahl der Threads für LoRA-Export"
        self.LORA_EXPORT_COMMAND = "LoRA-Exportbefehl"

        # HuggingFace to GGUF Conversion
        self.HF_TO_GGUF_CONVERSION = "HuggingFace zu GGUF Konvertierung"
        self.MODEL_DIRECTORY = "Modellverzeichnis:"
        self.OUTPUT_FILE = "Ausgabedatei:"
        self.OUTPUT_TYPE = "Ausgabetyp:"
        self.VOCAB_ONLY = "Nur Vokabular"
        self.USE_TEMP_FILE = "Temporäre Datei verwenden"
        self.NO_LAZY_EVALUATION = "Keine verzögerte Auswertung"
        self.MODEL_NAME = "Modellname:"
        self.VERBOSE = "Ausführlich"
        self.SPLIT_MAX_SIZE = "Maximale Teilungsgröße:"
        self.DRY_RUN = "Testlauf"
        self.CONVERT_HF_TO_GGUF = "HF zu GGUF konvertieren"
        self.SELECT_HF_MODEL_DIRECTORY = "HuggingFace-Modellverzeichnis auswählen"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = "HuggingFace-Modellverzeichnis durchsuchen"
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = (
            "HuggingFace zu GGUF Ausgabedatei durchsuchen"
        )

        # Update Checking
        self.UPDATE_AVAILABLE = "Update verfügbar"
        self.NEW_VERSION_AVAILABLE = "Eine neue Version ist verfügbar: {}"
        self.DOWNLOAD_NEW_VERSION = "Herunterladen?"
        self.ERROR_CHECKING_FOR_UPDATES = "Fehler beim Prüfen auf Updates:"
        self.CHECKING_FOR_UPDATES = "Prüfe auf Updates"

        # General Messages
        self.ERROR = "Fehler"
        self.WARNING = "Warnung"
        self.PROPERTIES = "Eigenschaften"
        self.CANCEL = "Abbrechen"
        self.RESTART = "Neustart"
        self.DELETE = "Löschen"
        self.RENAME = "Umbenennen"
        self.CONFIRM_DELETION = (
            "Sind Sie sicher, dass Sie diese Aufgabe löschen möchten?"
        )
        self.TASK_RUNNING_WARNING = (
            "Einige Aufgaben laufen noch. Sind Sie sicher, dass Sie beenden möchten?"
        )
        self.YES = "Ja"
        self.NO = "Nein"
        self.COMPLETED = "Abgeschlossen"

        # File Types
        self.ALL_FILES = "Alle Dateien (*)"
        self.GGUF_FILES = "GGUF-Dateien (*.gguf)"
        self.DAT_FILES = "DAT-Dateien (*.dat)"
        self.JSON_FILES = "JSON-Dateien (*.json)"
        self.BIN_FILES = "Binärdateien (*.bin)"
        self.LORA_FILES = "LoRA-Dateien (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "GGUF- und Binärdateien (*.gguf *.bin)"
        self.SHARDED = "geteilt"

        # Status Messages
        self.DOWNLOAD_COMPLETE = "Download abgeschlossen"
        self.CUDA_EXTRACTION_FAILED = "CUDA-Extraktion fehlgeschlagen"
        self.PRESET_SAVED = "Voreinstellung gespeichert"
        self.PRESET_LOADED = "Voreinstellung geladen"
        self.NO_ASSET_SELECTED = "Kein Asset ausgewählt"
        self.DOWNLOAD_FAILED = "Download fehlgeschlagen"
        self.NO_BACKEND_SELECTED = "Kein Backend ausgewählt"
        self.NO_MODEL_SELECTED = "Kein Modell ausgewählt"
        self.NO_SUITABLE_CUDA_BACKENDS = "Keine geeigneten CUDA-Backends gefunden"
        self.IN_PROGRESS = "In Bearbeitung"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "llama.cpp-Binärdatei heruntergeladen und extrahiert nach {0}"
        )
        self.CUDA_FILES_EXTRACTED = "CUDA-Dateien extrahiert nach"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Kein geeignetes CUDA-Backend für die Extraktion gefunden"
        )
        self.ERROR_FETCHING_RELEASES = "Fehler beim Abrufen der Releases: {0}"
        self.CONFIRM_DELETION_TITLE = "Löschen bestätigen"
        self.LOG_FOR = "Protokoll für {0}"
        self.FAILED_TO_LOAD_PRESET = "Fehler beim Laden der Voreinstellung: {0}"
        self.INITIALIZING_AUTOGGUF = "Initialisiere AutoGGUF-Anwendung"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF-Initialisierung abgeschlossen"
        self.REFRESHING_BACKENDS = "Aktualisiere Backends"
        self.NO_BACKENDS_AVAILABLE = "Keine Backends verfügbar"
        self.FOUND_VALID_BACKENDS = "{0} gültige Backends gefunden"
        self.SAVING_PRESET = "Speichere Voreinstellung"
        self.PRESET_SAVED_TO = "Voreinstellung gespeichert in {0}"
        self.LOADING_PRESET = "Lade Voreinstellung"
        self.PRESET_LOADED_FROM = "Voreinstellung geladen aus {0}"
        self.ADDING_KV_OVERRIDE = "Füge KV-Überschreibung hinzu: {0}"
        self.SAVING_TASK_PRESET = "Speichere Aufgaben-Voreinstellung für {0}"
        self.TASK_PRESET_SAVED = "Aufgaben-Voreinstellung gespeichert"
        self.TASK_PRESET_SAVED_TO = "Aufgaben-Voreinstellung gespeichert in {0}"
        self.RESTARTING_TASK = "Starte Aufgabe neu: {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "Download abgeschlossen. Extrahiert nach: {0}"
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp-Binärdatei heruntergeladen und extrahiert nach {0}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Kein geeignetes CUDA-Backend für die Extraktion gefunden"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp-Binärdatei heruntergeladen und extrahiert nach {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Aktualisiere llama.cpp-Releases"
        self.UPDATING_ASSET_LIST = "Aktualisiere Asset-Liste"
        self.UPDATING_CUDA_OPTIONS = "Aktualisiere CUDA-Optionen"
        self.STARTING_LLAMACPP_DOWNLOAD = "Starte llama.cpp-Download"
        self.UPDATING_CUDA_BACKENDS = "Aktualisiere CUDA-Backends"
        self.NO_CUDA_BACKEND_SELECTED = (
            "Kein CUDA-Backend für die Extraktion ausgewählt"
        )
        self.EXTRACTING_CUDA_FILES = "Extrahiere CUDA-Dateien von {0} nach {1}"
        self.DOWNLOAD_ERROR = "Download-Fehler: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Zeige Aufgaben-Kontextmenü"
        self.SHOWING_PROPERTIES_FOR_TASK = "Zeige Eigenschaften für Aufgabe: {0}"
        self.CANCELLING_TASK = "Breche Aufgabe ab: {0}"
        self.CANCELED = "Abgebrochen"
        self.DELETING_TASK = "Lösche Aufgabe: {0}"
        self.LOADING_MODELS = "Lade Modelle"
        self.LOADED_MODELS = "{0} Modelle geladen"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Durchsuche Modellverzeichnis"
        self.SELECT_MODELS_DIRECTORY = "Modellverzeichnis auswählen"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Durchsuche Ausgabeverzeichnis"
        self.SELECT_OUTPUT_DIRECTORY = "Ausgabeverzeichnis auswählen"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Durchsuche Protokollverzeichnis"
        self.SELECT_LOGS_DIRECTORY = "Protokollverzeichnis auswählen"
        self.BROWSING_FOR_IMATRIX_FILE = "Durchsuche IMatrix-Datei"
        self.SELECT_IMATRIX_FILE = "IMatrix-Datei auswählen"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU-Nutzung: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validiere Quantisierungseingaben"
        self.MODELS_PATH_REQUIRED = "Modellpfad ist erforderlich"
        self.OUTPUT_PATH_REQUIRED = "Ausgabepfad ist erforderlich"
        self.LOGS_PATH_REQUIRED = "Protokollpfad ist erforderlich"
        self.STARTING_MODEL_QUANTIZATION = "Starte Modellquantisierung"
        self.INPUT_FILE_NOT_EXIST = "Eingabedatei '{0}' existiert nicht."
        self.QUANTIZING_MODEL_TO = "Quantisiere {0} zu {1}"
        self.QUANTIZATION_TASK_STARTED = "Quantisierungsaufgabe gestartet für {0}"
        self.ERROR_STARTING_QUANTIZATION = "Fehler beim Starten der Quantisierung: {0}"
        self.UPDATING_MODEL_INFO = "Aktualisiere Modellinformationen: {0}"
        self.TASK_FINISHED = "Aufgabe abgeschlossen: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Zeige Aufgabendetails für: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Durchsuche IMatrix-Datendatei"
        self.SELECT_DATA_FILE = "Datendatei auswählen"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Durchsuche IMatrix-Modelldatei"
        self.SELECT_MODEL_FILE = "Modelldatei auswählen"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Durchsuche IMatrix-Ausgabedatei"
        self.SELECT_OUTPUT_FILE = "Ausgabedatei auswählen"
        self.STARTING_IMATRIX_GENERATION = "Starte IMatrix-Generierung"
        self.BACKEND_PATH_NOT_EXIST = "Backend-Pfad existiert nicht: {0}"
        self.GENERATING_IMATRIX = "Generiere IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Fehler beim Starten der IMatrix-Generierung: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix-Generierungsaufgabe gestartet"
        self.ERROR_MESSAGE = "Fehler: {0}"
        self.TASK_ERROR = "Aufgabenfehler: {0}"
        self.APPLICATION_CLOSING = "Anwendung wird geschlossen"
        self.APPLICATION_CLOSED = "Anwendung geschlossen"
        self.SELECT_QUANTIZATION_TYPE = "Wählen Sie den Quantisierungstyp"
        self.ALLOWS_REQUANTIZING = (
            "Erlaubt die Requantisierung von bereits quantisierten Tensoren"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Lässt output.weight un(re)quantisiert"
        self.DISABLE_K_QUANT_MIXTURES = "Deaktiviert k-Quant-Mischungen und quantisiert alle Tensoren zum gleichen Typ"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Verwendet Daten in der Datei als Wichtigkeitsmatrix für Quant-Optimierungen"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Verwendet Wichtigkeitsmatrix für diese Tensoren"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Verwendet keine Wichtigkeitsmatrix für diese Tensoren"
        )
        self.OUTPUT_TENSOR_TYPE = "Ausgabetensortyp:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Verwendet diesen Typ für den output.weight Tensor"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token-Einbettungstyp:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Verwendet diesen Typ für den Token-Einbettungstensor"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Generiert quantisiertes Modell in den gleichen Shards wie die Eingabe"
        )
        self.OVERRIDE_MODEL_METADATA = "Modellmetadaten überschreiben"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Eingabedatendatei für IMatrix-Generierung"
        self.MODEL_TO_BE_QUANTIZED = "Zu quantisierendes Modell"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Ausgabepfad für die generierte IMatrix"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Wie oft die IMatrix gespeichert werden soll"
        self.SET_GPU_OFFLOAD_VALUE = "GPU-Auslagerungswert setzen (-ngl)"
        self.STARTING_LORA_CONVERSION = "Starte LoRA-Konvertierung"
        self.LORA_INPUT_PATH_REQUIRED = "LoRA-Eingabepfad ist erforderlich."
        self.LORA_OUTPUT_PATH_REQUIRED = "LoRA-Ausgabepfad ist erforderlich."
        self.ERROR_STARTING_LORA_CONVERSION = (
            "Fehler beim Starten der LoRA-Konvertierung: {}"
        )
        self.LORA_CONVERSION_TASK_STARTED = "LoRA-Konvertierungsaufgabe gestartet."
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = "Durchsuche LoRA-Eingabeverzeichnis..."
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "Durchsuche LoRA-Ausgabedatei..."
        self.CONVERTING_LORA = "LoRA-Konvertierung"
        self.LORA_CONVERSION_FINISHED = "LoRA-Konvertierung abgeschlossen."
        self.LORA_FILE_MOVED = "LoRA-Datei von {} nach {} verschoben."
        self.LORA_FILE_NOT_FOUND = "LoRA-Datei nicht gefunden: {}."
        self.ERROR_MOVING_LORA_FILE = "Fehler beim Verschieben der LoRA-Datei: {}"
        self.MODEL_PATH_REQUIRED = "Modellpfad ist erforderlich."
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = (
            "Mindestens ein LoRA-Adapter ist erforderlich."
        )
        self.INVALID_LORA_SCALE_VALUE = "Ungültiger LoRA-Skalenwert."
        self.ERROR_STARTING_LORA_EXPORT = "Fehler beim Starten des LoRA-Exports: {}"
        self.LORA_EXPORT_TASK_STARTED = "LoRA-Exportaufgabe gestartet."
        self.EXPORTING_LORA = "Exportiere LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = (
            "Durchsuche Export-LoRA-Modelldatei..."
        )
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = (
            "Durchsuche Export-LoRA-Ausgabedatei..."
        )
        self.ADDING_LORA_ADAPTER = "Füge LoRA-Adapter hinzu..."
        self.DELETING_LORA_ADAPTER = "Lösche LoRA-Adapter..."
        self.SELECT_LORA_ADAPTER_FILE = "LoRA-Adapterdatei auswählen"
        self.STARTING_LORA_EXPORT = "Starte LoRA-Export..."
        self.SELECT_OUTPUT_TYPE = "Ausgabetyp auswählen (GGUF oder GGML)"
        self.BASE_MODEL = "Basismodell"
        self.SELECT_BASE_MODEL_FILE = "Basismodelldatei auswählen (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = (
            "Basismodellpfad ist für GGUF-Ausgabe erforderlich."
        )
        self.BROWSING_FOR_BASE_MODEL_FILE = "Durchsuche Basismodelldatei..."
        self.SELECT_BASE_MODEL_FOLDER = "Basismodellordner auswählen (mit safetensors)"
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "Durchsuche Basismodellordner..."
        self.LORA_CONVERSION_FROM_TO = "LoRA-Konvertierung von {} nach {}"
        self.GENERATING_IMATRIX_FOR = "Generiere IMatrix für {}"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = (
            "Modellpfad ist für IMatrix-Generierung erforderlich."
        )
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = (
            "Kein Asset für CUDA-Überprüfung ausgewählt"
        )
        self.NO_QUANTIZATION_TYPE_SELECTED = "Kein Quantisierungstyp ausgewählt. Bitte wählen Sie mindestens einen Quantisierungstyp aus."
        self.STARTING_HF_TO_GGUF_CONVERSION = "Starte HuggingFace zu GGUF Konvertierung"
        self.MODEL_DIRECTORY_REQUIRED = "Modellverzeichnis ist erforderlich"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "HF zu GGUF Konvertierungsbefehl: {}"
        self.CONVERTING_TO_GGUF = "Konvertiere {} zu GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "Fehler beim Starten der HuggingFace zu GGUF Konvertierung: {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = (
            "HuggingFace zu GGUF Konvertierungsaufgabe gestartet"
        )

        # Split GGUF
        self.SPLIT_GGUF = "GGUF teilen"
        self.SPLIT_MAX_SIZE = "Maximale Teilungsgröße"
        self.SPLIT_MAX_TENSORS = "Maximale Anzahl an Tensoren"
        self.SPLIT_GGUF_TASK_STARTED = "GGUF-Teilungsaufgabe gestartet"
        self.SPLIT_GGUF_TASK_FINISHED = "GGUF-Teilungsaufgabe abgeschlossen"
        self.SPLIT_GGUF_COMMAND = "GGUF-Teilungsbefehl"
        self.SPLIT_GGUF_ERROR = "Fehler beim Starten der GGUF-Teilung"
        self.NUMBER_OF_TENSORS = "Anzahl der Tensoren"
        self.SIZE_IN_UNITS = "Größe in G/M"

        # Model actions
        self.CONFIRM_DELETE = "Löschen bestätigen"
        self.DELETE_MODEL_WARNING = (
            "Sind Sie sicher, dass Sie das Modell löschen möchten: {}?"
        )
        self.MODEL_RENAMED_SUCCESSFULLY = "Modell erfolgreich umbenannt."
        self.MODEL_DELETED_SUCCESSFULLY = "Modell erfolgreich gelöscht."

        # HuggingFace Transfer
        self.ALL_FIELDS_REQUIRED = "Alle Felder sind erforderlich."
        self.HUGGINGFACE_UPLOAD_COMMAND = "HuggingFace Upload-Befehl: "
        self.UPLOADING = "Hochladen"
        self.UPLOADING_FOLDER = "Ordner hochladen"
        self.HF_TRANSFER_TASK_NAME = "{} {} zu {} von {}"
        self.ERROR_STARTING_HF_TRANSFER = "Fehler beim Starten des HF-Transfers: {}"
        self.STARTED_HUGGINGFACE_TRANSFER = "HuggingFace {}-Operation gestartet."
        self.SELECT_FOLDER = "Ordner auswählen"
        self.SELECT_FILE = "Datei auswählen"


class _Portuguese(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Quantizador Automático de Modelos GGUF)"
        self.RAM_USAGE = "Uso de RAM:"
        self.CPU_USAGE = "Uso da CPU:"
        self.BACKEND = "Backend do Llama.cpp:"
        self.REFRESH_BACKENDS = "Atualizar Backends"
        self.MODELS_PATH = "Caminho dos Modelos:"
        self.OUTPUT_PATH = "Caminho de Saída:"
        self.LOGS_PATH = "Caminho dos Logs:"
        self.BROWSE = "Navegar"
        self.AVAILABLE_MODELS = "Modelos Disponíveis:"
        self.QUANTIZATION_TYPE = "Tipo de Quantização:"
        self.ALLOW_REQUANTIZE = "Permitir Requantização"
        self.LEAVE_OUTPUT_TENSOR = "Manter Tensor de Saída"
        self.PURE = "Puro"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Incluir Pesos:"
        self.EXCLUDE_WEIGHTS = "Excluir Pesos:"
        self.USE_OUTPUT_TENSOR_TYPE = "Usar Tipo de Tensor de Saída"
        self.USE_TOKEN_EMBEDDING_TYPE = "Usar Tipo de Incorporação de Token"
        self.KEEP_SPLIT = "Manter Divisão"
        self.KV_OVERRIDES = "Substituições KV:"
        self.ADD_NEW_OVERRIDE = "Adicionar Nova Substituição"
        self.QUANTIZE_MODEL = "Quantizar Modelo"
        self.SAVE_PRESET = "Salvar Predefinição"
        self.LOAD_PRESET = "Carregar Predefinição"
        self.TASKS = "Tarefas:"
        self.DOWNLOAD_LLAMACPP = "Baixar llama.cpp"
        self.SELECT_RELEASE = "Selecionar Versão:"
        self.SELECT_ASSET = "Selecionar Ativo:"
        self.EXTRACT_CUDA_FILES = "Extrair Arquivos CUDA"
        self.SELECT_CUDA_BACKEND = "Selecionar Backend CUDA:"
        self.DOWNLOAD = "Baixar"
        self.IMATRIX_GENERATION = "Geração de IMatrix"
        self.DATA_FILE = "Arquivo de Dados:"
        self.MODEL = "Modelo:"
        self.OUTPUT = "Saída:"
        self.OUTPUT_FREQUENCY = "Frequência de Saída:"
        self.GPU_OFFLOAD = "Offload da GPU:"
        self.AUTO = "Automático"
        self.GENERATE_IMATRIX = "Gerar IMatrix"
        self.ERROR = "Erro"
        self.WARNING = "Aviso"
        self.PROPERTIES = "Propriedades"
        self.CANCEL = "Cancelar"
        self.RESTART = "Reiniciar"
        self.DELETE = "Excluir"
        self.CONFIRM_DELETION = "Tem certeza de que deseja excluir esta tarefa?"
        self.TASK_RUNNING_WARNING = (
            "Algumas tarefas ainda estão em execução. Tem certeza de que deseja sair?"
        )
        self.YES = "Sim"
        self.NO = "Não"
        self.DOWNLOAD_COMPLETE = "Download Concluído"
        self.CUDA_EXTRACTION_FAILED = "Falha na Extração do CUDA"
        self.PRESET_SAVED = "Predefinição Salva"
        self.PRESET_LOADED = "Predefinição Carregada"
        self.NO_ASSET_SELECTED = "Nenhum ativo selecionado"
        self.DOWNLOAD_FAILED = "Falha no download"
        self.NO_BACKEND_SELECTED = "Nenhum backend selecionado"
        self.NO_MODEL_SELECTED = "Nenhum modelo selecionado"
        self.REFRESH_RELEASES = "Atualizar Versões"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nenhum backend CUDA adequado encontrado"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Binário llama.cpp baixado e extraído para {0}\nArquivos CUDA extraídos para {1}"
        self.CUDA_FILES_EXTRACTED = "Arquivos CUDA extraídos para"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nenhum backend CUDA adequado encontrado para extração"
        )
        self.ERROR_FETCHING_RELEASES = "Erro ao buscar versões: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirmar Exclusão"
        self.LOG_FOR = "Log para {0}"
        self.ALL_FILES = "Todos os Arquivos (*)"
        self.GGUF_FILES = "Arquivos GGUF (*.gguf)"
        self.DAT_FILES = "Arquivos DAT (*.dat)"
        self.JSON_FILES = "Arquivos JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Falha ao carregar a predefinição: {0}"
        self.INITIALIZING_AUTOGGUF = "Inicializando o aplicativo AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Inicialização do AutoGGUF concluída"
        self.REFRESHING_BACKENDS = "Atualizando backends"
        self.NO_BACKENDS_AVAILABLE = "Nenhum backend disponível"
        self.FOUND_VALID_BACKENDS = "{0} backends válidos encontrados"
        self.SAVING_PRESET = "Salvando predefinição"
        self.PRESET_SAVED_TO = "Predefinição salva em {0}"
        self.LOADING_PRESET = "Carregando predefinição"
        self.PRESET_LOADED_FROM = "Predefinição carregada de {0}"
        self.ADDING_KV_OVERRIDE = "Adicionando substituição KV: {0}"
        self.SAVING_TASK_PRESET = "Salvando predefinição de tarefa para {0}"
        self.TASK_PRESET_SAVED = "Predefinição de Tarefa Salva"
        self.TASK_PRESET_SAVED_TO = "Predefinição de tarefa salva em {0}"
        self.RESTARTING_TASK = "Reiniciando tarefa: {0}"
        self.IN_PROGRESS = "Em Andamento"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download concluído. Extraído para: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Binário llama.cpp baixado e extraído para {0}\nArquivos CUDA extraídos para {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nenhum backend CUDA adequado encontrado para extração"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Binário llama.cpp baixado e extraído para {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Atualizando versões do llama.cpp"
        self.UPDATING_ASSET_LIST = "Atualizando lista de ativos"
        self.UPDATING_CUDA_OPTIONS = "Atualizando opções CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Iniciando download do llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Atualizando backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Nenhum backend CUDA selecionado para extração"
        self.EXTRACTING_CUDA_FILES = "Extraindo arquivos CUDA de {0} para {1}"
        self.DOWNLOAD_ERROR = "Erro de download: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Exibindo menu de contexto da tarefa"
        self.SHOWING_PROPERTIES_FOR_TASK = "Exibindo propriedades para a tarefa: {0}"
        self.CANCELLING_TASK = "Cancelando tarefa: {0}"
        self.CANCELED = "Cancelado"
        self.DELETING_TASK = "Excluindo tarefa: {0}"
        self.LOADING_MODELS = "Carregando modelos"
        self.LOADED_MODELS = "{0} modelos carregados"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Navegando pelo diretório de modelos"
        self.SELECT_MODELS_DIRECTORY = "Selecionar Diretório de Modelos"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Navegando pelo diretório de saída"
        self.SELECT_OUTPUT_DIRECTORY = "Selecionar Diretório de Saída"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Navegando pelo diretório de logs"
        self.SELECT_LOGS_DIRECTORY = "Selecionar Diretório de Logs"
        self.BROWSING_FOR_IMATRIX_FILE = "Navegando pelo arquivo IMatrix"
        self.SELECT_IMATRIX_FILE = "Selecionar Arquivo IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Uso da CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validando entradas de quantização"
        self.MODELS_PATH_REQUIRED = "O caminho dos modelos é obrigatório"
        self.OUTPUT_PATH_REQUIRED = "O caminho de saída é obrigatório"
        self.LOGS_PATH_REQUIRED = "O caminho dos logs é obrigatório"
        self.STARTING_MODEL_QUANTIZATION = "Iniciando a quantização do modelo"
        self.INPUT_FILE_NOT_EXIST = "O arquivo de entrada '{0}' não existe."
        self.QUANTIZING_MODEL_TO = "Quantizando {0} para {1}"
        self.QUANTIZATION_TASK_STARTED = "Tarefa de quantização iniciada para {0}"
        self.ERROR_STARTING_QUANTIZATION = "Erro ao iniciar a quantização: {0}"
        self.UPDATING_MODEL_INFO = "Atualizando informações do modelo: {0}"
        self.TASK_FINISHED = "Tarefa concluída: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Mostrando detalhes da tarefa para: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Navegando pelo arquivo de dados IMatrix"
        self.SELECT_DATA_FILE = "Selecionar Arquivo de Dados"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = (
            "Navegando pelo arquivo de modelo IMatrix"
        )
        self.SELECT_MODEL_FILE = "Selecionar Arquivo de Modelo"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = (
            "Navegando pelo arquivo de saída IMatrix"
        )
        self.SELECT_OUTPUT_FILE = "Selecionar Arquivo de Saída"
        self.STARTING_IMATRIX_GENERATION = "Iniciando a geração de IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "O caminho do backend não existe: {0}"
        self.GENERATING_IMATRIX = "Gerando IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Erro ao iniciar a geração de IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Tarefa de geração de IMatrix iniciada"
        self.ERROR_MESSAGE = "Erro: {0}"
        self.TASK_ERROR = "Erro de tarefa: {0}"
        self.APPLICATION_CLOSING = "Fechando o aplicativo"
        self.APPLICATION_CLOSED = "Aplicativo fechado"
        self.SELECT_QUANTIZATION_TYPE = "Selecione o tipo de quantização"
        self.ALLOWS_REQUANTIZING = (
            "Permite requantizar tensores que já foram quantizados"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Deixará output.weight não (re)quantizado"
        self.DISABLE_K_QUANT_MIXTURES = "Desabilitar misturas k-quant e quantizar todos os tensores para o mesmo tipo"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Usar os dados no arquivo como matriz de importância para otimizações de quantização"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Usar matriz de importância para estes tensores"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Não usar matriz de importância para estes tensores"
        )
        self.OUTPUT_TENSOR_TYPE = "Tipo de Tensor de Saída:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Usar este tipo para o tensor output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Tipo de Incorporação de Token:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Usar este tipo para o tensor de incorporações de token"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Irá gerar o modelo quantizado nos mesmos shards da entrada"
        )
        self.OVERRIDE_MODEL_METADATA = "Substituir metadados do modelo"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Arquivo de dados de entrada para geração de IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modelo a ser quantizado"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Caminho de saída para o IMatrix gerado"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Com que frequência salvar o IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Definir valor de offload da GPU (-ngl)"
        self.COMPLETED = "Concluído"
        self.REFRESH_MODELS = "Atualizar modelos"


class _Arabic(_Localization):
    def __init__(self):
        super().__init__()

        # واجهة المستخدم العامة
        self.WINDOW_TITLE = "AutoGGUF (محول نماذج GGUF الآلي)"
        self.RAM_USAGE = "استخدام الذاكرة:"
        self.CPU_USAGE = "استخدام المعالج:"
        self.BACKEND = "خلفية Llama.cpp:"
        self.REFRESH_BACKENDS = "تحديث الخلفيات"
        self.MODELS_PATH = "مسار النماذج:"
        self.OUTPUT_PATH = "مسار الإخراج:"
        self.LOGS_PATH = "مسار السجلات:"
        self.BROWSE = "تصفح"
        self.AVAILABLE_MODELS = "النماذج المتاحة:"
        self.REFRESH_MODELS = "تحديث النماذج"

        # استيراد النموذج
        self.IMPORT_MODEL = "استيراد نموذج"
        self.SELECT_MODEL_TO_IMPORT = "اختر النموذج للاستيراد"
        self.CONFIRM_IMPORT = "تأكيد الاستيراد"
        self.IMPORT_MODEL_CONFIRMATION = "هل تريد استيراد النموذج {}؟"
        self.MODEL_IMPORTED_SUCCESSFULLY = "تم استيراد النموذج {} بنجاح"
        self.IMPORTING_MODEL = "جاري استيراد النموذج"
        self.IMPORTED_MODEL_TOOLTIP = "النموذج المستورد: {}"

        # التحقق من GGUF
        self.INVALID_GGUF_FILE = "ملف GGUF غير صالح: {}"
        self.SHARDED_MODEL_NAME = "{} (مجزأ)"
        self.IMPORTED_MODEL_TOOLTIP = "النموذج المستورد: {}"
        self.CONCATENATED_FILE_WARNING = (
            "هذا جزء من ملف مدمج. لن يعمل مع llama-quantize؛ يرجى دمج الملف أولاً."
        )
        self.CONCATENATED_FILES_FOUND = (
            "تم العثور على {} أجزاء ملفات مدمجة. يرجى دمج الملفات أولاً."
        )

        # مراقبة وحدة معالجة الرسومات
        self.GPU_USAGE = "استخدام وحدة معالجة الرسومات:"
        self.GPU_USAGE_FORMAT = "وحدة معالجة الرسومات: {:.1f}% | ذاكرة الفيديو: {:.1f}% ({} ميجابايت / {} ميجابايت)"
        self.GPU_DETAILS = "تفاصيل وحدة معالجة الرسومات"
        self.GPU_USAGE_OVER_TIME = "استخدام وحدة معالجة الرسومات عبر الوقت"
        self.VRAM_USAGE_OVER_TIME = "استخدام ذاكرة الفيديو عبر الوقت"
        self.PERCENTAGE = "النسبة المئوية"
        self.TIME = "الوقت (ثانية)"
        self.NO_GPU_DETECTED = "لم يتم اكتشاف وحدة معالجة رسومات"
        self.SELECT_GPU = "اختر وحدة معالجة الرسومات"
        self.AMD_GPU_NOT_SUPPORTED = (
            "تم اكتشاف وحدة معالجة رسومات AMD، لكنها غير مدعومة"
        )

        # التكميم
        self.QUANTIZATION_TYPE = "نوع التكميم:"
        self.ALLOW_REQUANTIZE = "السماح بإعادة التكميم"
        self.LEAVE_OUTPUT_TENSOR = "ترك تنسور الإخراج"
        self.PURE = "نقي"
        self.IMATRIX = "مصفوفة الأهمية:"
        self.INCLUDE_WEIGHTS = "تضمين الأوزان:"
        self.EXCLUDE_WEIGHTS = "استبعاد الأوزان:"
        self.USE_OUTPUT_TENSOR_TYPE = "استخدام نوع تنسور الإخراج"
        self.USE_TOKEN_EMBEDDING_TYPE = "استخدام نوع تضمين الرمز"
        self.KEEP_SPLIT = "الحفاظ على التقسيم"
        self.KV_OVERRIDES = "تجاوزات KV:"
        self.ADD_NEW_OVERRIDE = "إضافة تجاوز جديد"
        self.QUANTIZE_MODEL = "تكميم النموذج"
        self.EXTRA_ARGUMENTS = "وسائط إضافية:"
        self.EXTRA_ARGUMENTS_LABEL = "وسائط سطر الأوامر الإضافية"
        self.QUANTIZATION_COMMAND = "أمر التكميم"

        # الإعدادات المسبقة
        self.SAVE_PRESET = "حفظ الإعداد المسبق"
        self.LOAD_PRESET = "تحميل الإعداد المسبق"

        # المهام
        self.TASKS = "المهام:"

        # تنزيل llama.cpp
        self.DOWNLOAD_LLAMACPP = "تنزيل llama.cpp"
        self.SELECT_RELEASE = "اختر الإصدار:"
        self.SELECT_ASSET = "اختر الأصل:"
        self.EXTRACT_CUDA_FILES = "استخراج ملفات CUDA"
        self.SELECT_CUDA_BACKEND = "اختر خلفية CUDA:"
        self.DOWNLOAD = "تنزيل"
        self.REFRESH_RELEASES = "تحديث الإصدارات"

        # توليد مصفوفة الأهمية
        self.IMATRIX_GENERATION = "توليد مصفوفة الأهمية"
        self.DATA_FILE = "ملف البيانات:"
        self.MODEL = "النموذج:"
        self.OUTPUT = "الإخراج:"
        self.OUTPUT_FREQUENCY = "تردد الإخراج:"
        self.GPU_OFFLOAD = "تحميل وحدة معالجة الرسومات:"
        self.AUTO = "تلقائي"
        self.GENERATE_IMATRIX = "توليد مصفوفة الأهمية"
        self.CONTEXT_SIZE = "حجم السياق:"
        self.CONTEXT_SIZE_FOR_IMATRIX = "حجم السياق لتوليد مصفوفة الأهمية"
        self.THREADS = "عدد المسارات:"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = "عدد المسارات لتوليد مصفوفة الأهمية"
        self.IMATRIX_GENERATION_COMMAND = "أمر توليد مصفوفة الأهمية"

        # تحويل LoRA
        self.LORA_CONVERSION = "تحويل LoRA"
        self.LORA_INPUT_PATH = "مسار إدخال LoRA"
        self.LORA_OUTPUT_PATH = "مسار إخراج LoRA"
        self.SELECT_LORA_INPUT_DIRECTORY = "اختر مجلد إدخال LoRA"
        self.SELECT_LORA_OUTPUT_FILE = "اختر ملف إخراج LoRA"
        self.CONVERT_LORA = "تحويل LoRA"
        self.LORA_CONVERSION_COMMAND = "أمر تحويل LoRA"

        # تصدير LoRA
        self.EXPORT_LORA = "تصدير LoRA"
        self.GGML_LORA_ADAPTERS = "محولات GGML LoRA"
        self.SELECT_LORA_ADAPTER_FILES = "اختر ملفات محول LoRA"
        self.ADD_ADAPTER = "إضافة محول"
        self.DELETE_ADAPTER = "حذف"
        self.LORA_SCALE = "مقياس LoRA"
        self.ENTER_LORA_SCALE_VALUE = "أدخل قيمة مقياس LoRA (اختياري)"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = "عدد المسارات لتصدير LoRA"
        self.LORA_EXPORT_COMMAND = "أمر تصدير LoRA"

        # تحويل HuggingFace إلى GGUF
        self.HF_TO_GGUF_CONVERSION = "تحويل HuggingFace إلى GGUF"
        self.MODEL_DIRECTORY = "مجلد النموذج:"
        self.OUTPUT_FILE = "ملف الإخراج:"
        self.OUTPUT_TYPE = "نوع الإخراج:"
        self.VOCAB_ONLY = "المفردات فقط"
        self.USE_TEMP_FILE = "استخدام ملف مؤقت"
        self.NO_LAZY_EVALUATION = "بدون تقييم كسول"
        self.MODEL_NAME = "اسم النموذج:"
        self.VERBOSE = "مفصل"
        self.SPLIT_MAX_SIZE = "الحجم الأقصى للتقسيم:"
        self.DRY_RUN = "تشغيل تجريبي"
        self.CONVERT_HF_TO_GGUF = "تحويل HF إلى GGUF"
        self.SELECT_HF_MODEL_DIRECTORY = "اختر مجلد نموذج HuggingFace"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = "تصفح مجلد نموذج HuggingFace"
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = "تصفح ملف إخراج تحويل HuggingFace إلى GGUF"

        # التحقق من التحديثات
        self.UPDATE_AVAILABLE = "تحديث متاح"
        self.NEW_VERSION_AVAILABLE = "إصدار جديد متاح: {}"
        self.DOWNLOAD_NEW_VERSION = "تنزيل؟"
        self.ERROR_CHECKING_FOR_UPDATES = "خطأ في التحقق من التحديثات:"
        self.CHECKING_FOR_UPDATES = "جاري التحقق من التحديثات"

        # رسائل عامة
        self.ERROR = "خطأ"
        self.WARNING = "تحذير"
        self.PROPERTIES = "الخصائص"
        self.CANCEL = "إلغاء"
        self.RESTART = "إعادة تشغيل"
        self.DELETE = "حذف"
        self.CONFIRM_DELETION = "هل أنت متأكد أنك تريد حذف هذه المهمة؟"
        self.TASK_RUNNING_WARNING = (
            "بعض المهام ما زالت قيد التشغيل. هل أنت متأكد أنك تريد الخروج؟"
        )
        self.YES = "نعم"
        self.NO = "لا"
        self.COMPLETED = "مكتمل"

        # أنواع الملفات
        self.ALL_FILES = "جميع الملفات (*)"
        self.GGUF_FILES = "ملفات GGUF (*.gguf)"
        self.DAT_FILES = "ملفات DAT (*.dat)"
        self.JSON_FILES = "ملفات JSON (*.json)"
        self.BIN_FILES = "ملفات ثنائية (*.bin)"
        self.LORA_FILES = "ملفات LoRA (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "ملفات GGUF وثنائية (*.gguf *.bin)"
        self.SHARDED = "مجزأ"

        # رسائل الحالة
        self.DOWNLOAD_COMPLETE = "اكتمل التنزيل"
        self.CUDA_EXTRACTION_FAILED = "فشل استخراج CUDA"
        self.PRESET_SAVED = "تم حفظ الإعداد المسبق"
        self.PRESET_LOADED = "تم تحميل الإعداد المسبق"
        self.NO_ASSET_SELECTED = "لم يتم اختيار أي أصل"
        self.DOWNLOAD_FAILED = "فشل التنزيل"
        self.NO_BACKEND_SELECTED = "لم يتم اختيار أي خلفية"
        self.NO_MODEL_SELECTED = "لم يتم اختيار أي نموذج"
        self.NO_SUITABLE_CUDA_BACKENDS = "لم يتم العثور على خلفيات CUDA مناسبة"
        self.IN_PROGRESS = "قيد التنفيذ"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "تم تنزيل واستخراج ثنائي llama.cpp إلى {0}"
        self.CUDA_FILES_EXTRACTED = "تم استخراج ملفات CUDA إلى"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "لم يتم العثور على خلفية CUDA مناسبة للاستخراج"
        )
        self.ERROR_FETCHING_RELEASES = "خطأ في جلب الإصدارات: {0}"
        self.CONFIRM_DELETION_TITLE = "تأكيد الحذف"
        self.LOG_FOR = "سجل لـ {0}"
        self.FAILED_TO_LOAD_PRESET = "فشل تحميل الإعداد المسبق: {0}"
        self.INITIALIZING_AUTOGGUF = "جاري تهيئة تطبيق AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "اكتملت تهيئة AutoGGUF"
        self.REFRESHING_BACKENDS = "جاري تحديث الخلفيات"
        self.NO_BACKENDS_AVAILABLE = "لا توجد خلفيات متاحة"
        self.FOUND_VALID_BACKENDS = "تم العثور على {0} خلفيات صالحة"
        self.SAVING_PRESET = "جاري حفظ الإعداد المسبق"
        self.PRESET_SAVED_TO = "تم حفظ الإعداد المسبق في {0}"
        self.LOADING_PRESET = "جاري تحميل الإعداد المسبق"
        self.PRESET_LOADED_FROM = "تم تحميل الإعداد المسبق من {0}"
        self.ADDING_KV_OVERRIDE = "إضافة تجاوز KV: {0}"
        self.SAVING_TASK_PRESET = "جاري حفظ إعداد مسبق للمهمة {0}"
        self.TASK_PRESET_SAVED = "تم حفظ الإعداد المسبق للمهمة"
        self.TASK_PRESET_SAVED_TO = "تم حفظ الإعداد المسبق للمهمة في {0}"
        self.RESTARTING_TASK = "إعادة تشغيل المهمة: {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "اكتمل التنزيل. تم الاستخراج إلى: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "تم تنزيل واستخراج ثنائي llama.cpp إلى {0}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "لم يتم العثور على خلفية CUDA مناسبة للاستخراج"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "تم تنزيل واستخراج ثنائي llama.cpp إلى {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "جاري تحديث إصدارات llama.cpp"
        self.UPDATING_ASSET_LIST = "جاري تحديث قائمة الأصول"
        self.UPDATING_CUDA_OPTIONS = "جاري تحديث خيارات CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "بدء تنزيل llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "جاري تحديث خلفيات CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "لم يتم اختيار خلفية CUDA للاستخراج"
        self.EXTRACTING_CUDA_FILES = "جاري استخراج ملفات CUDA من {0} إلى {1}"
        self.DOWNLOAD_ERROR = "خطأ في التنزيل: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "عرض قائمة سياق المهمة"
        self.SHOWING_PROPERTIES_FOR_TASK = "عرض خصائص المهمة: {0}"
        self.CANCELLING_TASK = "إلغاء المهمة: {0}"
        self.CANCELED = "تم الإلغاء"
        self.DELETING_TASK = "حذف المهمة: {0}"
        self.LOADING_MODELS = "جاري تحميل النماذج"
        self.LOADED_MODELS = "تم تحميل {0} نماذج"
        self.BROWSING_FOR_MODELS_DIRECTORY = "تصفح مجلد النماذج"
        self.SELECT_MODELS_DIRECTORY = "اختر مجلد النماذج"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "تصفح مجلد الإخراج"
        self.SELECT_OUTPUT_DIRECTORY = "اختر مجلد الإخراج"
        self.BROWSING_FOR_LOGS_DIRECTORY = "تصفح مجلد السجلات"
        self.SELECT_LOGS_DIRECTORY = "اختر مجلد السجلات"
        self.BROWSING_FOR_IMATRIX_FILE = "تصفح ملف مصفوفة الأهمية"
        self.SELECT_IMATRIX_FILE = "اختر ملف مصفوفة الأهمية"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} ميجابايت / {2} ميجابايت)"
        self.CPU_USAGE_FORMAT = "استخدام المعالج: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "التحقق من صحة مدخلات التكميم"
        self.MODELS_PATH_REQUIRED = "مسار النماذج مطلوب"
        self.OUTPUT_PATH_REQUIRED = "مسار الإخراج مطلوب"
        self.LOGS_PATH_REQUIRED = "مسار السجلات مطلوب"
        self.STARTING_MODEL_QUANTIZATION = "بدء تكميم النموذج"
        self.INPUT_FILE_NOT_EXIST = "ملف الإدخال '{0}' غير موجود."
        self.QUANTIZING_MODEL_TO = "تكميم {0} إلى {1}"
        self.QUANTIZATION_TASK_STARTED = "بدأت مهمة تكميم {0}"
        self.ERROR_STARTING_QUANTIZATION = "خطأ في بدء التكميم: {0}"
        self.UPDATING_MODEL_INFO = "تحديث معلومات النموذج: {0}"
        self.TASK_FINISHED = "انتهت المهمة: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "عرض تفاصيل المهمة لـ: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "تصفح ملف بيانات مصفوفة الأهمية"
        self.SELECT_DATA_FILE = "اختر ملف البيانات"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "تصفح ملف نموذج مصفوفة الأهمية"
        self.SELECT_MODEL_FILE = "اختر ملف النموذج"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "تصفح ملف إخراج مصفوفة الأهمية"
        self.SELECT_OUTPUT_FILE = "اختر ملف الإخراج"
        self.STARTING_IMATRIX_GENERATION = "بدء توليد مصفوفة الأهمية"
        self.BACKEND_PATH_NOT_EXIST = "مسار الخلفية غير موجود: {0}"
        self.GENERATING_IMATRIX = "جاري توليد مصفوفة الأهمية"
        self.ERROR_STARTING_IMATRIX_GENERATION = "خطأ في بدء توليد مصفوفة الأهمية: {0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "بدأت مهمة توليد مصفوفة الأهمية"
        self.ERROR_MESSAGE = "خطأ: {0}"
        self.TASK_ERROR = "خطأ في المهمة: {0}"
        self.APPLICATION_CLOSING = "جاري إغلاق التطبيق"
        self.APPLICATION_CLOSED = "تم إغلاق التطبيق"
        self.SELECT_QUANTIZATION_TYPE = "اختر نوع التكميم"
        self.ALLOWS_REQUANTIZING = "يسمح بإعادة تكميم التنسورات التي تم تكميمها بالفعل"
        self.LEAVE_OUTPUT_WEIGHT = "سيترك output.weight بدون (إعادة) تكميم"
        self.DISABLE_K_QUANT_MIXTURES = (
            "تعطيل خلطات k-quant وتكميم جميع التنسورات إلى نفس النوع"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "استخدام البيانات في الملف كمصفوفة أهمية لتحسينات التكميم"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = "استخدام مصفوفة الأهمية لهذه التنسورات"
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "عدم استخدام مصفوفة الأهمية لهذه التنسورات"
        )
        self.OUTPUT_TENSOR_TYPE = "نوع تنسور الإخراج:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = "استخدام هذا النوع لتنسور output.weight"
        self.TOKEN_EMBEDDING_TYPE = "نوع تضمين الرمز:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "استخدام هذا النوع لتنسور تضمينات الرموز"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "سيولد النموذج المكمم في نفس الأجزاء كالإدخال"
        )
        self.OVERRIDE_MODEL_METADATA = "تجاوز بيانات تعريف النموذج"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "ملف بيانات الإدخال لتوليد مصفوفة الأهمية"
        self.MODEL_TO_BE_QUANTIZED = "النموذج المراد تكميمه"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "مسار الإخراج لمصفوفة الأهمية المولدة"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "عدد مرات حفظ مصفوفة الأهمية"
        self.SET_GPU_OFFLOAD_VALUE = "تعيين قيمة تحميل وحدة معالجة الرسومات (-ngl)"
        self.STARTING_LORA_CONVERSION = "بدء تحويل LoRA"
        self.LORA_INPUT_PATH_REQUIRED = "مسار إدخال LoRA مطلوب."
        self.LORA_OUTPUT_PATH_REQUIRED = "مسار إخراج LoRA مطلوب."
        self.ERROR_STARTING_LORA_CONVERSION = "خطأ في بدء تحويل LoRA: {}"
        self.LORA_CONVERSION_TASK_STARTED = "بدأت مهمة تحويل LoRA."
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = "تصفح مجلد إدخال LoRA..."
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "تصفح ملف إخراج LoRA..."
        self.CONVERTING_LORA = "تحويل LoRA"
        self.LORA_CONVERSION_FINISHED = "اكتمل تحويل LoRA."
        self.LORA_FILE_MOVED = "تم نقل ملف LoRA من {} إلى {}."
        self.LORA_FILE_NOT_FOUND = "لم يتم العثور على ملف LoRA: {}."
        self.ERROR_MOVING_LORA_FILE = "خطأ في نقل ملف LoRA: {}"
        self.MODEL_PATH_REQUIRED = "مسار النموذج مطلوب."
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = "مطلوب محول LoRA واحد على الأقل."
        self.INVALID_LORA_SCALE_VALUE = "قيمة مقياس LoRA غير صالحة."
        self.ERROR_STARTING_LORA_EXPORT = "خطأ في بدء تصدير LoRA: {}"
        self.LORA_EXPORT_TASK_STARTED = "بدأت مهمة تصدير LoRA."
        self.EXPORTING_LORA = "جاري تصدير LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = "تصفح ملف نموذج تصدير LoRA..."
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = "تصفح ملف إخراج تصدير LoRA..."
        self.ADDING_LORA_ADAPTER = "إضافة محول LoRA..."
        self.DELETING_LORA_ADAPTER = "حذف محول LoRA..."
        self.SELECT_LORA_ADAPTER_FILE = "اختر ملف محول LoRA"
        self.STARTING_LORA_EXPORT = "بدء تصدير LoRA..."
        self.SELECT_OUTPUT_TYPE = "اختر نوع الإخراج (GGUF أو GGML)"
        self.BASE_MODEL = "النموذج الأساسي"
        self.SELECT_BASE_MODEL_FILE = "اختر ملف النموذج الأساسي (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = "مسار النموذج الأساسي مطلوب لإخراج GGUF."
        self.BROWSING_FOR_BASE_MODEL_FILE = "تصفح ملف النموذج الأساسي..."
        self.SELECT_BASE_MODEL_FOLDER = (
            "اختر مجلد النموذج الأساسي (يحتوي على safetensors)"
        )
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "تصفح مجلد النموذج الأساسي..."
        self.LORA_CONVERSION_FROM_TO = "تحويل LoRA من {} إلى {}"
        self.GENERATING_IMATRIX_FOR = "توليد مصفوفة الأهمية لـ {}"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = (
            "مسار النموذج مطلوب لتوليد مصفوفة الأهمية."
        )
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = "لم يتم اختيار أي أصل للتحقق من CUDA"
        self.NO_QUANTIZATION_TYPE_SELECTED = (
            "لم يتم اختيار نوع تكميم. يرجى اختيار نوع تكميم واحد على الأقل."
        )
        self.STARTING_HF_TO_GGUF_CONVERSION = "بدء تحويل HuggingFace إلى GGUF"
        self.MODEL_DIRECTORY_REQUIRED = "مجلد النموذج مطلوب"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "أمر تحويل HF إلى GGUF: {}"
        self.CONVERTING_TO_GGUF = "تحويل {} إلى GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "خطأ في بدء تحويل HuggingFace إلى GGUF: {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = "بدأت مهمة تحويل HuggingFace إلى GGUF"


class _Korean(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (자동 GGUF 모델 양자화기)"
        self.RAM_USAGE = "RAM 사용량:"
        self.CPU_USAGE = "CPU 사용량:"
        self.BACKEND = "Llama.cpp 백엔드:"
        self.REFRESH_BACKENDS = "백엔드 새로 고침"
        self.MODELS_PATH = "모델 경로:"
        self.OUTPUT_PATH = "출력 경로:"
        self.LOGS_PATH = "로그 경로:"
        self.BROWSE = "찾아보기"
        self.AVAILABLE_MODELS = "사용 가능한 모델:"
        self.QUANTIZATION_TYPE = "양자화 유형:"
        self.ALLOW_REQUANTIZE = "재양자화 허용"
        self.LEAVE_OUTPUT_TENSOR = "출력 텐서 유지"
        self.PURE = "순수"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "가중치 포함:"
        self.EXCLUDE_WEIGHTS = "가중치 제외:"
        self.USE_OUTPUT_TENSOR_TYPE = "출력 텐서 유형 사용"
        self.USE_TOKEN_EMBEDDING_TYPE = "토큰 임베딩 유형 사용"
        self.KEEP_SPLIT = "분할 유지"
        self.KV_OVERRIDES = "KV 재정의:"
        self.ADD_NEW_OVERRIDE = "새 재정의 추가"
        self.QUANTIZE_MODEL = "모델 양자화"
        self.SAVE_PRESET = "프리셋 저장"
        self.LOAD_PRESET = "프리셋 로드"
        self.TASKS = "작업:"
        self.DOWNLOAD_LLAMACPP = "llama.cpp 다운로드"
        self.SELECT_RELEASE = "릴리스 선택:"
        self.SELECT_ASSET = "자산 선택:"
        self.EXTRACT_CUDA_FILES = "CUDA 파일 추출"
        self.SELECT_CUDA_BACKEND = "CUDA 백엔드 선택:"
        self.DOWNLOAD = "다운로드"
        self.IMATRIX_GENERATION = "IMatrix 생성"
        self.DATA_FILE = "데이터 파일:"
        self.MODEL = "모델:"
        self.OUTPUT = "출력:"
        self.OUTPUT_FREQUENCY = "출력 빈도:"
        self.GPU_OFFLOAD = "GPU 오프로드:"
        self.AUTO = "자동"
        self.GENERATE_IMATRIX = "IMatrix 생성"
        self.ERROR = "오류"
        self.WARNING = "경고"
        self.PROPERTIES = "속성"
        self.CANCEL = "취소"
        self.RESTART = "다시 시작"
        self.DELETE = "삭제"
        self.CONFIRM_DELETION = "이 작업을 삭제하시겠습니까?"
        self.TASK_RUNNING_WARNING = "일부 작업이 아직 실행 중입니다. 종료하시겠습니까?"
        self.YES = "예"
        self.NO = "아니요"
        self.DOWNLOAD_COMPLETE = "다운로드 완료"
        self.CUDA_EXTRACTION_FAILED = "CUDA 추출 실패"
        self.PRESET_SAVED = "프리셋 저장됨"
        self.PRESET_LOADED = "프리셋 로드됨"
        self.NO_ASSET_SELECTED = "자산이 선택되지 않았습니다"
        self.DOWNLOAD_FAILED = "다운로드 실패"
        self.NO_BACKEND_SELECTED = "백엔드가 선택되지 않았습니다"
        self.NO_MODEL_SELECTED = "모델이 선택되지 않았습니다"
        self.REFRESH_RELEASES = "릴리스 새로 고침"
        self.NO_SUITABLE_CUDA_BACKENDS = "적합한 CUDA 백엔드를 찾을 수 없습니다"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp 바이너리가 다운로드되어 {0}에 추출되었습니다.\nCUDA 파일이 {1}에 추출되었습니다."
        self.CUDA_FILES_EXTRACTED = "CUDA 파일이 에 추출되었습니다."
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "추출에 적합한 CUDA 백엔드를 찾을 수 없습니다."
        )
        self.ERROR_FETCHING_RELEASES = "릴리스를 가져오는 중 오류가 발생했습니다: {0}"
        self.CONFIRM_DELETION_TITLE = "삭제 확인"
        self.LOG_FOR = "{0}에 대한 로그"
        self.ALL_FILES = "모든 파일 (*)"
        self.GGUF_FILES = "GGUF 파일 (*.gguf)"
        self.DAT_FILES = "DAT 파일 (*.dat)"
        self.JSON_FILES = "JSON 파일 (*.json)"
        self.FAILED_LOAD_PRESET = "프리셋을 로드하지 못했습니다: {0}"
        self.INITIALIZING_AUTOGGUF = "AutoGGUF 애플리케이션을 초기화하는 중입니다."
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF 초기화가 완료되었습니다."
        self.REFRESHING_BACKENDS = "백엔드를 새로 고치는 중입니다."
        self.NO_BACKENDS_AVAILABLE = "사용 가능한 백엔드가 없습니다."
        self.FOUND_VALID_BACKENDS = "{0}개의 유효한 백엔드를 찾았습니다."
        self.SAVING_PRESET = "프리셋을 저장하는 중입니다."
        self.PRESET_SAVED_TO = "프리셋이 {0}에 저장되었습니다."
        self.LOADING_PRESET = "프리셋을 로드하는 중입니다."
        self.PRESET_LOADED_FROM = "{0}에서 프리셋을 로드했습니다."
        self.ADDING_KV_OVERRIDE = "KV 재정의를 추가하는 중입니다: {0}"
        self.SAVING_TASK_PRESET = "{0}에 대한 작업 프리셋을 저장하는 중입니다."
        self.TASK_PRESET_SAVED = "작업 프리셋이 저장되었습니다."
        self.TASK_PRESET_SAVED_TO = "작업 프리셋이 {0}에 저장되었습니다."
        self.RESTARTING_TASK = "작업을 다시 시작하는 중입니다: {0}"
        self.IN_PROGRESS = "진행 중"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "다운로드가 완료되었습니다. 추출 위치: {0}"
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp 바이너리가 다운로드되어 {0}에 추출되었습니다.\nCUDA 파일이 {1}에 추출되었습니다."
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "추출에 적합한 CUDA 백엔드를 찾을 수 없습니다."
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp 바이너리가 다운로드되어 {0}에 추출되었습니다."
        )
        self.REFRESHING_LLAMACPP_RELEASES = "llama.cpp 릴리스를 새로 고치는 중입니다."
        self.UPDATING_ASSET_LIST = "자산 목록을 업데이트하는 중입니다."
        self.UPDATING_CUDA_OPTIONS = "CUDA 옵션을 업데이트하는 중입니다."
        self.STARTING_LLAMACPP_DOWNLOAD = "llama.cpp 다운로드를 시작하는 중입니다."
        self.UPDATING_CUDA_BACKENDS = "CUDA 백엔드를 업데이트하는 중입니다."
        self.NO_CUDA_BACKEND_SELECTED = "추출에 CUDA 백엔드가 선택되지 않았습니다."
        self.EXTRACTING_CUDA_FILES = "{0}에서 {1}로 CUDA 파일을 추출하는 중입니다."
        self.DOWNLOAD_ERROR = "다운로드 오류: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "작업 컨텍스트 메뉴를 표시하는 중입니다."
        self.SHOWING_PROPERTIES_FOR_TASK = "작업에 대한 속성을 표시하는 중입니다: {0}"
        self.CANCELLING_TASK = "작업을 취소하는 중입니다: {0}"
        self.CANCELED = "취소됨"
        self.DELETING_TASK = "작업을 삭제하는 중입니다: {0}"
        self.LOADING_MODELS = "모델을 로드하는 중입니다."
        self.LOADED_MODELS = "{0}개의 모델이 로드되었습니다."
        self.BROWSING_FOR_MODELS_DIRECTORY = "모델 디렉토리를 찾아보는 중입니다."
        self.SELECT_MODELS_DIRECTORY = "모델 디렉토리 선택"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "출력 디렉토리를 찾아보는 중입니다."
        self.SELECT_OUTPUT_DIRECTORY = "출력 디렉토리 선택"
        self.BROWSING_FOR_LOGS_DIRECTORY = "로그 디렉토리를 찾아보는 중입니다."
        self.SELECT_LOGS_DIRECTORY = "로그 디렉토리 선택"
        self.BROWSING_FOR_IMATRIX_FILE = "IMatrix 파일을 찾아보는 중입니다."
        self.SELECT_IMATRIX_FILE = "IMatrix 파일 선택"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU 사용량: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "양자화 입력을 검증하는 중입니다."
        self.MODELS_PATH_REQUIRED = "모델 경로가 필요합니다."
        self.OUTPUT_PATH_REQUIRED = "출력 경로가 필요합니다."
        self.LOGS_PATH_REQUIRED = "로그 경로가 필요합니다."
        self.STARTING_MODEL_QUANTIZATION = "모델 양자화를 시작하는 중입니다."
        self.INPUT_FILE_NOT_EXIST = "입력 파일 '{0}'이 존재하지 않습니다."
        self.QUANTIZING_MODEL_TO = "{0}을 {1}(으)로 양자화하는 중입니다."
        self.QUANTIZATION_TASK_STARTED = "{0}에 대한 양자화 작업이 시작되었습니다."
        self.ERROR_STARTING_QUANTIZATION = (
            "양자화를 시작하는 중 오류가 발생했습니다: {0}"
        )
        self.UPDATING_MODEL_INFO = "모델 정보를 업데이트하는 중입니다: {0}"
        self.TASK_FINISHED = "작업이 완료되었습니다: {0}"
        self.SHOWING_TASK_DETAILS_FOR = (
            "다음에 대한 작업 세부 정보를 표시하는 중입니다: {0}"
        )
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "IMatrix 데이터 파일을 찾아보는 중입니다."
        self.SELECT_DATA_FILE = "데이터 파일 선택"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "IMatrix 모델 파일을 찾아보는 중입니다."
        self.SELECT_MODEL_FILE = "모델 파일 선택"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "IMatrix 출력 파일을 찾아보는 중입니다."
        self.SELECT_OUTPUT_FILE = "출력 파일 선택"
        self.STARTING_IMATRIX_GENERATION = "IMatrix 생성을 시작하는 중입니다."
        self.BACKEND_PATH_NOT_EXIST = "백엔드 경로가 존재하지 않습니다: {0}"
        self.GENERATING_IMATRIX = "IMatrix를 생성하는 중입니다."
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "IMatrix 생성을 시작하는 중 오류가 발생했습니다: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix 생성 작업이 시작되었습니다."
        self.ERROR_MESSAGE = "오류: {0}"
        self.TASK_ERROR = "작업 오류: {0}"
        self.APPLICATION_CLOSING = "애플리케이션을 닫는 중입니다."
        self.APPLICATION_CLOSED = "애플리케이션이 닫혔습니다."
        self.SELECT_QUANTIZATION_TYPE = "양자화 유형을 선택하세요."
        self.ALLOWS_REQUANTIZING = "이미 양자화된 텐서의 재양자화를 허용합니다."
        self.LEAVE_OUTPUT_WEIGHT = "output.weight를 (재)양자화하지 않은 상태로 둡니다."
        self.DISABLE_K_QUANT_MIXTURES = (
            "k-양자 혼합을 비활성화하고 모든 텐서를 동일한 유형으로 양자화합니다."
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "양자 최적화를 위한 중요도 행렬로 파일의 데이터를 사용합니다."
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "이러한 텐서에 중요도 행렬을 사용합니다."
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "이러한 텐서에 중요도 행렬을 사용하지 않습니다."
        )
        self.OUTPUT_TENSOR_TYPE = "출력 텐서 유형:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "output.weight 텐서에 이 유형을 사용합니다."
        )
        self.TOKEN_EMBEDDING_TYPE = "토큰 임베딩 유형:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "토큰 임베딩 텐서에 이 유형을 사용합니다."
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "입력과 동일한 샤드에 양자화된 모델을 생성합니다."
        )
        self.OVERRIDE_MODEL_METADATA = "모델 메타데이터를 재정의합니다."
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix 생성을 위한 입력 데이터 파일"
        self.MODEL_TO_BE_QUANTIZED = "양자화될 모델"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "생성된 IMatrix의 출력 경로"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "IMatrix를 저장할 빈도"
        self.SET_GPU_OFFLOAD_VALUE = "GPU 오프로드 값 설정 (-ngl)"
        self.COMPLETED = "완료됨"
        self.REFRESH_MODELS = "모델 새로고침"


class _Italian(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Quantizzatore Automatico di Modelli GGUF)"
        self.RAM_USAGE = "Utilizzo RAM:"
        self.CPU_USAGE = "Utilizzo CPU:"
        self.BACKEND = "Backend Llama.cpp:"
        self.REFRESH_BACKENDS = "Aggiorna Backend"
        self.MODELS_PATH = "Percorso Modelli:"
        self.OUTPUT_PATH = "Percorso Output:"
        self.LOGS_PATH = "Percorso Log:"
        self.BROWSE = "Sfoglia"
        self.AVAILABLE_MODELS = "Modelli Disponibili:"
        self.QUANTIZATION_TYPE = "Tipo di Quantizzazione:"
        self.ALLOW_REQUANTIZE = "Consenti Riquantizzazione"
        self.LEAVE_OUTPUT_TENSOR = "Lascia Tensore di Output"
        self.PURE = "Puro"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Includi Pesi:"
        self.EXCLUDE_WEIGHTS = "Escludi Pesi:"
        self.USE_OUTPUT_TENSOR_TYPE = "Usa Tipo di Tensore di Output"
        self.USE_TOKEN_EMBEDDING_TYPE = "Usa Tipo di Incorporamento Token"
        self.KEEP_SPLIT = "Mantieni Divisione"
        self.KV_OVERRIDES = "Override KV:"
        self.ADD_NEW_OVERRIDE = "Aggiungi Nuovo Override"
        self.QUANTIZE_MODEL = "Quantizza Modello"
        self.SAVE_PRESET = "Salva Preimpostazione"
        self.LOAD_PRESET = "Carica Preimpostazione"
        self.TASKS = "Attività:"
        self.DOWNLOAD_LLAMACPP = "Scarica llama.cpp"
        self.SELECT_RELEASE = "Seleziona Versione:"
        self.SELECT_ASSET = "Seleziona Asset:"
        self.EXTRACT_CUDA_FILES = "Estrai File CUDA"
        self.SELECT_CUDA_BACKEND = "Seleziona Backend CUDA:"
        self.DOWNLOAD = "Scarica"
        self.IMATRIX_GENERATION = "Generazione IMatrix"
        self.DATA_FILE = "File Dati:"
        self.MODEL = "Modello:"
        self.OUTPUT = "Output:"
        self.OUTPUT_FREQUENCY = "Frequenza di Output:"
        self.GPU_OFFLOAD = "Offload GPU:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Genera IMatrix"
        self.ERROR = "Errore"
        self.WARNING = "Avviso"
        self.PROPERTIES = "Proprietà"
        self.CANCEL = "Annulla"
        self.RESTART = "Riavvia"
        self.DELETE = "Elimina"
        self.CONFIRM_DELETION = "Sei sicuro di voler eliminare questa attività?"
        self.TASK_RUNNING_WARNING = (
            "Alcune attività sono ancora in esecuzione. Sei sicuro di voler uscire?"
        )
        self.YES = "Sì"
        self.NO = "No"
        self.DOWNLOAD_COMPLETE = "Download Completato"
        self.CUDA_EXTRACTION_FAILED = "Estrazione CUDA Fallita"
        self.PRESET_SAVED = "Preimpostazione Salvata"
        self.PRESET_LOADED = "Preimpostazione Caricata"
        self.NO_ASSET_SELECTED = "Nessun asset selezionato"
        self.DOWNLOAD_FAILED = "Download fallito"
        self.NO_BACKEND_SELECTED = "Nessun backend selezionato"
        self.NO_MODEL_SELECTED = "Nessun modello selezionato"
        self.REFRESH_RELEASES = "Aggiorna Versioni"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nessun backend CUDA adatto trovato"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "Binario llama.cpp scaricato ed estratto in {0}\nFile CUDA estratti in {1}"
        )
        self.CUDA_FILES_EXTRACTED = "File CUDA estratti in"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nessun backend CUDA adatto trovato per l'estrazione"
        )
        self.ERROR_FETCHING_RELEASES = "Errore durante il recupero delle versioni: {0}"
        self.CONFIRM_DELETION_TITLE = "Conferma Eliminazione"
        self.LOG_FOR = "Log per {0}"
        self.ALL_FILES = "Tutti i File (*)"
        self.GGUF_FILES = "File GGUF (*.gguf)"
        self.DAT_FILES = "File DAT (*.dat)"
        self.JSON_FILES = "File JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Impossibile caricare la preimpostazione: {0}"
        self.INITIALIZING_AUTOGGUF = "Inizializzazione dell'applicazione AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = (
            "Inizializzazione di AutoGGUF completata"
        )
        self.REFRESHING_BACKENDS = "Aggiornamento backend"
        self.NO_BACKENDS_AVAILABLE = "Nessun backend disponibile"
        self.FOUND_VALID_BACKENDS = "Trovati {0} backend validi"
        self.SAVING_PRESET = "Salvataggio preimpostazione"
        self.PRESET_SAVED_TO = "Preimpostazione salvata in {0}"
        self.LOADING_PRESET = "Caricamento preimpostazione"
        self.PRESET_LOADED_FROM = "Preimpostazione caricata da {0}"
        self.ADDING_KV_OVERRIDE = "Aggiunta override KV: {0}"
        self.SAVING_TASK_PRESET = "Salvataggio preimpostazione attività per {0}"
        self.TASK_PRESET_SAVED = "Preimpostazione Attività Salvata"
        self.TASK_PRESET_SAVED_TO = "Preimpostazione attività salvata in {0}"
        self.RESTARTING_TASK = "Riavvio attività: {0}"
        self.IN_PROGRESS = "In Corso"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download completato. Estratto in: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "Binario llama.cpp scaricato ed estratto in {0}\nFile CUDA estratti in {1}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nessun backend CUDA adatto trovato per l'estrazione"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Binario llama.cpp scaricato ed estratto in {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Aggiornamento versioni di llama.cpp"
        self.UPDATING_ASSET_LIST = "Aggiornamento elenco asset"
        self.UPDATING_CUDA_OPTIONS = "Aggiornamento opzioni CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Avvio download di llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Aggiornamento backend CUDA"
        self.NO_CUDA_BACKEND_SELECTED = (
            "Nessun backend CUDA selezionato per l'estrazione"
        )
        self.EXTRACTING_CUDA_FILES = "Estrazione file CUDA da {0} a {1}"
        self.DOWNLOAD_ERROR = "Errore di download: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Visualizzazione menu contestuale attività"
        self.SHOWING_PROPERTIES_FOR_TASK = (
            "Visualizzazione proprietà per l'attività: {0}"
        )
        self.CANCELLING_TASK = "Annullamento attività: {0}"
        self.CANCELED = "Annullato"
        self.DELETING_TASK = "Eliminazione attività: {0}"
        self.LOADING_MODELS = "Caricamento modelli"
        self.LOADED_MODELS = "{0} modelli caricati"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Esplorazione directory modelli"
        self.SELECT_MODELS_DIRECTORY = "Seleziona Directory Modelli"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Esplorazione directory output"
        self.SELECT_OUTPUT_DIRECTORY = "Seleziona Directory Output"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Esplorazione directory log"
        self.SELECT_LOGS_DIRECTORY = "Seleziona Directory Log"
        self.BROWSING_FOR_IMATRIX_FILE = "Esplorazione file IMatrix"
        self.SELECT_IMATRIX_FILE = "Seleziona File IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Utilizzo CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Convalida input di quantizzazione"
        self.MODELS_PATH_REQUIRED = "Il percorso dei modelli è obbligatorio"
        self.OUTPUT_PATH_REQUIRED = "Il percorso di output è obbligatorio"
        self.LOGS_PATH_REQUIRED = "Il percorso dei log è obbligatorio"
        self.STARTING_MODEL_QUANTIZATION = "Avvio quantizzazione del modello"
        self.INPUT_FILE_NOT_EXIST = "Il file di input '{0}' non esiste."
        self.QUANTIZING_MODEL_TO = "Quantizzazione di {0} a {1}"
        self.QUANTIZATION_TASK_STARTED = "Attività di quantizzazione avviata per {0}"
        self.ERROR_STARTING_QUANTIZATION = (
            "Errore durante l'avvio della quantizzazione: {0}"
        )
        self.UPDATING_MODEL_INFO = "Aggiornamento informazioni sul modello: {0}"
        self.TASK_FINISHED = "Attività completata: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Visualizzazione dettagli attività per: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Esplorazione file dati IMatrix"
        self.SELECT_DATA_FILE = "Seleziona File Dati"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Esplorazione file modello IMatrix"
        self.SELECT_MODEL_FILE = "Seleziona File Modello"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Esplorazione file output IMatrix"
        self.SELECT_OUTPUT_FILE = "Seleziona File Output"
        self.STARTING_IMATRIX_GENERATION = "Avvio generazione IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Il percorso del backend non esiste: {0}"
        self.GENERATING_IMATRIX = "Generazione IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Errore durante l'avvio della generazione di IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Attività di generazione IMatrix avviata"
        self.ERROR_MESSAGE = "Errore: {0}"
        self.TASK_ERROR = "Errore attività: {0}"
        self.APPLICATION_CLOSING = "Chiusura applicazione"
        self.APPLICATION_CLOSED = "Applicazione chiusa"
        self.SELECT_QUANTIZATION_TYPE = "Seleziona il tipo di quantizzazione"
        self.ALLOWS_REQUANTIZING = (
            "Consente di riquantizzare tensori che sono già stati quantizzati"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Lascerà output.weight non (ri)quantizzato"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Disabilita le miscele k-quant e quantizza tutti i tensori allo stesso tipo"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Utilizza i dati nel file come matrice di importanza per le ottimizzazioni di quantizzazione"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Usa la matrice di importanza per questi tensori"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Non usare la matrice di importanza per questi tensori"
        )
        self.OUTPUT_TENSOR_TYPE = "Tipo di Tensore di Output:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Usa questo tipo per il tensore output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Tipo di Incorporamento Token:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Usa questo tipo per il tensore di incorporamenti token"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Genererà il modello quantizzato negli stessi frammenti dell'input"
        )
        self.OVERRIDE_MODEL_METADATA = "Sovrascrivi i metadati del modello"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "File di dati di input per la generazione di IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modello da quantizzare"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Percorso di output per l'IMatrix generato"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Con quale frequenza salvare l'IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Imposta il valore di offload GPU (-ngl)"
        self.COMPLETED = "Completato"
        self.REFRESH_MODELS = "Aggiorna modelli"


class _Turkish(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Otomatik GGUF Modeli Niceleyici)"
        self.RAM_USAGE = "RAM Kullanımı:"
        self.CPU_USAGE = "CPU Kullanımı:"
        self.BACKEND = "Llama.cpp Arka Uç:"
        self.REFRESH_BACKENDS = "Arka Uçları Yenile"
        self.MODELS_PATH = "Modeller Yolu:"
        self.OUTPUT_PATH = "Çıkış Yolu:"
        self.LOGS_PATH = "Günlükler Yolu:"
        self.BROWSE = "Gözat"
        self.AVAILABLE_MODELS = "Kullanılabilir Modeller:"
        self.QUANTIZATION_TYPE = "Niceleme Türü:"
        self.ALLOW_REQUANTIZE = "Yeniden Nicelemeye İzin Ver"
        self.LEAVE_OUTPUT_TENSOR = "Çıkış Tensörünü Bırak"
        self.PURE = "Saf"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Ağırlıkları Dahil Et:"
        self.EXCLUDE_WEIGHTS = "Ağırlıkları Hariç Tut:"
        self.USE_OUTPUT_TENSOR_TYPE = "Çıkış Tensör Türünü Kullan"
        self.USE_TOKEN_EMBEDDING_TYPE = "Token Gömme Türünü Kullan"
        self.KEEP_SPLIT = "Bölmeyi Koru"
        self.KV_OVERRIDES = "KV Geçersiz Kılmaları:"
        self.ADD_NEW_OVERRIDE = "Yeni Geçersiz Kılma Ekle"
        self.QUANTIZE_MODEL = "Modeli Nicele"
        self.SAVE_PRESET = "Ön Ayarı Kaydet"
        self.LOAD_PRESET = "Ön Ayarı Yükle"
        self.TASKS = "Görevler:"
        self.DOWNLOAD_LLAMACPP = "llama.cpp'yi İndir"
        self.SELECT_RELEASE = "Sürümü Seç:"
        self.SELECT_ASSET = "Varlığı Seç:"
        self.EXTRACT_CUDA_FILES = "CUDA Dosyalarını Çıkar"
        self.SELECT_CUDA_BACKEND = "CUDA Arka Ucunu Seç:"
        self.DOWNLOAD = "İndir"
        self.IMATRIX_GENERATION = "IMatrix Üretimi"
        self.DATA_FILE = "Veri Dosyası:"
        self.MODEL = "Model:"
        self.OUTPUT = "Çıkış:"
        self.OUTPUT_FREQUENCY = "Çıkış Sıklığı:"
        self.GPU_OFFLOAD = "GPU Yük Boşaltma:"
        self.AUTO = "Otomatik"
        self.GENERATE_IMATRIX = "IMatrix Oluştur"
        self.ERROR = "Hata"
        self.WARNING = "Uyarı"
        self.PROPERTIES = "Özellikler"
        self.CANCEL = "İptal"
        self.RESTART = "Yeniden Başlat"
        self.DELETE = "Sil"
        self.CONFIRM_DELETION = "Bu görevi silmek istediğinizden emin misiniz?"
        self.TASK_RUNNING_WARNING = (
            "Bazı görevler hala çalışıyor. Çıkmak istediğinizden emin misiniz?"
        )
        self.YES = "Evet"
        self.NO = "Hayır"
        self.DOWNLOAD_COMPLETE = "İndirme Tamamlandı"
        self.CUDA_EXTRACTION_FAILED = "CUDA Çıkarma Başarısız"
        self.PRESET_SAVED = "Ön Ayar Kaydedildi"
        self.PRESET_LOADED = "Ön Ayar Yüklendi"
        self.NO_ASSET_SELECTED = "Varlık seçilmedi"
        self.DOWNLOAD_FAILED = "İndirme başarısız"
        self.NO_BACKEND_SELECTED = "Arka uç seçilmedi"
        self.NO_MODEL_SELECTED = "Model seçilmedi"
        self.REFRESH_RELEASES = "Sürümleri Yenile"
        self.NO_SUITABLE_CUDA_BACKENDS = "Uygun CUDA arka uçları bulunamadı"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp ikili dosyası indirildi ve {0} konumuna çıkarıldı\nCUDA dosyaları {1} konumuna çıkarıldı"
        self.CUDA_FILES_EXTRACTED = "CUDA dosyaları konumuna çıkarıldı"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Çıkarma için uygun bir CUDA arka ucu bulunamadı"
        )
        self.ERROR_FETCHING_RELEASES = "Sürümleri getirirken hata oluştu: {0}"
        self.CONFIRM_DELETION_TITLE = "Silmeyi Onayla"
        self.LOG_FOR = "{0} için Günlük"
        self.ALL_FILES = "Tüm Dosyalar (*)"
        self.GGUF_FILES = "GGUF Dosyaları (*.gguf)"
        self.DAT_FILES = "DAT Dosyaları (*.dat)"
        self.JSON_FILES = "JSON Dosyaları (*.json)"
        self.FAILED_LOAD_PRESET = "Ön ayarı yükleme başarısız: {0}"
        self.INITIALIZING_AUTOGGUF = "AutoGGUF uygulaması başlatılıyor"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF başlatması tamamlandı"
        self.REFRESHING_BACKENDS = "Arka uçlar yenileniyor"
        self.NO_BACKENDS_AVAILABLE = "Kullanılabilir arka uç yok"
        self.FOUND_VALID_BACKENDS = "{0} geçerli arka uç bulundu"
        self.SAVING_PRESET = "Ön ayar kaydediliyor"
        self.PRESET_SAVED_TO = "Ön ayar {0} konumuna kaydedildi"
        self.LOADING_PRESET = "Ön ayar yükleniyor"
        self.PRESET_LOADED_FROM = "Ön ayar {0} konumundan yüklendi"
        self.ADDING_KV_OVERRIDE = "KV geçersiz kılma ekleniyor: {0}"
        self.SAVING_TASK_PRESET = "{0} için görev ön ayarı kaydediliyor"
        self.TASK_PRESET_SAVED = "Görev Ön Ayarı Kaydedildi"
        self.TASK_PRESET_SAVED_TO = "Görev ön ayarı {0} konumuna kaydedildi"
        self.RESTARTING_TASK = "Görev yeniden başlatılıyor: {0}"
        self.IN_PROGRESS = "Devam Ediyor"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "İndirme tamamlandı. Şuraya çıkarıldı: {0}"
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp ikili dosyası indirildi ve {0} konumuna çıkarıldı\nCUDA dosyaları {1} konumuna çıkarıldı"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Çıkarma için uygun bir CUDA arka ucu bulunamadı"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp ikili dosyası indirildi ve {0} konumuna çıkarıldı"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "llama.cpp sürümleri yenileniyor"
        self.UPDATING_ASSET_LIST = "Varlık listesi güncelleniyor"
        self.UPDATING_CUDA_OPTIONS = "CUDA seçenekleri güncelleniyor"
        self.STARTING_LLAMACPP_DOWNLOAD = "llama.cpp indirme başlatılıyor"
        self.UPDATING_CUDA_BACKENDS = "CUDA arka uçları güncelleniyor"
        self.NO_CUDA_BACKEND_SELECTED = "Çıkarma için CUDA arka ucu seçilmedi"
        self.EXTRACTING_CUDA_FILES = (
            "CUDA dosyaları {0} konumundan {1} konumuna çıkarılıyor"
        )
        self.DOWNLOAD_ERROR = "İndirme hatası: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Görev bağlam menüsü gösteriliyor"
        self.SHOWING_PROPERTIES_FOR_TASK = "Görev için özellikler gösteriliyor: {0}"
        self.CANCELLING_TASK = "Görev iptal ediliyor: {0}"
        self.CANCELED = "İptal Edildi"
        self.DELETING_TASK = "Görev siliniyor: {0}"
        self.LOADING_MODELS = "Modeller yükleniyor"
        self.LOADED_MODELS = "{0} model yüklendi"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Modeller dizinine göz atılıyor"
        self.SELECT_MODELS_DIRECTORY = "Modeller Dizini Seç"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Çıkış dizinine göz atılıyor"
        self.SELECT_OUTPUT_DIRECTORY = "Çıkış Dizini Seç"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Günlükler dizinine göz atılıyor"
        self.SELECT_LOGS_DIRECTORY = "Günlükler Dizini Seç"
        self.BROWSING_FOR_IMATRIX_FILE = "IMatrix dosyasına göz atılıyor"
        self.SELECT_IMATRIX_FILE = "IMatrix Dosyası Seç"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU Kullanımı: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Niceleme girişleri doğrulanıyor"
        self.MODELS_PATH_REQUIRED = "Modeller yolu gerekli"
        self.OUTPUT_PATH_REQUIRED = "Çıkış yolu gerekli"
        self.LOGS_PATH_REQUIRED = "Günlükler yolu gerekli"
        self.STARTING_MODEL_QUANTIZATION = "Model niceleme başlatılıyor"
        self.INPUT_FILE_NOT_EXIST = "Giriş dosyası '{0}' mevcut değil."
        self.QUANTIZING_MODEL_TO = "{0} öğesini {1} öğesine niceleme"
        self.QUANTIZATION_TASK_STARTED = "{0} için niceleme görevi başlatıldı"
        self.ERROR_STARTING_QUANTIZATION = "Niceleme başlatılırken hata oluştu: {0}"
        self.UPDATING_MODEL_INFO = "Model bilgileri güncelleniyor: {0}"
        self.TASK_FINISHED = "Görev tamamlandı: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Şunun için görev ayrıntıları gösteriliyor: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "IMatrix veri dosyasına göz atılıyor"
        self.SELECT_DATA_FILE = "Veri Dosyası Seç"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "IMatrix model dosyasına göz atılıyor"
        self.SELECT_MODEL_FILE = "Model Dosyası Seç"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "IMatrix çıkış dosyasına göz atılıyor"
        self.SELECT_OUTPUT_FILE = "Çıkış Dosyası Seç"
        self.STARTING_IMATRIX_GENERATION = "IMatrix üretimi başlatılıyor"
        self.BACKEND_PATH_NOT_EXIST = "Arka uç yolu mevcut değil: {0}"
        self.GENERATING_IMATRIX = "IMatrix oluşturuluyor"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "IMatrix üretimi başlatılırken hata oluştu: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix oluşturma görevi başlatıldı"
        self.ERROR_MESSAGE = "Hata: {0}"
        self.TASK_ERROR = "Görev hatası: {0}"
        self.APPLICATION_CLOSING = "Uygulama kapatılıyor"
        self.APPLICATION_CLOSED = "Uygulama kapatıldı"
        self.SELECT_QUANTIZATION_TYPE = "Niceleme türünü seçin"
        self.ALLOWS_REQUANTIZING = (
            "Zaten niceleme yapılmış tensörlerin yeniden nicelemesine izin verir"
        )
        self.LEAVE_OUTPUT_WEIGHT = (
            "output.weight öğesini (yeniden) nicelememiş halde bırakır"
        )
        self.DISABLE_K_QUANT_MIXTURES = "k-Quant karışımlarını devre dışı bırakın ve tüm tensörleri aynı türe niceleyin"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Quant optimizasyonları için dosyadaki verileri önem matrisi olarak kullanın"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Bu tensörler için önem matrisini kullanın"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Bu tensörler için önem matrisini kullanmayın"
        )
        self.OUTPUT_TENSOR_TYPE = "Çıkış Tensör Türü:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "output.weight tensörü için bu türü kullanın"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token Gömme Türü:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Token gömme tensörü için bu türü kullanın"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Nicelemeli modeli girişle aynı parçalarda oluşturacaktır"
        )
        self.OVERRIDE_MODEL_METADATA = "Model meta verilerini geçersiz kıl"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix oluşturma için giriş veri dosyası"
        self.MODEL_TO_BE_QUANTIZED = "Nicelemeli model"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Oluşturulan IMatrix için çıkış yolu"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "IMatrix'in ne sıklıkta kaydedileceği"
        self.SET_GPU_OFFLOAD_VALUE = "GPU yük boşaltma değerini ayarla (-ngl)"
        self.COMPLETED = "Tamamlandı"
        self.REFRESH_MODELS = "Modelleri yenile"


class _Dutch(_Localization):
    def __init__(self):
        super().__init__()

        # Algemene gebruikersinterface
        self.WINDOW_TITLE = "AutoGGUF (geautomatiseerde GGUF-modelkwantisering)"
        self.RAM_USAGE = "RAM-gebruik:"
        self.CPU_USAGE = "CPU-gebruik:"
        self.BACKEND = "Llama.cpp Backend:"
        self.REFRESH_BACKENDS = "Backends vernieuwen"
        self.MODELS_PATH = "Modelpad:"
        self.OUTPUT_PATH = "Uitvoerpad:"
        self.LOGS_PATH = "Logboekpad:"
        self.BROWSE = "Bladeren"
        self.AVAILABLE_MODELS = "Beschikbare modellen:"
        self.REFRESH_MODELS = "Modellen vernieuwen"
        self.STARTUP_ELASPED_TIME = "Initialisatie duurde {0} ms"

        # Omgevingsvariabelen
        self.DOTENV_FILE_NOT_FOUND = ".env-bestand niet gevonden."
        self.COULD_NOT_PARSE_LINE = "Kan regel niet parseren: {0}"
        self.ERROR_LOADING_DOTENV = "Fout bij laden van .env: {0}"

        # Model importeren
        self.IMPORT_MODEL = "Model importeren"
        self.SELECT_MODEL_TO_IMPORT = "Selecteer model om te importeren"
        self.CONFIRM_IMPORT = "Import bevestigen"
        self.IMPORT_MODEL_CONFIRMATION = "Wilt u het model {} importeren?"
        self.MODEL_IMPORTED_SUCCESSFULLY = "Model {} succesvol geïmporteerd"
        self.IMPORTING_MODEL = "Model importeren"
        self.IMPORTED_MODEL_TOOLTIP = "Geïmporteerd model: {}"

        # AutoFP8-kwantisering
        self.AUTOFP8_QUANTIZATION_TASK_STARTED = "AutoFP8-kwantiseringstaak gestart"
        self.ERROR_STARTING_AUTOFP8_QUANTIZATION = (
            "Fout bij starten van AutoFP8-kwantisering"
        )
        self.QUANTIZING_WITH_AUTOFP8 = "Kwantiseren van {0} met AutoFP8"
        self.QUANTIZING_TO_WITH_AUTOFP8 = "Kwantiseren van {0} naar {1}"
        self.QUANTIZE_TO_FP8_DYNAMIC = "Kwantiseren naar FP8 Dynamic"
        self.OPEN_MODEL_FOLDER = "Modelmap openen"
        self.QUANTIZE = "Kwantiseren"
        self.OPEN_MODEL_FOLDER = "Modelmap openen"
        self.INPUT_MODEL = "Invoermodel:"

        # GGUF-verificatie
        self.INVALID_GGUF_FILE = "Ongeldig GGUF-bestand: {}"
        self.SHARDED_MODEL_NAME = "{} (Geshard)"
        self.IMPORTED_MODEL_TOOLTIP = "Geïmporteerd model: {}"
        self.CONCATENATED_FILE_WARNING = "Dit is een samengevoegd bestandsonderdeel. Het werkt niet met llama-quantize; voeg het bestand eerst samen."
        self.CONCATENATED_FILES_FOUND = "{} samengevoegde bestandsonderdelen gevonden. Voeg de bestanden eerst samen."

        # Plugins
        self.PLUGINS_DIR_NOT_EXIST = (
            "Pluginmap '{}' bestaat niet. Er worden geen plugins geladen."
        )
        self.PLUGINS_DIR_NOT_DIRECTORY = (
            "'{}' bestaat, maar is geen map. Er worden geen plugins geladen."
        )
        self.PLUGIN_LOADED = "Plugin geladen: {} {}"
        self.PLUGIN_INCOMPATIBLE = "Plugin {} {} is niet compatibel met AutoGGUF versie {}. Ondersteunde versies: {}"
        self.PLUGIN_LOAD_FAILED = "Kan plugin {} niet laden: {}"
        self.NO_PLUGINS_LOADED = "Geen plugins geladen."

        # GPU-monitoring
        self.GPU_USAGE = "GPU-gebruik:"
        self.GPU_USAGE_FORMAT = "GPU: {:.1f}% | VRAM: {:.1f}% ({} MB / {} MB)"
        self.GPU_DETAILS = "GPU-details"
        self.GPU_USAGE_OVER_TIME = "GPU-gebruik in de tijd"
        self.VRAM_USAGE_OVER_TIME = "VRAM-gebruik in de tijd"
        self.PERCENTAGE = "Percentage"
        self.TIME = "Tijd (s)"
        self.NO_GPU_DETECTED = "Geen GPU gedetecteerd"
        self.SELECT_GPU = "Selecteer GPU"
        self.AMD_GPU_NOT_SUPPORTED = "AMD GPU gedetecteerd, maar niet ondersteund"

        # Kwantisering
        self.QUANTIZATION_TYPE = "Kwantiseringstype:"
        self.ALLOW_REQUANTIZE = "Herkwantisering toestaan"
        self.LEAVE_OUTPUT_TENSOR = "Uitvoertensor behouden"
        self.PURE = "Zuiver"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Gewichten opnemen:"
        self.EXCLUDE_WEIGHTS = "Gewichten uitsluiten:"
        self.USE_OUTPUT_TENSOR_TYPE = "Uitvoertensortype gebruiken"
        self.USE_TOKEN_EMBEDDING_TYPE = "Token Embedding-type gebruiken"
        self.KEEP_SPLIT = "Splitsing behouden"
        self.KV_OVERRIDES = "KV-overschrijvingen:"
        self.ADD_NEW_OVERRIDE = "Nieuwe overschrijving toevoegen"
        self.QUANTIZE_MODEL = "Model kwantiseren"
        self.EXTRA_ARGUMENTS = "Extra argumenten:"
        self.EXTRA_ARGUMENTS_LABEL = "Aanvullende opdrachtregelargumenten"
        self.QUANTIZATION_COMMAND = "Kwantiseringsopdracht"

        # Voorinstellingen
        self.SAVE_PRESET = "Voorinstelling opslaan"
        self.LOAD_PRESET = "Voorinstelling laden"

        # Taken
        self.TASKS = "Taken:"

        # llama.cpp downloaden
        self.DOWNLOAD_LLAMACPP = "Download llama.cpp"
        self.SELECT_RELEASE = "Selecteer release:"
        self.SELECT_ASSET = "Selecteer asset:"
        self.EXTRACT_CUDA_FILES = "CUDA-bestanden uitpakken"
        self.SELECT_CUDA_BACKEND = "Selecteer CUDA-backend:"
        self.DOWNLOAD = "Downloaden"
        self.REFRESH_RELEASES = "Releases vernieuwen"

        # IMatrix-generatie
        self.IMATRIX_GENERATION = "IMatrix-generatie"
        self.DATA_FILE = "Gegevensbestand:"
        self.MODEL = "Model:"
        self.OUTPUT = "Uitvoer:"
        self.OUTPUT_FREQUENCY = "Uitvoerfrequentie:"
        self.GPU_OFFLOAD = "GPU-offload:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "IMatrix genereren"
        self.CONTEXT_SIZE = "Contextgrootte:"
        self.CONTEXT_SIZE_FOR_IMATRIX = "Contextgrootte voor IMatrix-generatie"
        self.THREADS = "Threads:"
        self.NUMBER_OF_THREADS_FOR_IMATRIX = "Aantal threads voor IMatrix-generatie"
        self.IMATRIX_GENERATION_COMMAND = "IMatrix-generatieopdracht"

        # LoRA-conversie
        self.LORA_CONVERSION = "LoRA-conversie"
        self.LORA_INPUT_PATH = "LoRA-invoerpad"
        self.LORA_OUTPUT_PATH = "LoRA-uitvoerpad"
        self.SELECT_LORA_INPUT_DIRECTORY = "Selecteer LoRA-invoermap"
        self.SELECT_LORA_OUTPUT_FILE = "Selecteer LoRA-uitvoerbestand"
        self.CONVERT_LORA = "LoRA converteren"
        self.LORA_CONVERSION_COMMAND = "LoRA-conversieopdracht"

        # LoRA-export
        self.EXPORT_LORA = "LoRA exporteren"
        self.GGML_LORA_ADAPTERS = "GGML LoRA-adapters"
        self.SELECT_LORA_ADAPTER_FILES = "Selecteer LoRA-adapterbestanden"
        self.ADD_ADAPTER = "Adapter toevoegen"
        self.DELETE_ADAPTER = "Verwijderen"
        self.LORA_SCALE = "LoRA-schaal"
        self.ENTER_LORA_SCALE_VALUE = "Voer LoRA-schaalwaarde in (optioneel)"
        self.NUMBER_OF_THREADS_FOR_LORA_EXPORT = "Aantal threads voor LoRA-export"
        self.LORA_EXPORT_COMMAND = "LoRA-exportopdracht"

        # HuggingFace naar GGUF-conversie
        self.HF_TO_GGUF_CONVERSION = "HuggingFace naar GGUF-conversie"
        self.MODEL_DIRECTORY = "Modelmap:"
        self.OUTPUT_FILE = "Uitvoerbestand:"
        self.OUTPUT_TYPE = "Uitvoertype:"
        self.VOCAB_ONLY = "Alleen vocabulaire"
        self.USE_TEMP_FILE = "Tijdelijk bestand gebruiken"
        self.NO_LAZY_EVALUATION = "Geen luie evaluatie"
        self.MODEL_NAME = "Modelnaam:"
        self.VERBOSE = "Uitgebreid"
        self.SPLIT_MAX_SIZE = "Maximale splitsingsgrootte:"
        self.DRY_RUN = "Droog uitvoeren"
        self.CONVERT_HF_TO_GGUF = "HF naar GGUF converteren"
        self.SELECT_HF_MODEL_DIRECTORY = "Selecteer HuggingFace-modelmap"
        self.BROWSE_FOR_HF_MODEL_DIRECTORY = "Bladeren naar HuggingFace-modelmap"
        self.BROWSE_FOR_HF_TO_GGUF_OUTPUT = (
            "Bladeren naar HuggingFace naar GGUF-uitvoerbestand"
        )

        # Update controleren
        self.UPDATE_AVAILABLE = "Update beschikbaar"
        self.NEW_VERSION_AVAILABLE = "Er is een nieuwe versie beschikbaar: {}"
        self.DOWNLOAD_NEW_VERSION = "Downloaden?"
        self.ERROR_CHECKING_FOR_UPDATES = "Fout bij controleren op updates:"
        self.CHECKING_FOR_UPDATES = "Controleren op updates"

        # Algemene berichten
        self.ERROR = "Fout"
        self.WARNING = "Waarschuwing"
        self.PROPERTIES = "Eigenschappen"
        self.CANCEL = "Annuleren"
        self.RESTART = "Opnieuw starten"
        self.DELETE = "Verwijderen"
        self.CONFIRM_DELETION = "Weet u zeker dat u deze taak wilt verwijderen?"
        self.TASK_RUNNING_WARNING = (
            "Sommige taken zijn nog steeds actief. Weet u zeker dat u wilt afsluiten?"
        )
        self.YES = "Ja"
        self.NO = "Nee"
        self.COMPLETED = "Voltooid"

        # Bestandstypen
        self.ALL_FILES = "Alle bestanden (*)"
        self.GGUF_FILES = "GGUF-bestanden (*.gguf)"
        self.DAT_FILES = "DAT-bestanden (*.dat)"
        self.JSON_FILES = "JSON-bestanden (*.json)"
        self.BIN_FILES = "Binaire bestanden (*.bin)"
        self.LORA_FILES = "LoRA-bestanden (*.bin *.gguf)"
        self.GGUF_AND_BIN_FILES = "GGUF- en binaire bestanden (*.gguf *.bin)"
        self.SHARDED = "geshard"

        # Statusberichten
        self.DOWNLOAD_COMPLETE = "Download voltooid"
        self.CUDA_EXTRACTION_FAILED = "CUDA-extractie mislukt"
        self.PRESET_SAVED = "Voorinstelling opgeslagen"
        self.PRESET_LOADED = "Voorinstelling geladen"
        self.NO_ASSET_SELECTED = "Geen asset geselecteerd"
        self.DOWNLOAD_FAILED = "Download mislukt"
        self.NO_BACKEND_SELECTED = "Geen backend geselecteerd"
        self.NO_MODEL_SELECTED = "Geen model geselecteerd"
        self.NO_SUITABLE_CUDA_BACKENDS = "Geen geschikte CUDA-backends gevonden"
        self.IN_PROGRESS = "Bezig"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "llama.cpp binair bestand gedownload en uitgepakt naar {0}"
        )
        self.CUDA_FILES_EXTRACTED = "CUDA-bestanden uitgepakt naar"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Geen geschikte CUDA-backend gevonden voor extractie"
        )
        self.ERROR_FETCHING_RELEASES = "Fout bij ophalen van releases: {0}"
        self.CONFIRM_DELETION_TITLE = "Verwijdering bevestigen"
        self.LOG_FOR = "Logboek voor {0}"
        self.FAILED_TO_LOAD_PRESET = "Kan voorinstelling niet laden: {0}"
        self.INITIALIZING_AUTOGGUF = "AutoGGUF-applicatie initialiseren"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF-initialisatie voltooid"
        self.REFRESHING_BACKENDS = "Backends vernieuwen"
        self.NO_BACKENDS_AVAILABLE = "Geen backends beschikbaar"
        self.FOUND_VALID_BACKENDS = "{0} geldige backends gevonden"
        self.SAVING_PRESET = "Voorinstelling opslaan"
        self.PRESET_SAVED_TO = "Voorinstelling opgeslagen naar {0}"
        self.LOADING_PRESET = "Voorinstelling laden"
        self.PRESET_LOADED_FROM = "Voorinstelling geladen van {0}"
        self.ADDING_KV_OVERRIDE = "KV-overschrijving toevoegen: {0}"
        self.SAVING_TASK_PRESET = "Taakvoorinstelling opslaan voor {0}"
        self.TASK_PRESET_SAVED = "Taakvoorinstelling opgeslagen"
        self.TASK_PRESET_SAVED_TO = "Taakvoorinstelling opgeslagen naar {0}"
        self.RESTARTING_TASK = "Taak opnieuw starten: {0}"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download voltooid. Uitgepakt naar: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binair bestand gedownload en uitgepakt naar {0}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Geen geschikte CUDA-backend gevonden voor extractie"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binair bestand gedownload en uitgepakt naar {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "llama.cpp releases vernieuwen"
        self.UPDATING_ASSET_LIST = "Assetlijst bijwerken"
        self.UPDATING_CUDA_OPTIONS = "CUDA-opties bijwerken"
        self.STARTING_LLAMACPP_DOWNLOAD = "llama.cpp download starten"
        self.UPDATING_CUDA_BACKENDS = "CUDA-backends bijwerken"
        self.NO_CUDA_BACKEND_SELECTED = "Geen CUDA-backend geselecteerd voor extractie"
        self.EXTRACTING_CUDA_FILES = "CUDA-bestanden uitpakken van {0} naar {1}"
        self.DOWNLOAD_ERROR = "Downloadfout: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Contextmenu van taak weergeven"
        self.SHOWING_PROPERTIES_FOR_TASK = "Eigenschappen weergeven voor taak: {0}"
        self.CANCELLING_TASK = "Taak annuleren: {0}"
        self.CANCELED = "Geannuleerd"
        self.DELETING_TASK = "Taak verwijderen: {0}"
        self.LOADING_MODELS = "Modellen laden"
        self.LOADED_MODELS = "{0} modellen geladen"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Bladeren naar modelmap"
        self.SELECT_MODELS_DIRECTORY = "Selecteer modelmap"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Bladeren naar uitvoermap"
        self.SELECT_OUTPUT_DIRECTORY = "Selecteer uitvoermap"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Bladeren naar logboekmap"
        self.SELECT_LOGS_DIRECTORY = "Selecteer logboekmap"
        self.BROWSING_FOR_IMATRIX_FILE = "Bladeren naar IMatrix-bestand"
        self.SELECT_IMATRIX_FILE = "Selecteer IMatrix-bestand"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU-gebruik: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Kwantiseringsinvoer valideren"
        self.MODELS_PATH_REQUIRED = "Modelpad is vereist"
        self.OUTPUT_PATH_REQUIRED = "Uitvoerpad is vereist"
        self.LOGS_PATH_REQUIRED = "Logboekpad is vereist"
        self.STARTING_MODEL_QUANTIZATION = "Modelkwantisering starten"
        self.INPUT_FILE_NOT_EXIST = "Invoerbestand '{0}' bestaat niet."
        self.QUANTIZING_MODEL_TO = "Kwantiseren van {0} naar {1}"
        self.QUANTIZATION_TASK_STARTED = "Kwantiseringstaak gestart voor {0}"
        self.ERROR_STARTING_QUANTIZATION = "Fout bij starten van kwantisering: {0}"
        self.UPDATING_MODEL_INFO = "Modelinformatie bijwerken: {0}"
        self.TASK_FINISHED = "Taak voltooid: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Taakdetails weergeven voor: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Bladeren naar IMatrix-gegevensbestand"
        self.SELECT_DATA_FILE = "Selecteer gegevensbestand"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Bladeren naar IMatrix-modelbestand"
        self.SELECT_MODEL_FILE = "Selecteer modelbestand"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Bladeren naar IMatrix-uitvoerbestand"
        self.SELECT_OUTPUT_FILE = "Selecteer uitvoerbestand"
        self.STARTING_IMATRIX_GENERATION = "IMatrix-generatie starten"
        self.BACKEND_PATH_NOT_EXIST = "Backendpad bestaat niet: {0}"
        self.GENERATING_IMATRIX = "IMatrix genereren"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Fout bij starten van IMatrix-generatie: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix-generatietaak gestart"
        self.ERROR_MESSAGE = "Fout: {0}"
        self.TASK_ERROR = "Taakfout: {0}"
        self.APPLICATION_CLOSING = "Applicatie wordt afgesloten"
        self.APPLICATION_CLOSED = "Applicatie afgesloten"
        self.SELECT_QUANTIZATION_TYPE = "Selecteer het kwantiseringstype"
        self.ALLOWS_REQUANTIZING = "Maakt het mogelijk om tensoren die al gekwantiseerd zijn opnieuw te kwantiseren"
        self.LEAVE_OUTPUT_WEIGHT = "Laat output.weight niet (opnieuw) gekwantiseerd"
        self.DISABLE_K_QUANT_MIXTURES = "Schakel k-kwantmengsels uit en kwantiseer alle tensoren naar hetzelfde type"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Gebruik gegevens in bestand als belangrijkheidsmatrix voor kwantumoptimalisaties"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Gebruik belangrijkheidsmatrix voor deze tensoren"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Gebruik geen belangrijkheidsmatrix voor deze tensoren"
        )
        self.OUTPUT_TENSOR_TYPE = "Uitvoertensortype:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Gebruik dit type voor de output.weight-tensor"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token Embedding-type:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Gebruik dit type voor de token embeddings-tensor"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Genereert gekwantiseerd model in dezelfde shards als invoer"
        )
        self.OVERRIDE_MODEL_METADATA = "Modelmetadata overschrijven"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Invoergegevensbestand voor IMatrix-generatie"
        )
        self.MODEL_TO_BE_QUANTIZED = "Te kwantiseren model"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Uitvoerpad voor de gegenereerde IMatrix"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Hoe vaak de IMatrix moet worden opgeslagen"
        self.SET_GPU_OFFLOAD_VALUE = "Stel GPU-offloadwaarde in (-ngl)"
        self.STARTING_LORA_CONVERSION = "LoRA-conversie starten"
        self.LORA_INPUT_PATH_REQUIRED = "LoRA-invoerpad is vereist."
        self.LORA_OUTPUT_PATH_REQUIRED = "LoRA-uitvoerpad is vereist."
        self.ERROR_STARTING_LORA_CONVERSION = "Fout bij starten van LoRA-conversie: {}"
        self.LORA_CONVERSION_TASK_STARTED = "LoRA-conversietaak gestart."
        self.BROWSING_FOR_LORA_INPUT_DIRECTORY = "Bladeren naar LoRA-invoermap..."
        self.BROWSING_FOR_LORA_OUTPUT_FILE = "Bladeren naar LoRA-uitvoerbestand..."
        self.CONVERTING_LORA = "LoRA-conversie"
        self.LORA_CONVERSION_FINISHED = "LoRA-conversie voltooid."
        self.LORA_FILE_MOVED = "LoRA-bestand verplaatst van {} naar {}."
        self.LORA_FILE_NOT_FOUND = "LoRA-bestand niet gevonden: {}."
        self.ERROR_MOVING_LORA_FILE = "Fout bij verplaatsen van LoRA-bestand: {}"
        self.MODEL_PATH_REQUIRED = "Modelpad is vereist."
        self.AT_LEAST_ONE_LORA_ADAPTER_REQUIRED = (
            "Minstens één LoRA-adapter is vereist."
        )
        self.INVALID_LORA_SCALE_VALUE = "Ongeldige LoRA-schaalwaarde."
        self.ERROR_STARTING_LORA_EXPORT = "Fout bij starten van LoRA-export: {}"
        self.LORA_EXPORT_TASK_STARTED = "LoRA-exporttaak gestart."
        self.EXPORTING_LORA = "LoRA exporteren..."
        self.BROWSING_FOR_EXPORT_LORA_MODEL_FILE = (
            "Bladeren naar LoRA-modelbestand voor export..."
        )
        self.BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE = (
            "Bladeren naar LoRA-uitvoerbestand voor export..."
        )
        self.ADDING_LORA_ADAPTER = "LoRA-adapter toevoegen..."
        self.DELETING_LORA_ADAPTER = "LoRA-adapter verwijderen..."
        self.SELECT_LORA_ADAPTER_FILE = "Selecteer LoRA-adapterbestand"
        self.STARTING_LORA_EXPORT = "LoRA-export starten..."
        self.SELECT_OUTPUT_TYPE = "Selecteer uitvoertype (GGUF of GGML)"
        self.BASE_MODEL = "Basismodel"
        self.SELECT_BASE_MODEL_FILE = "Selecteer basismodelbestand (GGUF)"
        self.BASE_MODEL_PATH_REQUIRED = "Basismodelpad is vereist voor GGUF-uitvoer."
        self.BROWSING_FOR_BASE_MODEL_FILE = "Bladeren naar basismodelbestand..."
        self.SELECT_BASE_MODEL_FOLDER = "Selecteer basismodelmap (met safetensors)"
        self.BROWSING_FOR_BASE_MODEL_FOLDER = "Bladeren naar basismodelmap..."
        self.LORA_CONVERSION_FROM_TO = "LoRA-conversie van {} naar {}"
        self.GENERATING_IMATRIX_FOR = "IMatrix genereren voor {}"
        self.MODEL_PATH_REQUIRED_FOR_IMATRIX = (
            "Modelpad is vereist voor IMatrix-generatie."
        )
        self.NO_ASSET_SELECTED_FOR_CUDA_CHECK = (
            "Geen asset geselecteerd voor CUDA-controle"
        )
        self.NO_QUANTIZATION_TYPE_SELECTED = "Geen kwantiseringstype geselecteerd. Selecteer ten minste één kwantiseringstype."
        self.STARTING_HF_TO_GGUF_CONVERSION = "HuggingFace naar GGUF-conversie starten"
        self.MODEL_DIRECTORY_REQUIRED = "Modelmap is vereist"
        self.HF_TO_GGUF_CONVERSION_COMMAND = "HF naar GGUF-conversieopdracht: {}"
        self.CONVERTING_TO_GGUF = "Converteren van {} naar GGUF"
        self.ERROR_STARTING_HF_TO_GGUF_CONVERSION = (
            "Fout bij starten van HuggingFace naar GGUF-conversie: {}"
        )
        self.HF_TO_GGUF_CONVERSION_TASK_STARTED = (
            "HuggingFace naar GGUF-conversietaak gestart"
        )


class _Finnish(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (automaattinen GGUF-mallien kvantisoija)"
        self.RAM_USAGE = "RAM-muistin käyttö:"
        self.CPU_USAGE = "CPU:n käyttö:"
        self.BACKEND = "Llama.cpp-taustaosa:"
        self.REFRESH_BACKENDS = "Päivitä taustaosat"
        self.MODELS_PATH = "Mallien polku:"
        self.OUTPUT_PATH = "Tulostepolku:"
        self.LOGS_PATH = "Lokien polku:"
        self.BROWSE = "Selaa"
        self.AVAILABLE_MODELS = "Käytettävissä olevat mallit:"
        self.QUANTIZATION_TYPE = "Kvantisointityyppi:"
        self.ALLOW_REQUANTIZE = "Salli uudelleenkvantisointi"
        self.LEAVE_OUTPUT_TENSOR = "Jätä tulostensori"
        self.PURE = "Puhdas"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Sisällytä painot:"
        self.EXCLUDE_WEIGHTS = "Sulje pois painot:"
        self.USE_OUTPUT_TENSOR_TYPE = "Käytä tulostensorin tyyppiä"
        self.USE_TOKEN_EMBEDDING_TYPE = "Käytä token-upotustyyppiä"
        self.KEEP_SPLIT = "Säilytä jako"
        self.KV_OVERRIDES = "KV-ohitukset:"
        self.ADD_NEW_OVERRIDE = "Lisää uusi ohitus"
        self.QUANTIZE_MODEL = "Kvantisoi malli"
        self.SAVE_PRESET = "Tallenna esiasetus"
        self.LOAD_PRESET = "Lataa esiasetus"
        self.TASKS = "Tehtävät:"
        self.DOWNLOAD_LLAMACPP = "Lataa llama.cpp"
        self.SELECT_RELEASE = "Valitse julkaisu:"
        self.SELECT_ASSET = "Valitse resurssi:"
        self.EXTRACT_CUDA_FILES = "Pura CUDA-tiedostot"
        self.SELECT_CUDA_BACKEND = "Valitse CUDA-taustaosa:"
        self.DOWNLOAD = "Lataa"
        self.IMATRIX_GENERATION = "IMatrix-generointi"
        self.DATA_FILE = "Datatiedosto:"
        self.MODEL = "Malli:"
        self.OUTPUT = "Tuloste:"
        self.OUTPUT_FREQUENCY = "Tulostetaajuus:"
        self.GPU_OFFLOAD = "GPU-kuormansiirto:"
        self.AUTO = "Automaattinen"
        self.GENERATE_IMATRIX = "Generoi IMatrix"
        self.ERROR = "Virhe"
        self.WARNING = "Varoitus"
        self.PROPERTIES = "Ominaisuudet"
        self.CANCEL = "Peruuta"
        self.RESTART = "Käynnistä uudelleen"
        self.DELETE = "Poista"
        self.CONFIRM_DELETION = "Haluatko varmasti poistaa tämän tehtävän?"
        self.TASK_RUNNING_WARNING = (
            "Jotkin tehtävät ovat vielä käynnissä. Haluatko varmasti lopettaa?"
        )
        self.YES = "Kyllä"
        self.NO = "Ei"
        self.DOWNLOAD_COMPLETE = "Lataus valmis"
        self.CUDA_EXTRACTION_FAILED = "CUDA-purku epäonnistui"
        self.PRESET_SAVED = "Esiasetus tallennettu"
        self.PRESET_LOADED = "Esiasetus ladattu"
        self.NO_ASSET_SELECTED = "Ei resurssia valittuna"
        self.DOWNLOAD_FAILED = "Lataus epäonnistui"
        self.NO_BACKEND_SELECTED = "Ei taustaosaa valittuna"
        self.NO_MODEL_SELECTED = "Ei mallia valittuna"
        self.REFRESH_RELEASES = "Päivitä julkaisut"
        self.NO_SUITABLE_CUDA_BACKENDS = "Sopivia CUDA-taustaosoja ei löytynyt"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp-binaaritiedosto ladattu ja purettu kansioon {0}\nCUDA-tiedostot purettu kansioon {1}"
        self.CUDA_FILES_EXTRACTED = "CUDA-tiedostot purettu kansioon"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Sopivaa CUDA-taustaosaa purkua varten ei löytynyt"
        )
        self.ERROR_FETCHING_RELEASES = "Virhe haettaessa julkaisuja: {0}"
        self.CONFIRM_DELETION_TITLE = "Vahvista poisto"
        self.LOG_FOR = "Loki kohteelle {0}"
        self.ALL_FILES = "Kaikki tiedostot (*)"
        self.GGUF_FILES = "GGUF-tiedostot (*.gguf)"
        self.DAT_FILES = "DAT-tiedostot (*.dat)"
        self.JSON_FILES = "JSON-tiedostot (*.json)"
        self.FAILED_LOAD_PRESET = "Esiasetuksen lataus epäonnistui: {0}"
        self.INITIALIZING_AUTOGGUF = "Alustetaan AutoGGUF-sovellusta"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF-alustus valmis"
        self.REFRESHING_BACKENDS = "Päivitetään taustaosoja"
        self.NO_BACKENDS_AVAILABLE = "Ei käytettävissä olevia taustaosoja"
        self.FOUND_VALID_BACKENDS = "Löydettiin {0} kelvollista taustaosaa"
        self.SAVING_PRESET = "Tallennetaan esiasetusta"
        self.PRESET_SAVED_TO = "Esiasetus tallennettu kansioon {0}"
        self.LOADING_PRESET = "Ladataan esiasetusta"
        self.PRESET_LOADED_FROM = "Esiasetus ladattu kansiosta {0}"
        self.ADDING_KV_OVERRIDE = "Lisätään KV-ohitus: {0}"
        self.SAVING_TASK_PRESET = "Tallennetaan tehtäväesiasetusta kohteelle {0}"
        self.TASK_PRESET_SAVED = "Tehtäväesiasetus tallennettu"
        self.TASK_PRESET_SAVED_TO = "Tehtäväesiasetus tallennettu kansioon {0}"
        self.RESTARTING_TASK = "Käynnistetään tehtävä uudelleen: {0}"
        self.IN_PROGRESS = "Käynnissä"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Lataus valmis. Purettu kansioon: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp-binaaritiedosto ladattu ja purettu kansioon {0}\nCUDA-tiedostot purettu kansioon {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Sopivaa CUDA-taustaosaa purkua varten ei löytynyt"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp-binaaritiedosto ladattu ja purettu kansioon {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Päivitetään llama.cpp-julkaisuja"
        self.UPDATING_ASSET_LIST = "Päivitetään resurssilistaa"
        self.UPDATING_CUDA_OPTIONS = "Päivitetään CUDA-asetuksia"
        self.STARTING_LLAMACPP_DOWNLOAD = "Aloitetaan llama.cpp:n lataus"
        self.UPDATING_CUDA_BACKENDS = "Päivitetään CUDA-taustaosoja"
        self.NO_CUDA_BACKEND_SELECTED = "Ei CUDA-taustaosaa valittuna purkua varten"
        self.EXTRACTING_CUDA_FILES = (
            "Puretaan CUDA-tiedostoja kansiosta {0} kansioon {1}"
        )
        self.DOWNLOAD_ERROR = "Latausvirhe: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Näytetään tehtäväkontekstivalikko"
        self.SHOWING_PROPERTIES_FOR_TASK = "Näytetään tehtävän ominaisuudet: {0}"
        self.CANCELLING_TASK = "Peruutetaan tehtävää: {0}"
        self.CANCELED = "Peruutettu"
        self.DELETING_TASK = "Poistetaan tehtävää: {0}"
        self.LOADING_MODELS = "Ladataan malleja"
        self.LOADED_MODELS = "{0} mallia ladattu"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Selaillaan mallikansiota"
        self.SELECT_MODELS_DIRECTORY = "Valitse mallikansio"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Selaillaan tulostekansiota"
        self.SELECT_OUTPUT_DIRECTORY = "Valitse tulostekansio"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Selaillaan lokikansiota"
        self.SELECT_LOGS_DIRECTORY = "Valitse lokikansio"
        self.BROWSING_FOR_IMATRIX_FILE = "Selaillaan IMatrix-tiedostoa"
        self.SELECT_IMATRIX_FILE = "Valitse IMatrix-tiedosto"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} Mt / {2} Mt)"
        self.CPU_USAGE_FORMAT = "CPU:n käyttö: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Vahvistetaan kvantisointisyötteet"
        self.MODELS_PATH_REQUIRED = "Mallien polku on pakollinen"
        self.OUTPUT_PATH_REQUIRED = "Tulostepolku on pakollinen"
        self.LOGS_PATH_REQUIRED = "Lokien polku on pakollinen"
        self.STARTING_MODEL_QUANTIZATION = "Aloitetaan mallin kvantisointi"
        self.INPUT_FILE_NOT_EXIST = "Syötetiedostoa '{0}' ei ole."
        self.QUANTIZING_MODEL_TO = "Kvantisoidaan mallia {0} muotoon {1}"
        self.QUANTIZATION_TASK_STARTED = (
            "Kvantisointitehtävä käynnistetty kohteelle {0}"
        )
        self.ERROR_STARTING_QUANTIZATION = "Virhe kvantisoinnin käynnistyksessä: {0}"
        self.UPDATING_MODEL_INFO = "Päivitetään mallitietoja: {0}"
        self.TASK_FINISHED = "Tehtävä valmis: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Näytetään tehtävän tiedot kohteelle: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Selaillaan IMatrix-datatiedostoa"
        self.SELECT_DATA_FILE = "Valitse datatiedosto"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Selaillaan IMatrix-mallitiedostoa"
        self.SELECT_MODEL_FILE = "Valitse mallitiedosto"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Selaillaan IMatrix-tulostetiedostoa"
        self.SELECT_OUTPUT_FILE = "Valitse tulostetiedosto"
        self.STARTING_IMATRIX_GENERATION = "Aloitetaan IMatrix-generointi"
        self.BACKEND_PATH_NOT_EXIST = "Taustaosan polkua ei ole: {0}"
        self.GENERATING_IMATRIX = "Generoidaan IMatrixia"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Virhe IMatrix-generoinnin käynnistyksessä: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix-generointi käynnistetty"
        self.ERROR_MESSAGE = "Virhe: {0}"
        self.TASK_ERROR = "Tehtävävirhe: {0}"
        self.APPLICATION_CLOSING = "Sovellus suljetaan"
        self.APPLICATION_CLOSED = "Sovellus suljettu"
        self.SELECT_QUANTIZATION_TYPE = "Valitse kvantisointityyppi"
        self.ALLOWS_REQUANTIZING = (
            "Sallii jo kvantisoitujen tensoreiden uudelleenkvantisoinnin"
        )
        self.LEAVE_OUTPUT_WEIGHT = (
            "Jättää output.weight-tensorin (uudelleen)kvantisoimatta"
        )
        self.DISABLE_K_QUANT_MIXTURES = "Poista käytöstä k-kvanttisekoitukset ja kvantisoi kaikki tensorit samaan tyyppiin"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "Käytä tiedoston tietoja kvantisoinnin optimoinnin tärkeysmatriisina"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Käytä tärkeysmatriisia näille tensoreille"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Älä käytä tärkeysmatriisia näille tensoreille"
        )
        self.OUTPUT_TENSOR_TYPE = "Tulostensorin tyyppi:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Käytä tätä tyyppiä output.weight-tensorille"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token-upotustyyppi:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Käytä tätä tyyppiä token-upotustensorille"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Generoi kvantisoidun mallin samoihin osiin kuin syöte"
        )
        self.OVERRIDE_MODEL_METADATA = "Ohita mallitiedot"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix-generoinnin syötedatatiedosto"
        self.MODEL_TO_BE_QUANTIZED = "Kvantisoitava malli"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Generoidun IMatrixin tulostepolku"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Kuinka usein IMatrix tallennetaan"
        self.SET_GPU_OFFLOAD_VALUE = "Aseta GPU-kuormansiirron arvo (-ngl)"
        self.COMPLETED = "Valmis"
        self.REFRESH_MODELS = "Päivitä mallit"


class _Bengali(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (স্বয়ংক্রিয় GGUF মডেল কোয়ান্টাইজার)"
        self.RAM_USAGE = "RAM ব্যবহার:"
        self.CPU_USAGE = "CPU ব্যবহার:"
        self.BACKEND = "Llama.cpp ব্যাকএন্ড:"
        self.REFRESH_BACKENDS = "ব্যাকএন্ড রিফ্রেশ করুন"
        self.MODELS_PATH = "মডেল পাথ:"
        self.OUTPUT_PATH = "আউটপুট পাথ:"
        self.LOGS_PATH = "লগ পাথ:"
        self.BROWSE = "ব্রাউজ করুন"
        self.AVAILABLE_MODELS = "উপলব্ধ মডেল:"
        self.QUANTIZATION_TYPE = "কোয়ান্টাইজেশন ধরণ:"
        self.ALLOW_REQUANTIZE = "পুনরায় কোয়ান্টাইজ করার অনুমতি দিন"
        self.LEAVE_OUTPUT_TENSOR = "আউটপুট টেন্সর রেখে দিন"
        self.PURE = "বিশুদ্ধ"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "ওজন অন্তর্ভুক্ত করুন:"
        self.EXCLUDE_WEIGHTS = "ওজন বাদ দিন:"
        self.USE_OUTPUT_TENSOR_TYPE = "আউটপুট টেন্সর ধরণ ব্যবহার করুন"
        self.USE_TOKEN_EMBEDDING_TYPE = "টোকেন এম্বেডিং ধরণ ব্যবহার করুন"
        self.KEEP_SPLIT = "বিভাজন রাখুন"
        self.KV_OVERRIDES = "KV ওভাররাইড:"
        self.ADD_NEW_OVERRIDE = "নতুন ওভাররাইড যুক্ত করুন"
        self.QUANTIZE_MODEL = "মডেল কোয়ান্টাইজ করুন"
        self.SAVE_PRESET = "প্রিসেট সংরক্ষণ করুন"
        self.LOAD_PRESET = "প্রিসেট লোড করুন"
        self.TASKS = "কার্য:"
        self.DOWNLOAD_LLAMACPP = "llama.cpp ডাউনলোড করুন"
        self.SELECT_RELEASE = "রিলিজ নির্বাচন করুন:"
        self.SELECT_ASSET = "অ্যাসেট নির্বাচন করুন:"
        self.EXTRACT_CUDA_FILES = "CUDA ফাইলগুলি বের করুন"
        self.SELECT_CUDA_BACKEND = "CUDA ব্যাকএন্ড নির্বাচন করুন:"
        self.DOWNLOAD = "ডাউনলোড করুন"
        self.IMATRIX_GENERATION = "IMatrix জেনারেশন"
        self.DATA_FILE = "ডেটা ফাইল:"
        self.MODEL = "মডেল:"
        self.OUTPUT = "আউটপুট:"
        self.OUTPUT_FREQUENCY = "আউটপুট ফ্রিকোয়েন্সি:"
        self.GPU_OFFLOAD = "GPU অফলোড:"
        self.AUTO = "স্বয়ংক্রিয়"
        self.GENERATE_IMATRIX = "IMatrix তৈরি করুন"
        self.ERROR = "ত্রুটি"
        self.WARNING = "সতর্কীকরণ"
        self.PROPERTIES = "বৈশিষ্ট্য"
        self.CANCEL = "বাতিল করুন"
        self.RESTART = "পুনরায় আরম্ভ করুন"
        self.DELETE = "মুছে ফেলুন"
        self.CONFIRM_DELETION = "আপনি কি নিশ্চিত যে আপনি এই কাজটি মুছে ফেলতে চান?"
        self.TASK_RUNNING_WARNING = (
            "কিছু কাজ এখনও চলছে। আপনি কি নিশ্চিত যে আপনি প্রস্থান করতে চান?"
        )
        self.YES = "হ্যাঁ"
        self.NO = "না"
        self.DOWNLOAD_COMPLETE = "ডাউনলোড সম্পন্ন"
        self.CUDA_EXTRACTION_FAILED = "CUDA এক্সট্র্যাকশন ব্যর্থ"
        self.PRESET_SAVED = "প্রিসেট সংরক্ষিত"
        self.PRESET_LOADED = "প্রিসেট লোড করা হয়েছে"
        self.NO_ASSET_SELECTED = "কোন অ্যাসেট নির্বাচন করা হয়নি"
        self.DOWNLOAD_FAILED = "ডাউনলোড ব্যর্থ"
        self.NO_BACKEND_SELECTED = "কোন ব্যাকএন্ড নির্বাচন করা হয়নি"
        self.NO_MODEL_SELECTED = "কোন মডেল নির্বাচন করা হয়নি"
        self.REFRESH_RELEASES = "রিলিজগুলি রিফ্রেশ করুন"
        self.NO_SUITABLE_CUDA_BACKENDS = "কোন উপযুক্ত CUDA ব্যাকএন্ড পাওয়া যায়নি"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp বাইনারি ফাইল ডাউনলোড এবং {0} এ বের করা হয়েছে\nCUDA ফাইলগুলি {1} এ বের করা হয়েছে"
        self.CUDA_FILES_EXTRACTED = "CUDA ফাইলগুলি তে বের করা হয়েছে"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "এক্সট্র্যাকশনের জন্য কোন উপযুক্ত CUDA ব্যাকএন্ড পাওয়া যায়নি"
        )
        self.ERROR_FETCHING_RELEASES = "রিলিজগুলি আনতে ত্রুটি: {0}"
        self.CONFIRM_DELETION_TITLE = "মুছে ফেলা নিশ্চিত করুন"
        self.LOG_FOR = "{0} এর জন্য লগ"
        self.ALL_FILES = "সমস্ত ফাইল (*)"
        self.GGUF_FILES = "GGUF ফাইল (*.gguf)"
        self.DAT_FILES = "DAT ফাইল (*.dat)"
        self.JSON_FILES = "JSON ফাইল (*.json)"
        self.FAILED_LOAD_PRESET = "প্রিসেট লোড করতে ব্যর্থ: {0}"
        self.INITIALIZING_AUTOGGUF = "AutoGGUF অ্যাপ্লিকেশন শুরু হচ্ছে"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF ইনিশিয়ালাইজেশন সম্পন্ন"
        self.REFRESHING_BACKENDS = "ব্যাকএন্ডগুলি রিফ্রেশ করা হচ্ছে"
        self.NO_BACKENDS_AVAILABLE = "কোন ব্যাকএন্ড উপলব্ধ নেই"
        self.FOUND_VALID_BACKENDS = "{0} টি বৈধ ব্যাকএন্ড পাওয়া গেছে"
        self.SAVING_PRESET = "প্রিসেট সংরক্ষণ করা হচ্ছে"
        self.PRESET_SAVED_TO = "{0} এ প্রিসেট সংরক্ষিত"
        self.LOADING_PRESET = "প্রিসেট লোড করা হচ্ছে"
        self.PRESET_LOADED_FROM = "{0} থেকে প্রিসেট লোড করা হয়েছে"
        self.ADDING_KV_OVERRIDE = "KV ওভাররাইড যুক্ত করা হচ্ছে: {0}"
        self.SAVING_TASK_PRESET = "{0} এর জন্য টাস্ক প্রিসেট সংরক্ষণ করা হচ্ছে"
        self.TASK_PRESET_SAVED = "টাস্ক প্রিসেট সংরক্ষিত"
        self.TASK_PRESET_SAVED_TO = "{0} এ টাস্ক প্রিসেট সংরক্ষিত"
        self.RESTARTING_TASK = "টাস্ক পুনরায় শুরু করা হচ্ছে: {0}"
        self.IN_PROGRESS = "চলছে"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "ডাউনলোড সম্পন্ন। বের করা হয়েছে: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp বাইনারি ফাইল ডাউনলোড এবং {0} এ বের করা হয়েছে\nCUDA ফাইলগুলি {1} এ বের করা হয়েছে"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "এক্সট্র্যাকশনের জন্য কোন উপযুক্ত CUDA ব্যাকএন্ড পাওয়া যায়নি"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp বাইনারি ফাইল ডাউনলোড এবং {0} এ বের করা হয়েছে"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "llama.cpp রিলিজগুলি রিফ্রেশ করা হচ্ছে"
        self.UPDATING_ASSET_LIST = "অ্যাসেট তালিকা আপডেট করা হচ্ছে"
        self.UPDATING_CUDA_OPTIONS = "CUDA অপশনগুলি আপডেট করা হচ্ছে"
        self.STARTING_LLAMACPP_DOWNLOAD = "llama.cpp ডাউনলোড শুরু করা হচ্ছে"
        self.UPDATING_CUDA_BACKENDS = "CUDA ব্যাকএন্ডগুলি আপডেট করা হচ্ছে"
        self.NO_CUDA_BACKEND_SELECTED = (
            "এক্সট্র্যাকশনের জন্য কোন CUDA ব্যাকএন্ড নির্বাচন করা হয়নি"
        )
        self.EXTRACTING_CUDA_FILES = "{0} থেকে {1} এ CUDA ফাইলগুলি বের করা হচ্ছে"
        self.DOWNLOAD_ERROR = "ডাউনলোড ত্রুটি: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "টাস্ক কনটেক্সট মেনু দেখানো হচ্ছে"
        self.SHOWING_PROPERTIES_FOR_TASK = "টাস্কের জন্য বৈশিষ্ট্য দেখানো হচ্ছে: {0}"
        self.CANCELLING_TASK = "টাস্ক বাতিল করা হচ্ছে: {0}"
        self.CANCELED = "বাতিল করা হয়েছে"
        self.DELETING_TASK = "টাস্ক মুছে ফেলা হচ্ছে: {0}"
        self.LOADING_MODELS = "মডেলগুলি লোড করা হচ্ছে"
        self.LOADED_MODELS = "{0} টি মডেল লোড করা হয়েছে"
        self.BROWSING_FOR_MODELS_DIRECTORY = "মডেল ডিরেক্টরি ব্রাউজ করা হচ্ছে"
        self.SELECT_MODELS_DIRECTORY = "মডেল ডিরেক্টরি নির্বাচন করুন"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "আউটপুট ডিরেক্টরি ব্রাউজ করা হচ্ছে"
        self.SELECT_OUTPUT_DIRECTORY = "আউটপুট ডিরেক্টরি নির্বাচন করুন"
        self.BROWSING_FOR_LOGS_DIRECTORY = "লগ ডিরেক্টরি ব্রাউজ করা হচ্ছে"
        self.SELECT_LOGS_DIRECTORY = "লগ ডিরেক্টরি নির্বাচন করুন"
        self.BROWSING_FOR_IMATRIX_FILE = "IMatrix ফাইল ব্রাউজ করা হচ্ছে"
        self.SELECT_IMATRIX_FILE = "IMatrix ফাইল নির্বাচন করুন"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU ব্যবহার: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "কোয়ান্টাইজেশন ইনপুট যাচাই করা হচ্ছে"
        self.MODELS_PATH_REQUIRED = "মডেল পাথ প্রয়োজন"
        self.OUTPUT_PATH_REQUIRED = "আউটপুট পাথ প্রয়োজন"
        self.LOGS_PATH_REQUIRED = "লগ পাথ প্রয়োজন"
        self.STARTING_MODEL_QUANTIZATION = "মডেল কোয়ান্টাইজেশন শুরু হচ্ছে"
        self.INPUT_FILE_NOT_EXIST = "ইনপুট ফাইল '{0}' বিদ্যমান নেই।"
        self.QUANTIZING_MODEL_TO = "{0} কে {1} এ কোয়ান্টাইজ করা হচ্ছে"
        self.QUANTIZATION_TASK_STARTED = "{0} এর জন্য কোয়ান্টাইজেশন টাস্ক শুরু হয়েছে"
        self.ERROR_STARTING_QUANTIZATION = "কোয়ান্টাইজেশন শুরু করতে ত্রুটি: {0}"
        self.UPDATING_MODEL_INFO = "মডেল তথ্য আপডেট করা হচ্ছে: {0}"
        self.TASK_FINISHED = "টাস্ক সম্পন্ন: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "এর জন্য টাস্কের বিবরণ দেখানো হচ্ছে: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "IMatrix ডেটা ফাইল ব্রাউজ করা হচ্ছে"
        self.SELECT_DATA_FILE = "ডেটা ফাইল নির্বাচন করুন"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "IMatrix মডেল ফাইল ব্রাউজ করা হচ্ছে"
        self.SELECT_MODEL_FILE = "মডেল ফাইল নির্বাচন করুন"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "IMatrix আউটপুট ফাইল ব্রাউজ করা হচ্ছে"
        self.SELECT_OUTPUT_FILE = "আউটপুট ফাইল নির্বাচন করুন"
        self.STARTING_IMATRIX_GENERATION = "IMatrix জেনারেশন শুরু হচ্ছে"
        self.BACKEND_PATH_NOT_EXIST = "ব্যাকএন্ড পাথ বিদ্যমান নেই: {0}"
        self.GENERATING_IMATRIX = "IMatrix তৈরি করা হচ্ছে"
        self.ERROR_STARTING_IMATRIX_GENERATION = "IMatrix জেনারেশন শুরু করতে ত্রুটি: {0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix জেনারেশন টাস্ক শুরু হয়েছে"
        self.ERROR_MESSAGE = "ত্রুটি: {0}"
        self.TASK_ERROR = "টাস্ক ত্রুটি: {0}"
        self.APPLICATION_CLOSING = "অ্যাপ্লিকেশন বন্ধ করা হচ্ছে"
        self.APPLICATION_CLOSED = "অ্যাপ্লিকেশন বন্ধ"
        self.SELECT_QUANTIZATION_TYPE = "কোয়ান্টাইজেশন ধরণ নির্বাচন করুন"
        self.ALLOWS_REQUANTIZING = "যে টেন্সরগুলি ইতিমধ্যে কোয়ান্টাইজ করা হয়েছে তাদের পুনরায় কোয়ান্টাইজ করার অনুমতি দেয়"
        self.LEAVE_OUTPUT_WEIGHT = "output.weight কে (পুনরায়) কোয়ান্টাইজ না করে রেখে দেবে"
        self.DISABLE_K_QUANT_MIXTURES = (
            "k-কোয়ান্ট মিশ্রণগুলি অক্ষম করুন এবং সমস্ত টেন্সরকে একই ধরণের কোয়ান্টাইজ করুন"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "কোয়ান্ট অপ্টিমাইজেশনের জন্য ফাইলের ডেটা গুরুত্বপূর্ণ ম্যাট্রিক্স হিসাবে ব্যবহার করুন"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "এই টেন্সরগুলির জন্য গুরুত্বপূর্ণ ম্যাট্রিক্স ব্যবহার করুন"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "এই টেন্সরগুলির জন্য গুরুত্বপূর্ণ ম্যাট্রিক্স ব্যবহার করবেন না"
        )
        self.OUTPUT_TENSOR_TYPE = "আউটপুট টেন্সর ধরণ:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "output.weight টেন্সরের জন্য এই ধরণটি ব্যবহার করুন"
        )
        self.TOKEN_EMBEDDING_TYPE = "টোকেন এম্বেডিং ধরণ:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "টোকেন এম্বেডিং টেন্সরের জন্য এই ধরণটি ব্যবহার করুন"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "ইনপুটের মতো একই শার্ডে কোয়ান্টাইজ করা মডেল তৈরি করবে"
        )
        self.OVERRIDE_MODEL_METADATA = "মডেল মেটাডেটা ওভাররাইড করুন"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix জেনারেশনের জন্য ইনপুট ডেটা ফাইল"
        self.MODEL_TO_BE_QUANTIZED = "কোয়ান্টাইজ করার জন্য মডেল"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "তৈরি করা IMatrix এর জন্য আউটপুট পাথ"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "IMatrix কতবার সংরক্ষণ করবেন"
        self.SET_GPU_OFFLOAD_VALUE = "GPU অফলোড মান সেট করুন (-ngl)"
        self.COMPLETED = "সম্পন্ন"
        self.REFRESH_MODELS = "মডেল রিফ্রেশ করুন"


class _Polish(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Automatyczny kwantyzator modeli GGUF)"
        self.RAM_USAGE = "Użycie pamięci RAM:"
        self.CPU_USAGE = "Użycie procesora:"
        self.BACKEND = "Backend Llama.cpp:"
        self.REFRESH_BACKENDS = "Odśwież backendy"
        self.MODELS_PATH = "Ścieżka modeli:"
        self.OUTPUT_PATH = "Ścieżka wyjściowa:"
        self.LOGS_PATH = "Ścieżka logów:"
        self.BROWSE = "Przeglądaj"
        self.AVAILABLE_MODELS = "Dostępne modele:"
        self.QUANTIZATION_TYPE = "Typ kwantyzacji:"
        self.ALLOW_REQUANTIZE = "Zezwól na ponowną kwantyzację"
        self.LEAVE_OUTPUT_TENSOR = "Pozostaw tensor wyjściowy"
        self.PURE = "Czysty"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Uwzględnij wagi:"
        self.EXCLUDE_WEIGHTS = "Wyklucz wagi:"
        self.USE_OUTPUT_TENSOR_TYPE = "Użyj typu tensora wyjściowego"
        self.USE_TOKEN_EMBEDDING_TYPE = "Użyj typu osadzania tokenów"
        self.KEEP_SPLIT = "Zachowaj podział"
        self.KV_OVERRIDES = "Nadpisania KV:"
        self.ADD_NEW_OVERRIDE = "Dodaj nowe nadpisanie"
        self.QUANTIZE_MODEL = "Kwantyzuj model"
        self.SAVE_PRESET = "Zapisz ustawienia predefiniowane"
        self.LOAD_PRESET = "Wczytaj ustawienia predefiniowane"
        self.TASKS = "Zadania:"
        self.DOWNLOAD_LLAMACPP = "Pobierz llama.cpp"
        self.SELECT_RELEASE = "Wybierz wersję:"
        self.SELECT_ASSET = "Wybierz zasób:"
        self.EXTRACT_CUDA_FILES = "Wyodrębnij pliki CUDA"
        self.SELECT_CUDA_BACKEND = "Wybierz backend CUDA:"
        self.DOWNLOAD = "Pobierz"
        self.IMATRIX_GENERATION = "Generowanie IMatrix"
        self.DATA_FILE = "Plik danych:"
        self.MODEL = "Model:"
        self.OUTPUT = "Wyjście:"
        self.OUTPUT_FREQUENCY = "Częstotliwość wyjścia:"
        self.GPU_OFFLOAD = "Odciążenie GPU:"
        self.AUTO = "Automatyczny"
        self.GENERATE_IMATRIX = "Generuj IMatrix"
        self.ERROR = "Błąd"
        self.WARNING = "Ostrzeżenie"
        self.PROPERTIES = "Właściwości"
        self.CANCEL = "Anuluj"
        self.RESTART = "Uruchom ponownie"
        self.DELETE = "Usuń"
        self.CONFIRM_DELETION = "Czy na pewno chcesz usunąć to zadanie?"
        self.TASK_RUNNING_WARNING = (
            "Niektóre zadania są nadal uruchomione. Czy na pewno chcesz wyjść?"
        )
        self.YES = "Tak"
        self.NO = "Nie"
        self.DOWNLOAD_COMPLETE = "Pobieranie zakończone"
        self.CUDA_EXTRACTION_FAILED = "Wyodrębnianie CUDA nie powiodło się"
        self.PRESET_SAVED = "Ustawienia predefiniowane zapisane"
        self.PRESET_LOADED = "Ustawienia predefiniowane wczytane"
        self.NO_ASSET_SELECTED = "Nie wybrano zasobu"
        self.DOWNLOAD_FAILED = "Pobieranie nie powiodło się"
        self.NO_BACKEND_SELECTED = "Nie wybrano backendu"
        self.NO_MODEL_SELECTED = "Nie wybrano modelu"
        self.REFRESH_RELEASES = "Odśwież wersje"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nie znaleziono odpowiednich backendów CUDA"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Plik binarny llama.cpp został pobrany i wyodrębniony do {0}\nPliki CUDA wyodrębnione do {1}"
        self.CUDA_FILES_EXTRACTED = "Pliki CUDA wyodrębnione do"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nie znaleziono odpowiedniego backendu CUDA do wyodrębnienia"
        )
        self.ERROR_FETCHING_RELEASES = "Błąd podczas pobierania wersji: {0}"
        self.CONFIRM_DELETION_TITLE = "Potwierdź usunięcie"
        self.LOG_FOR = "Dziennik dla {0}"
        self.ALL_FILES = "Wszystkie pliki (*)"
        self.GGUF_FILES = "Pliki GGUF (*.gguf)"
        self.DAT_FILES = "Pliki DAT (*.dat)"
        self.JSON_FILES = "Pliki JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Nie udało się wczytać ustawień predefiniowanych: {0}"
        self.INITIALIZING_AUTOGGUF = "Inicjalizacja aplikacji AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Inicjalizacja AutoGGUF zakończona"
        self.REFRESHING_BACKENDS = "Odświeżanie backendów"
        self.NO_BACKENDS_AVAILABLE = "Brak dostępnych backendów"
        self.FOUND_VALID_BACKENDS = "Znaleziono {0} prawidłowych backendów"
        self.SAVING_PRESET = "Zapisywanie ustawień predefiniowanych"
        self.PRESET_SAVED_TO = "Ustawienia predefiniowane zapisane do {0}"
        self.LOADING_PRESET = "Wczytywanie ustawień predefiniowanych"
        self.PRESET_LOADED_FROM = "Ustawienia predefiniowane wczytane z {0}"
        self.ADDING_KV_OVERRIDE = "Dodawanie nadpisania KV: {0}"
        self.SAVING_TASK_PRESET = (
            "Zapisywanie ustawień predefiniowanych zadania dla {0}"
        )
        self.TASK_PRESET_SAVED = "Ustawienia predefiniowane zadania zapisane"
        self.TASK_PRESET_SAVED_TO = "Ustawienia predefiniowane zadania zapisane do {0}"
        self.RESTARTING_TASK = "Ponowne uruchamianie zadania: {0}"
        self.IN_PROGRESS = "W trakcie"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "Pobieranie zakończone. Wyodrębniono do: {0}"
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Plik binarny llama.cpp został pobrany i wyodrębniony do {0}\nPliki CUDA wyodrębnione do {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nie znaleziono odpowiedniego backendu CUDA do wyodrębnienia"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Plik binarny llama.cpp został pobrany i wyodrębniony do {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Odświeżanie wersji llama.cpp"
        self.UPDATING_ASSET_LIST = "Aktualizacja listy zasobów"
        self.UPDATING_CUDA_OPTIONS = "Aktualizacja opcji CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Rozpoczynanie pobierania llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Aktualizacja backendów CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Nie wybrano backendu CUDA do wyodrębnienia"
        self.EXTRACTING_CUDA_FILES = "Wyodrębnianie plików CUDA z {0} do {1}"
        self.DOWNLOAD_ERROR = "Błąd pobierania: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Wyświetlanie menu kontekstowego zadania"
        self.SHOWING_PROPERTIES_FOR_TASK = "Wyświetlanie właściwości zadania: {0}"
        self.CANCELLING_TASK = "Anulowanie zadania: {0}"
        self.CANCELED = "Anulowano"
        self.DELETING_TASK = "Usuwanie zadania: {0}"
        self.LOADING_MODELS = "Ładowanie modeli"
        self.LOADED_MODELS = "Załadowano {0} modeli"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Przeglądanie katalogu modeli"
        self.SELECT_MODELS_DIRECTORY = "Wybierz katalog modeli"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Przeglądanie katalogu wyjściowego"
        self.SELECT_OUTPUT_DIRECTORY = "Wybierz katalog wyjściowy"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Przeglądanie katalogu logów"
        self.SELECT_LOGS_DIRECTORY = "Wybierz katalog logów"
        self.BROWSING_FOR_IMATRIX_FILE = "Przeglądanie pliku IMatrix"
        self.SELECT_IMATRIX_FILE = "Wybierz plik IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Użycie procesora: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Walidacja danych wejściowych kwantyzacji"
        self.MODELS_PATH_REQUIRED = "Ścieżka modeli jest wymagana"
        self.OUTPUT_PATH_REQUIRED = "Ścieżka wyjściowa jest wymagana"
        self.LOGS_PATH_REQUIRED = "Ścieżka logów jest wymagana"
        self.STARTING_MODEL_QUANTIZATION = "Rozpoczynanie kwantyzacji modelu"
        self.INPUT_FILE_NOT_EXIST = "Plik wejściowy '{0}' nie istnieje."
        self.QUANTIZING_MODEL_TO = "Kwantyzacja {0} do {1}"
        self.QUANTIZATION_TASK_STARTED = "Zadanie kwantyzacji uruchomione dla {0}"
        self.ERROR_STARTING_QUANTIZATION = "Błąd podczas uruchamiania kwantyzacji: {0}"
        self.UPDATING_MODEL_INFO = "Aktualizacja informacji o modelu: {0}"
        self.TASK_FINISHED = "Zadanie zakończone: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Wyświetlanie szczegółów zadania dla: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Przeglądanie pliku danych IMatrix"
        self.SELECT_DATA_FILE = "Wybierz plik danych"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Przeglądanie pliku modelu IMatrix"
        self.SELECT_MODEL_FILE = "Wybierz plik modelu"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Przeglądanie pliku wyjściowego IMatrix"
        self.SELECT_OUTPUT_FILE = "Wybierz plik wyjściowy"
        self.STARTING_IMATRIX_GENERATION = "Rozpoczynanie generowania IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Ścieżka backendu nie istnieje: {0}"
        self.GENERATING_IMATRIX = "Generowanie IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Błąd podczas uruchamiania generowania IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Zadanie generowania IMatrix uruchomione"
        self.ERROR_MESSAGE = "Błąd: {0}"
        self.TASK_ERROR = "Błąd zadania: {0}"
        self.APPLICATION_CLOSING = "Zamykanie aplikacji"
        self.APPLICATION_CLOSED = "Aplikacja zamknięta"
        self.SELECT_QUANTIZATION_TYPE = "Wybierz typ kwantyzacji"
        self.ALLOWS_REQUANTIZING = (
            "Pozwala na ponowną kwantyzację tensorów, które zostały już skwantyzowane"
        )
        self.LEAVE_OUTPUT_WEIGHT = (
            "Pozostawi output.weight nieskwantyzowany (lub nieskwantyzowany ponownie)"
        )
        self.DISABLE_K_QUANT_MIXTURES = (
            "Wyłącz mieszanki k-kwant i kwantyzuj wszystkie tensory do tego samego typu"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "Użyj danych w pliku jako macierzy ważności dla optymalizacji kwantyzacji"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Użyj macierzy ważności dla tych tensorów"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Nie używaj macierzy ważności dla tych tensorów"
        )
        self.OUTPUT_TENSOR_TYPE = "Typ tensora wyjściowego:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Użyj tego typu dla tensora output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Typ osadzania tokenów:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Użyj tego typu dla tensora osadzania tokenów"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Wygeneruje skwantyzowany model w tych samych fragmentach co dane wejściowe"
        )
        self.OVERRIDE_MODEL_METADATA = "Zastąp metadane modelu"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Plik danych wejściowych do generowania IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Model do kwantyzacji"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Ścieżka wyjściowa dla wygenerowanego IMatrix"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Jak często zapisywać IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Ustaw wartość odciążenia GPU (-ngl)"
        self.COMPLETED = "Ukończono"
        self.REFRESH_MODELS = "Obnovit modely"


class _Romanian(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Cuantizator automat de modele GGUF)"
        self.RAM_USAGE = "Utilizare RAM:"
        self.CPU_USAGE = "Utilizare CPU:"
        self.BACKEND = "Backend Llama.cpp:"
        self.REFRESH_BACKENDS = "Reîmprospătați backends"
        self.MODELS_PATH = "Cale modele:"
        self.OUTPUT_PATH = "Cale ieșire:"
        self.LOGS_PATH = "Cale jurnale:"
        self.BROWSE = "Răsfoiți"
        self.AVAILABLE_MODELS = "Modele disponibile:"
        self.QUANTIZATION_TYPE = "Tipul de cuantizare:"
        self.ALLOW_REQUANTIZE = "Permiteți recuantizarea"
        self.LEAVE_OUTPUT_TENSOR = "Lăsați tensorul de ieșire"
        self.PURE = "Pur"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Includeți ponderile:"
        self.EXCLUDE_WEIGHTS = "Excludeți ponderile:"
        self.USE_OUTPUT_TENSOR_TYPE = "Utilizați tipul tensorului de ieșire"
        self.USE_TOKEN_EMBEDDING_TYPE = "Utilizați tipul de încorporare a tokenului"
        self.KEEP_SPLIT = "Păstrați divizarea"
        self.KV_OVERRIDES = "Suprascrieri KV:"
        self.ADD_NEW_OVERRIDE = "Adăugați o nouă suprascriere"
        self.QUANTIZE_MODEL = "Cuantizați modelul"
        self.SAVE_PRESET = "Salvați presetarea"
        self.LOAD_PRESET = "Încărcați presetarea"
        self.TASKS = "Sarcini:"
        self.DOWNLOAD_LLAMACPP = "Descărcați llama.cpp"
        self.SELECT_RELEASE = "Selectați versiunea:"
        self.SELECT_ASSET = "Selectați activul:"
        self.EXTRACT_CUDA_FILES = "Extrageți fișierele CUDA"
        self.SELECT_CUDA_BACKEND = "Selectați backend CUDA:"
        self.DOWNLOAD = "Descărcați"
        self.IMATRIX_GENERATION = "Generare IMatrix"
        self.DATA_FILE = "Fișier de date:"
        self.MODEL = "Model:"
        self.OUTPUT = "Ieșire:"
        self.OUTPUT_FREQUENCY = "Frecvența ieșirii:"
        self.GPU_OFFLOAD = "Descărcare GPU:"
        self.AUTO = "Automat"
        self.GENERATE_IMATRIX = "Generați IMatrix"
        self.ERROR = "Eroare"
        self.WARNING = "Avertisment"
        self.PROPERTIES = "Proprietăți"
        self.CANCEL = "Anulați"
        self.RESTART = "Reporniți"
        self.DELETE = "Ștergeți"
        self.CONFIRM_DELETION = "Sunteți sigur că doriți să ștergeți această sarcină?"
        self.TASK_RUNNING_WARNING = "Unele sarcini sunt încă în curs de execuție. Sunteți sigur că doriți să ieșiți?"
        self.YES = "Da"
        self.NO = "Nu"
        self.DOWNLOAD_COMPLETE = "Descărcare finalizată"
        self.CUDA_EXTRACTION_FAILED = "Extragerea CUDA a eșuat"
        self.PRESET_SAVED = "Presetare salvată"
        self.PRESET_LOADED = "Presetare încărcată"
        self.NO_ASSET_SELECTED = "Niciun activ selectat"
        self.DOWNLOAD_FAILED = "Descărcarea a eșuat"
        self.NO_BACKEND_SELECTED = "Niciun backend selectat"
        self.NO_MODEL_SELECTED = "Niciun model selectat"
        self.REFRESH_RELEASES = "Reîmprospătați versiunile"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nu s-au găsit backends CUDA potrivite"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Fișierul binar llama.cpp a fost descărcat și extras în {0}\nFișierele CUDA au fost extrase în {1}"
        self.CUDA_FILES_EXTRACTED = "Fișierele CUDA au fost extrase în"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nu s-a găsit un backend CUDA potrivit pentru extragere"
        )
        self.ERROR_FETCHING_RELEASES = "Eroare la preluarea versiunilor: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirmați ștergerea"
        self.LOG_FOR = "Jurnal pentru {0}"
        self.ALL_FILES = "Toate fișierele (*)"
        self.GGUF_FILES = "Fișiere GGUF (*.gguf)"
        self.DAT_FILES = "Fișiere DAT (*.dat)"
        self.JSON_FILES = "Fișiere JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Nu s-a putut încărca presetarea: {0}"
        self.INITIALIZING_AUTOGGUF = "Inițializarea aplicației AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Inițializarea AutoGGUF finalizată"
        self.REFRESHING_BACKENDS = "Reîmprospătarea backends"
        self.NO_BACKENDS_AVAILABLE = "Nu există backends disponibile"
        self.FOUND_VALID_BACKENDS = "S-au găsit {0} backends valide"
        self.SAVING_PRESET = "Salvarea presetării"
        self.PRESET_SAVED_TO = "Presetare salvată în {0}"
        self.LOADING_PRESET = "Încărcarea presetării"
        self.PRESET_LOADED_FROM = "Presetare încărcată din {0}"
        self.ADDING_KV_OVERRIDE = "Adăugarea suprascrierii KV: {0}"
        self.SAVING_TASK_PRESET = "Salvarea presetării sarcinii pentru {0}"
        self.TASK_PRESET_SAVED = "Presetare sarcină salvată"
        self.TASK_PRESET_SAVED_TO = "Presetare sarcină salvată în {0}"
        self.RESTARTING_TASK = "Repornirea sarcinii: {0}"
        self.IN_PROGRESS = "În curs"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Descărcare finalizată. Extras în: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Fișierul binar llama.cpp a fost descărcat și extras în {0}\nFișierele CUDA au fost extrase în {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nu s-a găsit un backend CUDA potrivit pentru extragere"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Fișierul binar llama.cpp a fost descărcat și extras în {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Reîmprospătarea versiunilor llama.cpp"
        self.UPDATING_ASSET_LIST = "Actualizarea listei de active"
        self.UPDATING_CUDA_OPTIONS = "Actualizarea opțiunilor CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Începerea descărcării llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Actualizarea backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Niciun backend CUDA selectat pentru extragere"
        self.EXTRACTING_CUDA_FILES = "Extragerea fișierelor CUDA din {0} în {1}"
        self.DOWNLOAD_ERROR = "Eroare de descărcare: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Afișarea meniului contextual al sarcinii"
        self.SHOWING_PROPERTIES_FOR_TASK = "Afișarea proprietăților pentru sarcina: {0}"
        self.CANCELLING_TASK = "Anularea sarcinii: {0}"
        self.CANCELED = "Anulat"
        self.DELETING_TASK = "Ștergerea sarcinii: {0}"
        self.LOADING_MODELS = "Încărcarea modelelor"
        self.LOADED_MODELS = "{0} modele încărcate"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Răsfoirea directorului de modele"
        self.SELECT_MODELS_DIRECTORY = "Selectați directorul de modele"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Răsfoirea directorului de ieșire"
        self.SELECT_OUTPUT_DIRECTORY = "Selectați directorul de ieșire"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Răsfoirea directorului de jurnale"
        self.SELECT_LOGS_DIRECTORY = "Selectați directorul de jurnale"
        self.BROWSING_FOR_IMATRIX_FILE = "Răsfoirea fișierului IMatrix"
        self.SELECT_IMATRIX_FILE = "Selectați fișierul IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Utilizare CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validarea intrărilor de cuantizare"
        self.MODELS_PATH_REQUIRED = "Calea modelelor este obligatorie"
        self.OUTPUT_PATH_REQUIRED = "Calea ieșirii este obligatorie"
        self.LOGS_PATH_REQUIRED = "Calea jurnalelor este obligatorie"
        self.STARTING_MODEL_QUANTIZATION = "Pornirea cuantizării modelului"
        self.INPUT_FILE_NOT_EXIST = "Fișierul de intrare '{0}' nu există."
        self.QUANTIZING_MODEL_TO = "Cuantizarea {0} la {1}"
        self.QUANTIZATION_TASK_STARTED = (
            "Sarcina de cuantizare a fost pornită pentru {0}"
        )
        self.ERROR_STARTING_QUANTIZATION = "Eroare la pornirea cuantizării: {0}"
        self.UPDATING_MODEL_INFO = "Actualizarea informațiilor despre model: {0}"
        self.TASK_FINISHED = "Sarcină finalizată: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Afișarea detaliilor sarcinii pentru: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Răsfoirea fișierului de date IMatrix"
        self.SELECT_DATA_FILE = "Selectați fișierul de date"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Răsfoirea fișierului de model IMatrix"
        self.SELECT_MODEL_FILE = "Selectați fișierul model"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Răsfoirea fișierului de ieșire IMatrix"
        self.SELECT_OUTPUT_FILE = "Selectați fișierul de ieșire"
        self.STARTING_IMATRIX_GENERATION = "Pornirea generării IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Calea backendului nu există: {0}"
        self.GENERATING_IMATRIX = "Generarea IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Eroare la pornirea generării IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = (
            "Sarcina de generare IMatrix a fost pornită"
        )
        self.ERROR_MESSAGE = "Eroare: {0}"
        self.TASK_ERROR = "Eroare de sarcină: {0}"
        self.APPLICATION_CLOSING = "Închiderea aplicației"
        self.APPLICATION_CLOSED = "Aplicație închisă"
        self.SELECT_QUANTIZATION_TYPE = "Selectați tipul de cuantizare"
        self.ALLOWS_REQUANTIZING = (
            "Permite recuantizarea tensorilor care au fost deja cuantizați"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Va lăsa output.weight necuantizat (sau recuantizat)"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Dezactivați mixurile k-quant și cuantizați toți tensorii la același tip"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Utilizați datele din fișier ca matrice de importanță pentru optimizările de cuantizare"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Utilizați matricea de importanță pentru acești tensori"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Nu utilizați matricea de importanță pentru acești tensori"
        )
        self.OUTPUT_TENSOR_TYPE = "Tipul tensorului de ieșire:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Utilizați acest tip pentru tensorul output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Tipul de încorporare a tokenului:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Utilizați acest tip pentru tensorul de încorporări ale tokenului"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Va genera modelul cuantizat în aceleași fragmente ca și intrarea"
        )
        self.OVERRIDE_MODEL_METADATA = "Suprascrieți metadatele modelului"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Fișier de date de intrare pentru generarea IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modelul de cuantizat"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Calea de ieșire pentru IMatrix generat"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Cât de des să salvați IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Setați valoarea de descărcare GPU (-ngl)"
        self.COMPLETED = "Finalizat"
        self.REFRESH_MODELS = "Odśwież modele"


class _Czech(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Automatický kvantizátor modelů GGUF)"
        self.RAM_USAGE = "Využití RAM:"
        self.CPU_USAGE = "Využití CPU:"
        self.BACKEND = "Backend Llama.cpp:"
        self.REFRESH_BACKENDS = "Obnovit backendy"
        self.MODELS_PATH = "Cesta k modelům:"
        self.OUTPUT_PATH = "Výstupní cesta:"
        self.LOGS_PATH = "Cesta k logům:"
        self.BROWSE = "Procházet"
        self.AVAILABLE_MODELS = "Dostupné modely:"
        self.QUANTIZATION_TYPE = "Typ kvantizace:"
        self.ALLOW_REQUANTIZE = "Povolit rekvantizaci"
        self.LEAVE_OUTPUT_TENSOR = "Ponechat výstupní tenzor"
        self.PURE = "Čistý"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Zahrnout váhy:"
        self.EXCLUDE_WEIGHTS = "Vyloučit váhy:"
        self.USE_OUTPUT_TENSOR_TYPE = "Použít typ výstupního tenzoru"
        self.USE_TOKEN_EMBEDDING_TYPE = "Použít typ vkládání tokenů"
        self.KEEP_SPLIT = "Zachovat rozdělení"
        self.KV_OVERRIDES = "Přepsání KV:"
        self.ADD_NEW_OVERRIDE = "Přidat nové přepsání"
        self.QUANTIZE_MODEL = "Kvantizovat model"
        self.SAVE_PRESET = "Uložit předvolbu"
        self.LOAD_PRESET = "Načíst předvolbu"
        self.TASKS = "Úkoly:"
        self.DOWNLOAD_LLAMACPP = "Stáhnout llama.cpp"
        self.SELECT_RELEASE = "Vybrat verzi:"
        self.SELECT_ASSET = "Vybrat aktivum:"
        self.EXTRACT_CUDA_FILES = "Extrahovat soubory CUDA"
        self.SELECT_CUDA_BACKEND = "Vybrat backend CUDA:"
        self.DOWNLOAD = "Stáhnout"
        self.IMATRIX_GENERATION = "Generování IMatrix"
        self.DATA_FILE = "Datový soubor:"
        self.MODEL = "Model:"
        self.OUTPUT = "Výstup:"
        self.OUTPUT_FREQUENCY = "Frekvence výstupu:"
        self.GPU_OFFLOAD = "Odlehčení GPU:"
        self.AUTO = "Automaticky"
        self.GENERATE_IMATRIX = "Generovat IMatrix"
        self.ERROR = "Chyba"
        self.WARNING = "Varování"
        self.PROPERTIES = "Vlastnosti"
        self.CANCEL = "Zrušit"
        self.RESTART = "Restartovat"
        self.DELETE = "Smazat"
        self.CONFIRM_DELETION = "Jste si jisti, že chcete smazat tento úkol?"
        self.TASK_RUNNING_WARNING = (
            "Některé úkoly stále běží. Jste si jisti, že chcete ukončit?"
        )
        self.YES = "Ano"
        self.NO = "Ne"
        self.DOWNLOAD_COMPLETE = "Stahování dokončeno"
        self.CUDA_EXTRACTION_FAILED = "Extrahování CUDA se nezdařilo"
        self.PRESET_SAVED = "Předvolba uložena"
        self.PRESET_LOADED = "Předvolba načtena"
        self.NO_ASSET_SELECTED = "Nebylo vybráno žádné aktivum"
        self.DOWNLOAD_FAILED = "Stahování se nezdařilo"
        self.NO_BACKEND_SELECTED = "Nebyl vybrán žádný backend"
        self.NO_MODEL_SELECTED = "Nebyl vybrán žádný model"
        self.REFRESH_RELEASES = "Obnovit verze"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nebyly nalezeny žádné vhodné backendy CUDA"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Binární soubor llama.cpp byl stažen a extrahován do {0}\nSoubory CUDA extrahovány do {1}"
        self.CUDA_FILES_EXTRACTED = "Soubory CUDA extrahovány do"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nebyl nalezen žádný vhodný backend CUDA pro extrakci"
        )
        self.ERROR_FETCHING_RELEASES = "Chyba při načítání verzí: {0}"
        self.CONFIRM_DELETION_TITLE = "Potvrdit smazání"
        self.LOG_FOR = "Log pro {0}"
        self.ALL_FILES = "Všechny soubory (*)"
        self.GGUF_FILES = "Soubory GGUF (*.gguf)"
        self.DAT_FILES = "Soubory DAT (*.dat)"
        self.JSON_FILES = "Soubory JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Nepodařilo se načíst předvolbu: {0}"
        self.INITIALIZING_AUTOGGUF = "Inicializace aplikace AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Inicializace AutoGGUF dokončena"
        self.REFRESHING_BACKENDS = "Obnovování backendů"
        self.NO_BACKENDS_AVAILABLE = "Žádné dostupné backendy"
        self.FOUND_VALID_BACKENDS = "Nalezeno {0} platných backendů"
        self.SAVING_PRESET = "Ukládání předvolby"
        self.PRESET_SAVED_TO = "Předvolba uložena do {0}"
        self.LOADING_PRESET = "Načítání předvolby"
        self.PRESET_LOADED_FROM = "Předvolba načtena z {0}"
        self.ADDING_KV_OVERRIDE = "Přidávání přepsání KV: {0}"
        self.SAVING_TASK_PRESET = "Ukládání předvolby úkolu pro {0}"
        self.TASK_PRESET_SAVED = "Předvolba úkolu uložena"
        self.TASK_PRESET_SAVED_TO = "Předvolba úkolu uložena do {0}"
        self.RESTARTING_TASK = "Restartování úkolu: {0}"
        self.IN_PROGRESS = "Probíhá"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Stahování dokončeno. Extrahováno do: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Binární soubor llama.cpp byl stažen a extrahován do {0}\nSoubory CUDA extrahovány do {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nebyl nalezen žádný vhodný backend CUDA pro extrakci"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Binární soubor llama.cpp byl stažen a extrahován do {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Obnovování verzí llama.cpp"
        self.UPDATING_ASSET_LIST = "Aktualizace seznamu aktiv"
        self.UPDATING_CUDA_OPTIONS = "Aktualizace možností CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Zahájení stahování llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Aktualizace backendů CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Nebyl vybrán žádný backend CUDA pro extrakci"
        self.EXTRACTING_CUDA_FILES = "Extrahování souborů CUDA z {0} do {1}"
        self.DOWNLOAD_ERROR = "Chyba stahování: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Zobrazení kontextové nabídky úkolu"
        self.SHOWING_PROPERTIES_FOR_TASK = "Zobrazení vlastností úkolu: {0}"
        self.CANCELLING_TASK = "Zrušení úkolu: {0}"
        self.CANCELED = "Zrušeno"
        self.DELETING_TASK = "Mazání úkolu: {0}"
        self.LOADING_MODELS = "Načítání modelů"
        self.LOADED_MODELS = "Načteno {0} modelů"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Procházení adresáře modelů"
        self.SELECT_MODELS_DIRECTORY = "Vyberte adresář modelů"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Procházení výstupního adresáře"
        self.SELECT_OUTPUT_DIRECTORY = "Vyberte výstupní adresář"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Procházení adresáře logů"
        self.SELECT_LOGS_DIRECTORY = "Vyberte adresář logů"
        self.BROWSING_FOR_IMATRIX_FILE = "Procházení souboru IMatrix"
        self.SELECT_IMATRIX_FILE = "Vyberte soubor IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Využití CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Ověřování vstupů kvantizace"
        self.MODELS_PATH_REQUIRED = "Cesta k modelům je vyžadována"
        self.OUTPUT_PATH_REQUIRED = "Výstupní cesta je vyžadována"
        self.LOGS_PATH_REQUIRED = "Cesta k logům je vyžadována"
        self.STARTING_MODEL_QUANTIZATION = "Spuštění kvantizace modelu"
        self.INPUT_FILE_NOT_EXIST = "Vstupní soubor '{0}' neexistuje."
        self.QUANTIZING_MODEL_TO = "Kvantizace {0} na {1}"
        self.QUANTIZATION_TASK_STARTED = "Úkol kvantizace spuštěn pro {0}"
        self.ERROR_STARTING_QUANTIZATION = "Chyba při spuštění kvantizace: {0}"
        self.UPDATING_MODEL_INFO = "Aktualizace informací o modelu: {0}"
        self.TASK_FINISHED = "Úkol dokončen: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Zobrazení detailů úkolu pro: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Procházení datového souboru IMatrix"
        self.SELECT_DATA_FILE = "Vyberte datový soubor"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Procházení souboru modelu IMatrix"
        self.SELECT_MODEL_FILE = "Vyberte soubor modelu"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Procházení výstupního souboru IMatrix"
        self.SELECT_OUTPUT_FILE = "Vyberte výstupní soubor"
        self.STARTING_IMATRIX_GENERATION = "Spuštění generování IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Cesta backendu neexistuje: {0}"
        self.GENERATING_IMATRIX = "Generování IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Chyba při spuštění generování IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Úkol generování IMatrix spuštěn"
        self.ERROR_MESSAGE = "Chyba: {0}"
        self.TASK_ERROR = "Chyba úkolu: {0}"
        self.APPLICATION_CLOSING = "Zavírání aplikace"
        self.APPLICATION_CLOSED = "Aplikace zavřena"
        self.SELECT_QUANTIZATION_TYPE = "Vyberte typ kvantizace"
        self.ALLOWS_REQUANTIZING = (
            "Umožňuje rekvantizovat tenzory, které již byly kvantizovány"
        )
        self.LEAVE_OUTPUT_WEIGHT = (
            "Ponechá output.weight nekvantizovaný (nebo rekvantizovaný)"
        )
        self.DISABLE_K_QUANT_MIXTURES = (
            "Zakázat k-kvantové směsi a kvantizovat všechny tenzory na stejný typ"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "Použít data v souboru jako matici důležitosti pro optimalizace kvantizace"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Použít matici důležitosti pro tyto tenzory"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Nepoužívat matici důležitosti pro tyto tenzory"
        )
        self.OUTPUT_TENSOR_TYPE = "Typ výstupního tenzoru:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Použít tento typ pro tenzor output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Typ vkládání tokenů:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Použít tento typ pro tenzor vkládání tokenů"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Vygeneruje kvantizovaný model ve stejných fragmentech jako vstup"
        )
        self.OVERRIDE_MODEL_METADATA = "Přepsat metadata modelu"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Vstupní datový soubor pro generování IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Model, který má být kvantizován"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Výstupní cesta pro generovaný IMatrix"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Jak často ukládat IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Nastavit hodnotu odlehčení GPU (-ngl)"
        self.COMPLETED = "Dokončeno"
        self.REFRESH_MODELS = "Reîmprospătează modelele"


class _CanadianFrench(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Quantificateur automatique de modèles GGUF)"
        self.RAM_USAGE = "Utilisation de la RAM :"  # Spacing
        self.CPU_USAGE = "Utilisation du CPU :"  # Spacing
        self.BACKEND = "Moteur d'arrière-plan Llama.cpp :"  # Spacing and terminology
        self.REFRESH_BACKENDS = "Actualiser les moteurs d'arrière-plan"
        self.MODELS_PATH = "Chemin des modèles :"  # Spacing
        self.OUTPUT_PATH = "Chemin de sortie :"  # Spacing
        self.LOGS_PATH = "Chemin des journaux :"  # Spacing
        self.BROWSE = "Parcourir"
        self.AVAILABLE_MODELS = "Modèles disponibles :"  # Spacing
        self.QUANTIZATION_TYPE = "Type de quantification :"  # Spacing
        self.ALLOW_REQUANTIZE = "Autoriser la requantification"
        self.LEAVE_OUTPUT_TENSOR = "Laisser le tenseur de sortie"
        self.PURE = "Pur"
        self.IMATRIX = "IMatrix :"  # Spacing
        self.INCLUDE_WEIGHTS = "Inclure les poids :"  # Spacing
        self.EXCLUDE_WEIGHTS = "Exclure les poids :"  # Spacing
        self.USE_OUTPUT_TENSOR_TYPE = "Utiliser le type de tenseur de sortie"
        self.USE_TOKEN_EMBEDDING_TYPE = "Utiliser le type d'intégration de jeton"
        self.KEEP_SPLIT = "Conserver la division"
        self.KV_OVERRIDES = "Remplacements KV :"  # Spacing
        self.ADD_NEW_OVERRIDE = "Ajouter un nouveau remplacement"
        self.QUANTIZE_MODEL = "Quantifier le modèle"
        self.SAVE_PRESET = "Enregistrer le préréglage"
        self.LOAD_PRESET = "Charger le préréglage"
        self.TASKS = "Tâches :"  # Spacing
        self.DOWNLOAD_LLAMACPP = "Télécharger llama.cpp"
        self.SELECT_RELEASE = "Sélectionner la version :"  # Spacing
        self.SELECT_ASSET = "Sélectionner l'actif :"  # Spacing
        self.EXTRACT_CUDA_FILES = "Extraire les fichiers CUDA"
        self.SELECT_CUDA_BACKEND = "Sélectionner le backend CUDA :"  # Spacing
        self.DOWNLOAD = "Télécharger"
        self.IMATRIX_GENERATION = "Génération d'IMatrix"
        self.DATA_FILE = "Fichier de données :"  # Spacing
        self.MODEL = "Modèle :"  # Spacing
        self.OUTPUT = "Sortie :"  # Spacing
        self.OUTPUT_FREQUENCY = "Fréquence de sortie :"  # Spacing
        self.GPU_OFFLOAD = "Déchargement GPU :"  # Spacing
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Générer IMatrix"
        self.ERROR = "Erreur"
        self.WARNING = "Avertissement"
        self.PROPERTIES = "Propriétés"
        self.CANCEL = "Annuler"
        self.RESTART = "Redémarrer"
        self.DELETE = "Supprimer"
        self.CONFIRM_DELETION = (
            "Êtes-vous sûr de vouloir supprimer cette tâche ?"  # Spacing
        )
        self.TASK_RUNNING_WARNING = "Certaines tâches sont encore en cours d'exécution. Êtes-vous sûr de vouloir quitter ?"  # Spacing
        self.YES = "Oui"
        self.NO = "Non"
        self.DOWNLOAD_COMPLETE = "Téléchargement terminé"
        self.CUDA_EXTRACTION_FAILED = "Échec de l'extraction CUDA"
        self.PRESET_SAVED = "Préréglage enregistré"
        self.PRESET_LOADED = "Préréglage chargé"
        self.NO_ASSET_SELECTED = "Aucun actif sélectionné"
        self.DOWNLOAD_FAILED = "Échec du téléchargement"
        self.NO_BACKEND_SELECTED = "Aucun backend sélectionné"
        self.NO_MODEL_SELECTED = "Aucun modèle sélectionné"
        self.REFRESH_RELEASES = "Actualiser les versions"
        self.NO_SUITABLE_CUDA_BACKENDS = "Aucun backend CUDA approprié trouvé"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Le fichier binaire llama.cpp a été téléchargé et extrait dans {0}\nLes fichiers CUDA ont été extraits dans {1}"
        self.CUDA_FILES_EXTRACTED = "Les fichiers CUDA ont été extraits dans"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Aucun backend CUDA approprié trouvé pour l'extraction"
        )
        self.ERROR_FETCHING_RELEASES = (
            "Erreur lors de la récupération des versions : {0}"  # Spacing
        )
        self.CONFIRM_DELETION_TITLE = "Confirmer la suppression"
        self.LOG_FOR = "Journal pour {0}"
        self.ALL_FILES = "Tous les fichiers (*)"
        self.GGUF_FILES = "Fichiers GGUF (*.gguf)"
        self.DAT_FILES = "Fichiers DAT (*.dat)"
        self.JSON_FILES = "Fichiers JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Échec du chargement du préréglage : {0}"  # Spacing
        self.INITIALIZING_AUTOGGUF = "Initialisation de l'application AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Initialisation d'AutoGGUF terminée"
        self.REFRESHING_BACKENDS = "Actualisation des moteurs d'arrière-plan"
        self.NO_BACKENDS_AVAILABLE = "Aucun moteur d'arrière-plan disponible"
        self.FOUND_VALID_BACKENDS = "{0} moteurs d'arrière-plan valides trouvés"
        self.SAVING_PRESET = "Enregistrement du préréglage"
        self.PRESET_SAVED_TO = "Préréglage enregistré dans {0}"
        self.LOADING_PRESET = "Chargement du préréglage"
        self.PRESET_LOADED_FROM = "Préréglage chargé depuis {0}"
        self.ADDING_KV_OVERRIDE = "Ajout de remplacement KV : {0}"  # Spacing
        self.SAVING_TASK_PRESET = "Enregistrement du préréglage de tâche pour {0}"
        self.TASK_PRESET_SAVED = "Préréglage de tâche enregistré"
        self.TASK_PRESET_SAVED_TO = "Préréglage de tâche enregistré dans {0}"
        self.RESTARTING_TASK = "Redémarrage de la tâche : {0}"  # Spacing
        self.IN_PROGRESS = "En cours"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "Téléchargement terminé. Extrait dans : {0}"  # Spacing
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Le fichier binaire llama.cpp a été téléchargé et extrait dans {0}\nLes fichiers CUDA ont été extraits dans {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Aucun backend CUDA approprié trouvé pour l'extraction"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Le fichier binaire llama.cpp a été téléchargé et extrait dans {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Actualisation des versions de llama.cpp"
        self.UPDATING_ASSET_LIST = "Mise à jour de la liste des actifs"
        self.UPDATING_CUDA_OPTIONS = "Mise à jour des options CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Démarrage du téléchargement de llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Mise à jour des backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = (
            "Aucun backend CUDA sélectionné pour l'extraction"
        )
        self.EXTRACTING_CUDA_FILES = "Extraction des fichiers CUDA de {0} à {1}"
        self.DOWNLOAD_ERROR = "Erreur de téléchargement : {0}"  # Spacing
        self.SHOWING_TASK_CONTEXT_MENU = "Affichage du menu contextuel de la tâche"
        self.SHOWING_PROPERTIES_FOR_TASK = (
            "Affichage des propriétés de la tâche : {0}"  # Spacing
        )
        self.CANCELLING_TASK = "Annulation de la tâche : {0}"  # Spacing
        self.CANCELED = "Annulée"
        self.DELETING_TASK = "Suppression de la tâche : {0}"  # Spacing
        self.LOADING_MODELS = "Chargement des modèles"
        self.LOADED_MODELS = "{0} modèles chargés"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Navigation dans le répertoire des modèles"
        self.SELECT_MODELS_DIRECTORY = "Sélectionner le répertoire des modèles"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Navigation dans le répertoire de sortie"
        self.SELECT_OUTPUT_DIRECTORY = "Sélectionner le répertoire de sortie"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Navigation dans le répertoire des journaux"
        self.SELECT_LOGS_DIRECTORY = "Sélectionner le répertoire des journaux"
        self.BROWSING_FOR_IMATRIX_FILE = "Navigation dans le fichier IMatrix"
        self.SELECT_IMATRIX_FILE = "Sélectionner le fichier IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} Mo / {2} Mo)"
        self.CPU_USAGE_FORMAT = "Utilisation du CPU : {0:.1f}%"  # Spacing
        self.VALIDATING_QUANTIZATION_INPUTS = "Validation des entrées de quantification"
        self.MODELS_PATH_REQUIRED = "Le chemin des modèles est requis"
        self.OUTPUT_PATH_REQUIRED = "Le chemin de sortie est requis"
        self.LOGS_PATH_REQUIRED = "Le chemin des journaux est requis"
        self.STARTING_MODEL_QUANTIZATION = "Démarrage de la quantification du modèle"
        self.INPUT_FILE_NOT_EXIST = "Le fichier d'entrée '{0}' n'existe pas."
        self.QUANTIZING_MODEL_TO = "Quantification de {0} en {1}"
        self.QUANTIZATION_TASK_STARTED = "Tâche de quantification démarrée pour {0}"
        self.ERROR_STARTING_QUANTIZATION = (
            "Erreur lors du démarrage de la quantification : {0}"  # Spacing
        )
        self.UPDATING_MODEL_INFO = (
            "Mise à jour des informations sur le modèle : {0}"  # Spacing
        )
        self.TASK_FINISHED = "Tâche terminée : {0}"  # Spacing
        self.SHOWING_TASK_DETAILS_FOR = (
            "Affichage des détails de la tâche pour : {0}"  # Spacing
        )
        self.BROWSING_FOR_IMATRIX_DATA_FILE = (
            "Navigation dans le fichier de données IMatrix"
        )
        self.SELECT_DATA_FILE = "Sélectionner le fichier de données"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = (
            "Navigation dans le fichier de modèle IMatrix"
        )
        self.SELECT_MODEL_FILE = "Sélectionner le fichier de modèle"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = (
            "Navigation dans le fichier de sortie IMatrix"
        )
        self.SELECT_OUTPUT_FILE = "Sélectionner le fichier de sortie"
        self.STARTING_IMATRIX_GENERATION = "Démarrage de la génération d'IMatrix"
        self.BACKEND_PATH_NOT_EXIST = (
            "Le chemin du backend n'existe pas : {0}"  # Spacing
        )
        self.GENERATING_IMATRIX = "Génération d'IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Erreur lors du démarrage de la génération d'IMatrix : {0}"  # Spacing
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Tâche de génération d'IMatrix démarrée"
        self.ERROR_MESSAGE = "Erreur : {0}"  # Spacing
        self.TASK_ERROR = "Erreur de tâche : {0}"  # Spacing
        self.APPLICATION_CLOSING = "Fermeture de l'application"
        self.APPLICATION_CLOSED = "Application fermée"
        self.SELECT_QUANTIZATION_TYPE = "Sélectionnez le type de quantification"
        self.ALLOWS_REQUANTIZING = (
            "Permet de requantifier les tenseurs qui ont déjà été quantifiés"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Laissera output.weight non (re)quantifié"
        self.DISABLE_K_QUANT_MIXTURES = "Désactiver les mélanges k-quant et quantifier tous les tenseurs du même type"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Utiliser les données du fichier comme matrice d'importance pour les optimisations de quant"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Utiliser la matrice d'importance pour ces tenseurs"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Ne pas utiliser la matrice d'importance pour ces tenseurs"
        )
        self.OUTPUT_TENSOR_TYPE = "Type de tenseur de sortie :"  # Spacing
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Utiliser ce type pour le tenseur output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Type d'intégration de jeton :"  # Spacing
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Utiliser ce type pour le tenseur d'intégration de jetons"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Générera le modèle quantifié dans les mêmes fragments que l'entrée"
        )
        self.OVERRIDE_MODEL_METADATA = "Remplacer les métadonnées du modèle"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Fichier de données d'entrée pour la génération d'IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modèle à quantifier"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Chemin de sortie pour l'IMatrix généré"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Fréquence d'enregistrement de l'IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Définir la valeur de déchargement GPU (-ngl)"
        self.COMPLETED = "Terminé"
        self.REFRESH_MODELS = "Rafraîchir les modèles"


class _Portuguese_PT(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Quantificador Automático de Modelos GGUF)"
        self.RAM_USAGE = "Utilização de RAM:"
        self.CPU_USAGE = "Utilização da CPU:"
        self.BACKEND = "Backend Llama.cpp:"
        self.REFRESH_BACKENDS = "Atualizar Backends"
        self.MODELS_PATH = "Caminho dos Modelos:"
        self.OUTPUT_PATH = "Caminho de Saída:"
        self.LOGS_PATH = "Caminho dos Logs:"
        self.BROWSE = "Navegar"
        self.AVAILABLE_MODELS = "Modelos Disponíveis:"
        self.QUANTIZATION_TYPE = "Tipo de Quantização:"
        self.ALLOW_REQUANTIZE = "Permitir Requantização"
        self.LEAVE_OUTPUT_TENSOR = "Manter Tensor de Saída"
        self.PURE = "Puro"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Incluir Pesos:"
        self.EXCLUDE_WEIGHTS = "Excluir Pesos:"
        self.USE_OUTPUT_TENSOR_TYPE = "Usar Tipo de Tensor de Saída"
        self.USE_TOKEN_EMBEDDING_TYPE = "Usar Tipo de Incorporação de Token"
        self.KEEP_SPLIT = "Manter Divisão"
        self.KV_OVERRIDES = "Substituições KV:"
        self.ADD_NEW_OVERRIDE = "Adicionar Nova Substituição"
        self.QUANTIZE_MODEL = "Quantizar Modelo"
        self.SAVE_PRESET = "Guardar Predefinição"
        self.LOAD_PRESET = "Carregar Predefinição"
        self.TASKS = "Tarefas:"
        self.DOWNLOAD_LLAMACPP = "Descarregar llama.cpp"
        self.SELECT_RELEASE = "Selecionar Versão:"
        self.SELECT_ASSET = "Selecionar Ativo:"
        self.EXTRACT_CUDA_FILES = "Extrair Ficheiros CUDA"
        self.SELECT_CUDA_BACKEND = "Selecionar Backend CUDA:"
        self.DOWNLOAD = "Descarregar"
        self.IMATRIX_GENERATION = "Geração de IMatrix"
        self.DATA_FILE = "Ficheiro de Dados:"
        self.MODEL = "Modelo:"
        self.OUTPUT = "Saída:"
        self.OUTPUT_FREQUENCY = "Frequência de Saída:"
        self.GPU_OFFLOAD = "Offload da GPU:"
        self.AUTO = "Automático"
        self.GENERATE_IMATRIX = "Gerar IMatrix"
        self.ERROR = "Erro"
        self.WARNING = "Aviso"
        self.PROPERTIES = "Propriedades"
        self.CANCEL = "Cancelar"
        self.RESTART = "Reiniciar"
        self.DELETE = "Eliminar"
        self.CONFIRM_DELETION = "Tem a certeza de que pretende eliminar esta tarefa?"
        self.TASK_RUNNING_WARNING = "Algumas tarefas ainda estão em execução. Tem a certeza de que pretende sair?"
        self.YES = "Sim"
        self.NO = "Não"
        self.DOWNLOAD_COMPLETE = "Transferência Concluída"
        self.CUDA_EXTRACTION_FAILED = "Falha na Extração do CUDA"
        self.PRESET_SAVED = "Predefinição Guardada"
        self.PRESET_LOADED = "Predefinição Carregada"
        self.NO_ASSET_SELECTED = "Nenhum ativo selecionado"
        self.DOWNLOAD_FAILED = "Falha na transferência"
        self.NO_BACKEND_SELECTED = "Nenhum backend selecionado"
        self.NO_MODEL_SELECTED = "Nenhum modelo selecionado"
        self.REFRESH_RELEASES = "Atualizar Versões"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nenhum backend CUDA adequado encontrado"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Binário llama.cpp transferido e extraído para {0}\nFicheiros CUDA extraídos para {1}"
        self.CUDA_FILES_EXTRACTED = "Ficheiros CUDA extraídos para"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nenhum backend CUDA adequado encontrado para extração"
        )
        self.ERROR_FETCHING_RELEASES = "Erro ao obter versões: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirmar Eliminação"
        self.LOG_FOR = "Log para {0}"
        self.ALL_FILES = "Todos os Ficheiros (*)"
        self.GGUF_FILES = "Ficheiros GGUF (*.gguf)"
        self.DAT_FILES = "Ficheiros DAT (*.dat)"
        self.JSON_FILES = "Ficheiros JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Falha ao carregar a predefinição: {0}"
        self.INITIALIZING_AUTOGGUF = "A inicializar a aplicação AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Inicialização do AutoGGUF concluída"
        self.REFRESHING_BACKENDS = "A atualizar backends"
        self.NO_BACKENDS_AVAILABLE = "Nenhum backend disponível"
        self.FOUND_VALID_BACKENDS = "{0} backends válidos encontrados"
        self.SAVING_PRESET = "A guardar predefinição"
        self.PRESET_SAVED_TO = "Predefinição guardada em {0}"
        self.LOADING_PRESET = "A carregar predefinição"
        self.PRESET_LOADED_FROM = "Predefinição carregada de {0}"
        self.ADDING_KV_OVERRIDE = "A adicionar substituição KV: {0}"
        self.SAVING_TASK_PRESET = "A guardar predefinição de tarefa para {0}"
        self.TASK_PRESET_SAVED = "Predefinição de Tarefa Guardada"
        self.TASK_PRESET_SAVED_TO = "Predefinição de tarefa guardada em {0}"
        self.RESTARTING_TASK = "A reiniciar tarefa: {0}"
        self.IN_PROGRESS = "Em Andamento"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = (
            "Transferência concluída. Extraído para: {0}"
        )
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Binário llama.cpp transferido e extraído para {0}\nFicheiros CUDA extraídos para {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nenhum backend CUDA adequado encontrado para extração"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Binário llama.cpp transferido e extraído para {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "A atualizar versões do llama.cpp"
        self.UPDATING_ASSET_LIST = "A atualizar lista de ativos"
        self.UPDATING_CUDA_OPTIONS = "A atualizar opções CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "A iniciar transferência do llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "A atualizar backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Nenhum backend CUDA selecionado para extração"
        self.EXTRACTING_CUDA_FILES = "A extrair ficheiros CUDA de {0} para {1}"
        self.DOWNLOAD_ERROR = "Erro de transferência: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "A exibir menu de contexto da tarefa"
        self.SHOWING_PROPERTIES_FOR_TASK = "A exibir propriedades para a tarefa: {0}"
        self.CANCELLING_TASK = "A cancelar tarefa: {0}"
        self.CANCELED = "Cancelado"
        self.DELETING_TASK = "A eliminar tarefa: {0}"
        self.LOADING_MODELS = "A carregar modelos"
        self.LOADED_MODELS = "{0} modelos carregados"
        self.BROWSING_FOR_MODELS_DIRECTORY = "A navegar pelo diretório de modelos"
        self.SELECT_MODELS_DIRECTORY = "Selecionar Diretório de Modelos"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "A navegar pelo diretório de saída"
        self.SELECT_OUTPUT_DIRECTORY = "Selecionar Diretório de Saída"
        self.BROWSING_FOR_LOGS_DIRECTORY = "A navegar pelo diretório de logs"
        self.SELECT_LOGS_DIRECTORY = "Selecionar Diretório de Logs"
        self.BROWSING_FOR_IMATRIX_FILE = "A navegar pelo ficheiro IMatrix"
        self.SELECT_IMATRIX_FILE = "Selecionar Ficheiro IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Utilização da CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "A validar entradas de quantização"
        self.MODELS_PATH_REQUIRED = "O caminho dos modelos é obrigatório"
        self.OUTPUT_PATH_REQUIRED = "O caminho de saída é obrigatório"
        self.LOGS_PATH_REQUIRED = "O caminho dos logs é obrigatório"
        self.STARTING_MODEL_QUANTIZATION = "A iniciar a quantização do modelo"
        self.INPUT_FILE_NOT_EXIST = "O ficheiro de entrada '{0}' não existe."
        self.QUANTIZING_MODEL_TO = "A quantizar {0} para {1}"
        self.QUANTIZATION_TASK_STARTED = "Tarefa de quantização iniciada para {0}"
        self.ERROR_STARTING_QUANTIZATION = "Erro ao iniciar a quantização: {0}"
        self.UPDATING_MODEL_INFO = "A atualizar informações do modelo: {0}"
        self.TASK_FINISHED = "Tarefa concluída: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "A mostrar detalhes da tarefa para: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "A navegar pelo ficheiro de dados IMatrix"
        self.SELECT_DATA_FILE = "Selecionar Ficheiro de Dados"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = (
            "A navegar pelo ficheiro de modelo IMatrix"
        )
        self.SELECT_MODEL_FILE = "Selecionar Ficheiro de Modelo"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = (
            "A navegar pelo ficheiro de saída IMatrix"
        )
        self.SELECT_OUTPUT_FILE = "Selecionar Ficheiro de Saída"
        self.STARTING_IMATRIX_GENERATION = "A iniciar a geração de IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "O caminho do backend não existe: {0}"
        self.GENERATING_IMATRIX = "A gerar IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Erro ao iniciar a geração de IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Tarefa de geração de IMatrix iniciada"
        self.ERROR_MESSAGE = "Erro: {0}"
        self.TASK_ERROR = "Erro de tarefa: {0}"
        self.APPLICATION_CLOSING = "A fechar a aplicação"
        self.APPLICATION_CLOSED = "Aplicação fechada"
        self.SELECT_QUANTIZATION_TYPE = "Selecione o tipo de quantização"
        self.ALLOWS_REQUANTIZING = (
            "Permite requantizar tensores que já foram quantizados"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Deixará output.weight não (re)quantizado"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Desativar misturas k-quant e quantizar todos os tensores para o mesmo tipo"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Usar os dados no ficheiro como matriz de importância para otimizações de quantização"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Usar matriz de importância para estes tensores"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Não usar matriz de importância para estes tensores"
        )
        self.OUTPUT_TENSOR_TYPE = "Tipo de Tensor de Saída:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Usar este tipo para o tensor output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Tipo de Incorporação de Token:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Usar este tipo para o tensor de incorporações de token"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Irá gerar o modelo quantizado nos mesmos shards da entrada"
        )
        self.OVERRIDE_MODEL_METADATA = "Substituir metadados do modelo"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Ficheiro de dados de entrada para geração de IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Modelo a ser quantizado"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Caminho de saída para o IMatrix gerado"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Com que frequência guardar o IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Definir valor de offload da GPU (-ngl)"
        self.COMPLETED = "Concluído"
        self.REFRESH_MODELS = "Atualizar modelos"


class _Greek(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Αυτόματος Κβαντιστής Μοντέλων GGUF)"
        self.RAM_USAGE = "Χρήση RAM:"
        self.CPU_USAGE = "Χρήση CPU:"
        self.BACKEND = "Backend Llama.cpp:"
        self.REFRESH_BACKENDS = "Ανανέωση Backends"
        self.MODELS_PATH = "Διαδρομή Μοντέλων:"
        self.OUTPUT_PATH = "Διαδρομή Εξόδου:"
        self.LOGS_PATH = "Διαδρομή Αρχείων Καταγραφής:"
        self.BROWSE = "Περιήγηση"
        self.AVAILABLE_MODELS = "Διαθέσιμα Μοντέλα:"
        self.QUANTIZATION_TYPE = "Τύπος Κβαντισμού:"
        self.ALLOW_REQUANTIZE = "Να Επιτρέπεται η Επανακβάντιση"
        self.LEAVE_OUTPUT_TENSOR = "Διατήρηση Tensor Εξόδου"
        self.PURE = "Καθαρό"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Συμπερίληψη Βαρών:"
        self.EXCLUDE_WEIGHTS = "Εξαίρεση Βαρών:"
        self.USE_OUTPUT_TENSOR_TYPE = "Χρήση Τύπου Tensor Εξόδου"
        self.USE_TOKEN_EMBEDDING_TYPE = "Χρήση Τύπου Ενσωμάτωσης Token"
        self.KEEP_SPLIT = "Διατήρηση Διαίρεσης"
        self.KV_OVERRIDES = "Υπερβάσεις KV:"
        self.ADD_NEW_OVERRIDE = "Προσθήκη Νέας Υπέρβασης"
        self.QUANTIZE_MODEL = "Κβάντιση Μοντέλου"
        self.SAVE_PRESET = "Αποθήκευση Προεπιλογής"
        self.LOAD_PRESET = "Φόρτωση Προεπιλογής"
        self.TASKS = "Εργασίες:"
        self.DOWNLOAD_LLAMACPP = "Λήψη llama.cpp"
        self.SELECT_RELEASE = "Επιλογή Έκδοσης:"
        self.SELECT_ASSET = "Επιλογή Στοιχείου:"
        self.EXTRACT_CUDA_FILES = "Εξαγωγή Αρχείων CUDA"
        self.SELECT_CUDA_BACKEND = "Επιλογή Backend CUDA:"
        self.DOWNLOAD = "Λήψη"
        self.IMATRIX_GENERATION = "Δημιουργία IMatrix"
        self.DATA_FILE = "Αρχείο Δεδομένων:"
        self.MODEL = "Μοντέλο:"
        self.OUTPUT = "Έξοδος:"
        self.OUTPUT_FREQUENCY = "Συχνότητα Εξόδου:"
        self.GPU_OFFLOAD = "Εκφόρτωση GPU:"
        self.AUTO = "Αυτόματο"
        self.GENERATE_IMATRIX = "Δημιουργία IMatrix"
        self.ERROR = "Σφάλμα"
        self.WARNING = "Προειδοποίηση"
        self.PROPERTIES = "Ιδιότητες"
        self.CANCEL = "Ακύρωση"
        self.RESTART = "Επανεκκίνηση"
        self.DELETE = "Διαγραφή"
        self.CONFIRM_DELETION = (
            "Είστε βέβαιοι ότι θέλετε να διαγράψετε αυτήν την εργασία;"
        )
        self.TASK_RUNNING_WARNING = "Ορισμένες εργασίες εκτελούνται ακόμη. Είστε βέβαιοι ότι θέλετε να τερματίσετε;"
        self.YES = "Ναι"
        self.NO = "Όχι"
        self.DOWNLOAD_COMPLETE = "Η Λήψη Ολοκληρώθηκε"
        self.CUDA_EXTRACTION_FAILED = "Αποτυχία Εξαγωγής CUDA"
        self.PRESET_SAVED = "Η Προεπιλογή Αποθηκεύτηκε"
        self.PRESET_LOADED = "Η Προεπιλογή Φορτώθηκε"
        self.NO_ASSET_SELECTED = "Δεν Έχει Επιλεγεί Στοιχείο"
        self.DOWNLOAD_FAILED = "Αποτυχία Λήψης"
        self.NO_BACKEND_SELECTED = "Δεν Έχει Επιλεγεί Backend"
        self.NO_MODEL_SELECTED = "Δεν Έχει Επιλεγεί Μοντέλο"
        self.REFRESH_RELEASES = "Ανανέωση Εκδόσεων"
        self.NO_SUITABLE_CUDA_BACKENDS = "Δεν Βρέθηκαν Κατάλληλα Backends CUDA"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "Το Δυαδικό Αρχείο llama.cpp Λήφθηκε και Εξήχθη στο {0}\nΤα Αρχεία CUDA Εξήχθησαν στο {1}"
        self.CUDA_FILES_EXTRACTED = "Τα Αρχεία CUDA Εξήχθησαν στο"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Δεν Βρέθηκε Κατάλληλο Backend CUDA για Εξαγωγή"
        )
        self.ERROR_FETCHING_RELEASES = "Σφάλμα κατά την Ανάκτηση Εκδόσεων: {0}"
        self.CONFIRM_DELETION_TITLE = "Επιβεβαίωση Διαγραφής"
        self.LOG_FOR = "Αρχείο Καταγραφής για {0}"
        self.ALL_FILES = "Όλα τα Αρχεία (*)"
        self.GGUF_FILES = "Αρχεία GGUF (*.gguf)"
        self.DAT_FILES = "Αρχεία DAT (*.dat)"
        self.JSON_FILES = "Αρχεία JSON (*.json)"
        self.FAILED_LOAD_PRESET = "Αποτυχία Φόρτωσης Προεπιλογής: {0}"
        self.INITIALIZING_AUTOGGUF = "Εκκίνηση Εφαρμογής AutoGGUF"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "Η Εκκίνηση του AutoGGUF Ολοκληρώθηκε"
        self.REFRESHING_BACKENDS = "Ανανέωση Backends"
        self.NO_BACKENDS_AVAILABLE = "Δεν Υπάρχουν Διαθέσιμα Backends"
        self.FOUND_VALID_BACKENDS = "Βρέθηκαν {0} Έγκυρα Backends"
        self.SAVING_PRESET = "Αποθήκευση Προεπιλογής"
        self.PRESET_SAVED_TO = "Η Προεπιλογή Αποθηκεύτηκε στο {0}"
        self.LOADING_PRESET = "Φόρτωση Προεπιλογής"
        self.PRESET_LOADED_FROM = "Η Προεπιλογή Φορτώθηκε από το {0}"
        self.ADDING_KV_OVERRIDE = "Προσθήκη Υπέρβασης KV: {0}"
        self.SAVING_TASK_PRESET = "Αποθήκευση Προεπιλογής Εργασίας για {0}"
        self.TASK_PRESET_SAVED = "Η Προεπιλογή Εργασίας Αποθηκεύτηκε"
        self.TASK_PRESET_SAVED_TO = "Η Προεπιλογή Εργασίας Αποθηκεύτηκε στο {0}"
        self.RESTARTING_TASK = "Επανεκκίνηση Εργασίας: {0}"
        self.IN_PROGRESS = "Σε Εξέλιξη"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Η Λήψη Ολοκληρώθηκε. Εξήχθη στο: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "Το Δυαδικό Αρχείο llama.cpp Λήφθηκε και Εξήχθη στο {0}\nΤα Αρχεία CUDA Εξήχθησαν στο {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Δεν Βρέθηκε Κατάλληλο Backend CUDA για Εξαγωγή"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "Το Δυαδικό Αρχείο llama.cpp Λήφθηκε και Εξήχθη στο {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Ανανέωση Εκδόσεων llama.cpp"
        self.UPDATING_ASSET_LIST = "Ενημέρωση Λίστας Στοιχείων"
        self.UPDATING_CUDA_OPTIONS = "Ενημέρωση Επιλογών CUDA"
        self.STARTING_LLAMACPP_DOWNLOAD = "Έναρξη Λήψης llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "Ενημέρωση Backends CUDA"
        self.NO_CUDA_BACKEND_SELECTED = "Δεν Έχει Επιλεγεί Backend CUDA για Εξαγωγή"
        self.EXTRACTING_CUDA_FILES = "Εξαγωγή Αρχείων CUDA από {0} στο {1}"
        self.DOWNLOAD_ERROR = "Σφάλμα Λήψης: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Εμφάνιση Μενού Περιεχομένου Εργασίας"
        self.SHOWING_PROPERTIES_FOR_TASK = "Εμφάνιση Ιδιοτήτων για την Εργασία: {0}"
        self.CANCELLING_TASK = "Ακύρωση Εργασίας: {0}"
        self.CANCELED = "Ακυρώθηκε"
        self.DELETING_TASK = "Διαγραφή Εργασίας: {0}"
        self.LOADING_MODELS = "Φόρτωση Μοντέλων"
        self.LOADED_MODELS = "{0} Μοντέλα Φορτώθηκαν"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Περιήγηση σε Φάκελο Μοντέλων"
        self.SELECT_MODELS_DIRECTORY = "Επιλέξτε Φάκελο Μοντέλων"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Περιήγηση σε Φάκελο Εξόδου"
        self.SELECT_OUTPUT_DIRECTORY = "Επιλέξτε Φάκελο Εξόδου"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Περιήγηση σε Φάκελο Αρχείων Καταγραφής"
        self.SELECT_LOGS_DIRECTORY = "Επιλέξτε Φάκελο Αρχείων Καταγραφής"
        self.BROWSING_FOR_IMATRIX_FILE = "Περιήγηση σε Αρχείο IMatrix"
        self.SELECT_IMATRIX_FILE = "Επιλέξτε Αρχείο IMatrix"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "Χρήση CPU: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Επικύρωση Εισόδων Κβαντισμού"
        self.MODELS_PATH_REQUIRED = "Απαιτείται η Διαδρομή Μοντέλων"
        self.OUTPUT_PATH_REQUIRED = "Απαιτείται η Διαδρομή Εξόδου"
        self.LOGS_PATH_REQUIRED = "Απαιτείται η Διαδρομή Αρχείων Καταγραφής"
        self.STARTING_MODEL_QUANTIZATION = "Έναρξη Κβαντισμού Μοντέλου"
        self.INPUT_FILE_NOT_EXIST = "Το Αρχείο Εισόδου '{0}' Δεν Υπάρχει."
        self.QUANTIZING_MODEL_TO = "Κβάντιση του {0} σε {1}"
        self.QUANTIZATION_TASK_STARTED = "Η Εργασία Κβαντισμού Ξεκίνησε για {0}"
        self.ERROR_STARTING_QUANTIZATION = "Σφάλμα κατά την Έναρξη Κβαντισμού: {0}"
        self.UPDATING_MODEL_INFO = "Ενημέρωση Πληροφοριών Μοντέλου: {0}"
        self.TASK_FINISHED = "Η Εργασία Ολοκληρώθηκε: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Εμφάνιση Λεπτομερειών Εργασίας για: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Περιήγηση σε Αρχείο Δεδομένων IMatrix"
        self.SELECT_DATA_FILE = "Επιλέξτε Αρχείο Δεδομένων"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Περιήγηση σε Αρχείο Μοντέλου IMatrix"
        self.SELECT_MODEL_FILE = "Επιλέξτε Αρχείο Μοντέλου"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Περιήγηση σε Αρχείο Εξόδου IMatrix"
        self.SELECT_OUTPUT_FILE = "Επιλέξτε Αρχείο Εξόδου"
        self.STARTING_IMATRIX_GENERATION = "Έναρξη Δημιουργίας IMatrix"
        self.BACKEND_PATH_NOT_EXIST = "Η Διαδρομή Backend Δεν Υπάρχει: {0}"
        self.GENERATING_IMATRIX = "Δημιουργία IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Σφάλμα κατά την Έναρξη Δημιουργίας IMatrix: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "Η Εργασία Δημιουργίας IMatrix Ξεκίνησε"
        self.ERROR_MESSAGE = "Σφάλμα: {0}"
        self.TASK_ERROR = "Σφάλμα Εργασίας: {0}"
        self.APPLICATION_CLOSING = "Κλείσιμο Εφαρμογής"
        self.APPLICATION_CLOSED = "Η Εφαρμογή Έκλεισε"
        self.SELECT_QUANTIZATION_TYPE = "Επιλέξτε τον τύπο κβαντισμού"
        self.ALLOWS_REQUANTIZING = (
            "Επιτρέπει την επανακβάντιση τανυστών που έχουν ήδη κβαντιστεί"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Θα αφήσει το output.weight χωρίς (επανα)κβάντιση"
        self.DISABLE_K_QUANT_MIXTURES = "Απενεργοποιήστε τα μείγματα k-quant και κβαντίστε όλους τους τανυστές στον ίδιο τύπο"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Χρησιμοποιήστε τα δεδομένα στο αρχείο ως πίνακα σημασίας για βελτιστοποιήσεις κβαντισμού"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Χρησιμοποιήστε τον πίνακα σημασίας για αυτούς τους τανυστές"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Μην χρησιμοποιείτε τον πίνακα σημασίας για αυτούς τους τανυστές"
        )
        self.OUTPUT_TENSOR_TYPE = "Τύπος Tensor Εξόδου:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Χρησιμοποιήστε αυτόν τον τύπο για τον τανυστή output.weight"
        )
        self.TOKEN_EMBEDDING_TYPE = "Τύπος Ενσωμάτωσης Token:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Χρησιμοποιήστε αυτόν τον τύπο για τον τανυστή ενσωματώσεων token"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Θα δημιουργήσει το κβαντισμένο μοντέλο στα ίδια θραύσματα με την είσοδο"
        )
        self.OVERRIDE_MODEL_METADATA = "Αντικατάσταση μεταδεδομένων μοντέλου"
        self.INPUT_DATA_FILE_FOR_IMATRIX = (
            "Αρχείο δεδομένων εισόδου για τη δημιουργία IMatrix"
        )
        self.MODEL_TO_BE_QUANTIZED = "Μοντέλο προς κβάντιση"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = (
            "Διαδρομή εξόδου για το δημιουργημένο IMatrix"
        )
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Πόσο συχνά να αποθηκεύεται το IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Ορίστε την τιμή εκφόρτωσης GPU (-ngl)"
        self.COMPLETED = "Ολοκληρώθηκε"
        self.REFRESH_MODELS = "Ανανέωση μοντέλων"


class _Hungarian(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (Automatizált GGUF modell kvantáló)"
        self.RAM_USAGE = "RAM használat:"
        self.CPU_USAGE = "CPU használat:"
        self.BACKEND = "Llama.cpp háttérrendszer:"
        self.REFRESH_BACKENDS = "Háttérrendszerek frissítése"
        self.MODELS_PATH = "Modellek elérési útja:"
        self.OUTPUT_PATH = "Kimeneti útvonal:"
        self.LOGS_PATH = "Naplók elérési útja:"
        self.BROWSE = "Tallózás"
        self.AVAILABLE_MODELS = "Elérhető modellek:"
        self.QUANTIZATION_TYPE = "Kvantálási típus:"
        self.ALLOW_REQUANTIZE = "Újrakvantálás engedélyezése"
        self.LEAVE_OUTPUT_TENSOR = "Kimeneti tenzor meghagyása"
        self.PURE = "Tiszta"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Súlyok belefoglalása:"
        self.EXCLUDE_WEIGHTS = "Súlyok kizárása:"
        self.USE_OUTPUT_TENSOR_TYPE = "Kimeneti tenzor típusának használata"
        self.USE_TOKEN_EMBEDDING_TYPE = "Token beágyazási típusának használata"
        self.KEEP_SPLIT = "Felosztás megtartása"
        self.KV_OVERRIDES = "KV felülbírálások:"
        self.ADD_NEW_OVERRIDE = "Új felülbírálás hozzáadása"
        self.QUANTIZE_MODEL = "Modell kvantálása"
        self.SAVE_PRESET = "Esetbeállítás mentése"
        self.LOAD_PRESET = "Esetbeállítás betöltése"
        self.TASKS = "Feladatok:"
        self.DOWNLOAD_LLAMACPP = "llama.cpp letöltése"
        self.SELECT_RELEASE = "Kiadás kiválasztása:"
        self.SELECT_ASSET = "Eszköz kiválasztása:"
        self.EXTRACT_CUDA_FILES = "CUDA fájlok kibontása"
        self.SELECT_CUDA_BACKEND = "CUDA háttérrendszer kiválasztása:"
        self.DOWNLOAD = "Letöltés"
        self.IMATRIX_GENERATION = "IMatrix generálás"
        self.DATA_FILE = "Adatfájl:"
        self.MODEL = "Modell:"
        self.OUTPUT = "Kimenet:"
        self.OUTPUT_FREQUENCY = "Kimeneti frekvencia:"
        self.GPU_OFFLOAD = "GPU tehermentesítés:"
        self.AUTO = "Automatikus"
        self.GENERATE_IMATRIX = "IMatrix generálása"
        self.ERROR = "Hiba"
        self.WARNING = "Figyelmeztetés"
        self.PROPERTIES = "Tulajdonságok"
        self.CANCEL = "Mégse"
        self.RESTART = "Újraindítás"
        self.DELETE = "Törlés"
        self.CONFIRM_DELETION = "Biztosan törölni szeretné ezt a feladatot?"
        self.TASK_RUNNING_WARNING = "Néhány feladat még fut. Biztosan kilép?"
        self.YES = "Igen"
        self.NO = "Nem"
        self.DOWNLOAD_COMPLETE = "Letöltés befejeződött"
        self.CUDA_EXTRACTION_FAILED = "CUDA kibontás sikertelen"
        self.PRESET_SAVED = "Esetbeállítás mentve"
        self.PRESET_LOADED = "Esetbeállítás betöltve"
        self.NO_ASSET_SELECTED = "Nincs kiválasztott eszköz"
        self.DOWNLOAD_FAILED = "Letöltés sikertelen"
        self.NO_BACKEND_SELECTED = "Nincs kiválasztott háttérrendszer"
        self.NO_MODEL_SELECTED = "Nincs kiválasztott modell"
        self.REFRESH_RELEASES = "Kiadások frissítése"
        self.NO_SUITABLE_CUDA_BACKENDS = "Nem található megfelelő CUDA háttérrendszer"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "A llama.cpp bináris fájl letöltve és kibontva ide: {0}\nA CUDA fájlok kibontva ide: {1}"
        self.CUDA_FILES_EXTRACTED = "A CUDA fájlok kibontva ide:"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "Nem található megfelelő CUDA háttérrendszer a kibontáshoz"
        )
        self.ERROR_FETCHING_RELEASES = "Hiba a kiadások lekérdezésekor: {0}"
        self.CONFIRM_DELETION_TITLE = "Törlés megerősítése"
        self.LOG_FOR = "Napló a következőhöz: {0}"
        self.ALL_FILES = "Minden fájl (*)"
        self.GGUF_FILES = "GGUF fájlok (*.gguf)"
        self.DAT_FILES = "DAT fájlok (*.dat)"
        self.JSON_FILES = "JSON fájlok (*.json)"
        self.FAILED_LOAD_PRESET = "Az esetbeállítás betöltése sikertelen: {0}"
        self.INITIALIZING_AUTOGGUF = "Az AutoGGUF alkalmazás inicializálása"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = (
            "Az AutoGGUF inicializálása befejeződött"
        )
        self.REFRESHING_BACKENDS = "Háttérrendszerek frissítése"
        self.NO_BACKENDS_AVAILABLE = "Nincsenek elérhető háttérrendszerek"
        self.FOUND_VALID_BACKENDS = "{0} érvényes háttérrendszer található"
        self.SAVING_PRESET = "Esetbeállítás mentése"
        self.PRESET_SAVED_TO = "Esetbeállítás mentve ide: {0}"
        self.LOADING_PRESET = "Esetbeállítás betöltése"
        self.PRESET_LOADED_FROM = "Esetbeállítás betöltve innen: {0}"
        self.ADDING_KV_OVERRIDE = "KV felülbírálás hozzáadása: {0}"
        self.SAVING_TASK_PRESET = "Feladat esetbeállítás mentése ehhez: {0}"
        self.TASK_PRESET_SAVED = "Feladat esetbeállítás mentve"
        self.TASK_PRESET_SAVED_TO = "Feladat esetbeállítás mentve ide: {0}"
        self.RESTARTING_TASK = "Feladat újraindítása: {0}"
        self.IN_PROGRESS = "Folyamatban"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Letöltés befejeződött. Kibontva ide: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "A llama.cpp bináris fájl letöltve és kibontva ide: {0}\nA CUDA fájlok kibontva ide: {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "Nem található megfelelő CUDA háttérrendszer a kibontáshoz"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "A llama.cpp bináris fájl letöltve és kibontva ide: {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "A llama.cpp kiadások frissítése"
        self.UPDATING_ASSET_LIST = "Eszközlista frissítése"
        self.UPDATING_CUDA_OPTIONS = "CUDA beállítások frissítése"
        self.STARTING_LLAMACPP_DOWNLOAD = "A llama.cpp letöltésének megkezdése"
        self.UPDATING_CUDA_BACKENDS = "CUDA háttérrendszerek frissítése"
        self.NO_CUDA_BACKEND_SELECTED = (
            "Nincs kiválasztott CUDA háttérrendszer a kibontáshoz"
        )
        self.EXTRACTING_CUDA_FILES = "CUDA fájlok kibontása innen: {0} ide: {1}"
        self.DOWNLOAD_ERROR = "Letöltési hiba: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Feladat helyi menüjének megjelenítése"
        self.SHOWING_PROPERTIES_FOR_TASK = "Feladat tulajdonságainak megjelenítése: {0}"
        self.CANCELLING_TASK = "Feladat megszakítása: {0}"
        self.CANCELED = "Megszakítva"
        self.DELETING_TASK = "Feladat törlése: {0}"
        self.LOADING_MODELS = "Modellek betöltése"
        self.LOADED_MODELS = "{0} modell betöltve"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Modellek könyvtárának tallózása"
        self.SELECT_MODELS_DIRECTORY = "Modellek könyvtárának kiválasztása"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Kimeneti könyvtár tallózása"
        self.SELECT_OUTPUT_DIRECTORY = "Kimeneti könyvtár kiválasztása"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Naplók könyvtárának tallózása"
        self.SELECT_LOGS_DIRECTORY = "Naplók könyvtárának kiválasztása"
        self.BROWSING_FOR_IMATRIX_FILE = "IMatrix fájl tallózása"
        self.SELECT_IMATRIX_FILE = "IMatrix fájl kiválasztása"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU használat: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Kvantálási bemenetek ellenőrzése"
        self.MODELS_PATH_REQUIRED = "A modellek elérési útja kötelező"
        self.OUTPUT_PATH_REQUIRED = "A kimeneti útvonal kötelező"
        self.LOGS_PATH_REQUIRED = "A naplók elérési útja kötelező"
        self.STARTING_MODEL_QUANTIZATION = "Modell kvantálásának indítása"
        self.INPUT_FILE_NOT_EXIST = "A bemeneti fájl '{0}' nem létezik."
        self.QUANTIZING_MODEL_TO = "{0} kvantálása erre: {1}"
        self.QUANTIZATION_TASK_STARTED = "Kvantálási feladat elindítva ehhez: {0}"
        self.ERROR_STARTING_QUANTIZATION = "Hiba a kvantálás indításakor: {0}"
        self.UPDATING_MODEL_INFO = "Modellinformációk frissítése: {0}"
        self.TASK_FINISHED = "Feladat befejezve: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Feladat részleteinek megjelenítése ehhez: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "IMatrix adatfájl tallózása"
        self.SELECT_DATA_FILE = "Adatfájl kiválasztása"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "IMatrix modellfájl tallózása"
        self.SELECT_MODEL_FILE = "Modellfájl kiválasztása"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "IMatrix kimeneti fájl tallózása"
        self.SELECT_OUTPUT_FILE = "Kimeneti fájl kiválasztása"
        self.STARTING_IMATRIX_GENERATION = "IMatrix generálásának indítása"
        self.BACKEND_PATH_NOT_EXIST = "A háttérrendszer elérési útja nem létezik: {0}"
        self.GENERATING_IMATRIX = "IMatrix generálása"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Hiba az IMatrix generálásának indításakor: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix generálási feladat elindítva"
        self.ERROR_MESSAGE = "Hiba: {0}"
        self.TASK_ERROR = "Feladat hiba: {0}"
        self.APPLICATION_CLOSING = "Alkalmazás bezárása"
        self.APPLICATION_CLOSED = "Alkalmazás bezárva"
        self.SELECT_QUANTIZATION_TYPE = "Válassza ki a kvantálási típust"
        self.ALLOWS_REQUANTIZING = (
            "Lehetővé teszi a már kvantált tenzorok újrakvantálását"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Az output.weight-et (újra)kvantálatlanul hagyja"
        self.DISABLE_K_QUANT_MIXTURES = "Tiltsa le a k-kvant keverékeket, és kvantálja az összes tenzort ugyanarra a típusra"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Használja a fájlban lévő adatokat fontossági mátrixként a kvantálási optimalizálásokhoz"
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Használja a fontossági mátrixot ezekre a tenzorokra"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Ne használja a fontossági mátrixot ezekre a tenzorokra"
        )
        self.OUTPUT_TENSOR_TYPE = "Kimeneti tenzor típusa:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Használja ezt a típust az output.weight tenzorhoz"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token beágyazási típusa:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Használja ezt a típust a token beágyazási tenzorhoz"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = "A kvantált modellt ugyanazokban a szegmensekben fogja generálni, mint a bemenet"
        self.OVERRIDE_MODEL_METADATA = "Modell metaadatok felülbírálása"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix generáláshoz bemeneti adatfájl"
        self.MODEL_TO_BE_QUANTIZED = "Kvantálandó modell"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "A generált IMatrix kimeneti útvonala"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "Milyen gyakran mentse az IMatrixot"
        self.SET_GPU_OFFLOAD_VALUE = "GPU tehermentesítési érték beállítása (-ngl)"
        self.COMPLETED = "Befejezve"
        self.REFRESH_MODELS = "Modellek frissítése"


class _BritishEnglish(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (automated GGUF model quantiser)"
        self.RAM_USAGE = "RAM Usage:"
        self.CPU_USAGE = "CPU Usage:"
        self.BACKEND = "Llama.cpp Backend:"
        self.REFRESH_BACKENDS = "Refresh Backends"
        self.MODELS_PATH = "Models Path:"
        self.OUTPUT_PATH = "Output Path:"
        self.LOGS_PATH = "Logs Path:"
        self.BROWSE = "Browse"
        self.AVAILABLE_MODELS = "Available Models:"
        self.QUANTIZATION_TYPE = "Quantisation Type:"  # Note the British spelling
        self.ALLOW_REQUANTIZE = "Allow Requantise"
        self.LEAVE_OUTPUT_TENSOR = "Leave Output Tensor"
        self.PURE = "Pure"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Include Weights:"
        self.EXCLUDE_WEIGHTS = "Exclude Weights:"
        self.USE_OUTPUT_TENSOR_TYPE = "Use Output Tensor Type"
        self.USE_TOKEN_EMBEDDING_TYPE = "Use Token Embedding Type"
        self.KEEP_SPLIT = "Keep Split"
        self.KV_OVERRIDES = "KV Overrides:"
        self.ADD_NEW_OVERRIDE = "Add new override"
        self.QUANTIZE_MODEL = "Quantise Model"  # Note the British spelling
        self.SAVE_PRESET = "Save Preset"
        self.LOAD_PRESET = "Load Preset"
        self.TASKS = "Tasks:"
        self.DOWNLOAD_LLAMACPP = "Download llama.cpp"
        self.SELECT_RELEASE = "Select Release:"
        self.SELECT_ASSET = "Select Asset:"
        self.EXTRACT_CUDA_FILES = "Extract CUDA files"
        self.SELECT_CUDA_BACKEND = "Select CUDA Backend:"
        self.DOWNLOAD = "Download"
        self.IMATRIX_GENERATION = "IMatrix Generation"
        self.DATA_FILE = "Data File:"
        self.MODEL = "Model:"
        self.OUTPUT = "Output:"
        self.OUTPUT_FREQUENCY = "Output Frequency:"
        self.GPU_OFFLOAD = "GPU Offload:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Generate IMatrix"
        self.ERROR = "Error"
        self.WARNING = "Warning"
        self.PROPERTIES = "Properties"
        self.CANCEL = "Cancel"
        self.RESTART = "Restart"
        self.DELETE = "Delete"
        self.CONFIRM_DELETION = "Are you sure you want to delete this task?"
        self.TASK_RUNNING_WARNING = (
            "Some tasks are still running. Are you sure you want to quit?"
        )
        self.YES = "Yes"
        self.NO = "No"
        self.DOWNLOAD_COMPLETE = "Download Complete"
        self.CUDA_EXTRACTION_FAILED = "CUDA Extraction Failed"
        self.PRESET_SAVED = "Preset Saved"
        self.PRESET_LOADED = "Preset Loaded"
        self.NO_ASSET_SELECTED = "No asset selected"
        self.DOWNLOAD_FAILED = "Download failed"
        self.NO_BACKEND_SELECTED = "No backend selected"
        self.NO_MODEL_SELECTED = "No model selected"
        self.REFRESH_RELEASES = "Refresh Releases"
        self.NO_SUITABLE_CUDA_BACKENDS = "No suitable CUDA backends found"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp binary downloaded and extracted to {0}\nCUDA files extracted to {1}"
        self.CUDA_FILES_EXTRACTED = "CUDA files extracted to"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "No suitable CUDA backend found for extraction"
        )
        self.ERROR_FETCHING_RELEASES = "Error fetching releases: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirm Deletion"
        self.LOG_FOR = "Log for {0}"
        self.ALL_FILES = "All Files (*)"
        self.GGUF_FILES = "GGUF Files (*.gguf)"
        self.DAT_FILES = "DAT Files (*.dat)"
        self.JSON_FILES = "JSON Files (*.json)"
        self.FAILED_LOAD_PRESET = "Failed to load preset: {0}"
        self.INITIALIZING_AUTOGGUF = (
            "Initialising AutoGGUF application"  # Note the British spelling
        )
        self.AUTOGGUF_INITIALIZATION_COMPLETE = (
            "AutoGGUF initialisation complete"  # Note the British spelling
        )
        self.REFRESHING_BACKENDS = "Refreshing backends"
        self.NO_BACKENDS_AVAILABLE = "No backends available"
        self.FOUND_VALID_BACKENDS = "Found {0} valid backends"
        self.SAVING_PRESET = "Saving preset"
        self.PRESET_SAVED_TO = "Preset saved to {0}"
        self.LOADING_PRESET = "Loading preset"
        self.PRESET_LOADED_FROM = "Preset loaded from {0}"
        self.ADDING_KV_OVERRIDE = "Adding KV override: {0}"
        self.SAVING_TASK_PRESET = "Saving task preset for {0}"
        self.TASK_PRESET_SAVED = "Task Preset Saved"
        self.TASK_PRESET_SAVED_TO = "Task preset saved to {0}"
        self.RESTARTING_TASK = "Restarting task: {0}"
        self.IN_PROGRESS = "In Progress"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download finished. Extracted to: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp binary downloaded and extracted to {0}\nCUDA files extracted to {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "No suitable CUDA backend found for extraction"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binary downloaded and extracted to {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Refreshing llama.cpp releases"
        self.UPDATING_ASSET_LIST = "Updating asset list"
        self.UPDATING_CUDA_OPTIONS = "Updating CUDA options"
        self.STARTING_LLAMACPP_DOWNLOAD = "Starting llama.cpp download"
        self.UPDATING_CUDA_BACKENDS = "Updating CUDA backends"
        self.NO_CUDA_BACKEND_SELECTED = "No CUDA backend selected for extraction"
        self.EXTRACTING_CUDA_FILES = "Extracting CUDA files from {0} to {1}"
        self.DOWNLOAD_ERROR = "Download error: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Showing task context menu"
        self.SHOWING_PROPERTIES_FOR_TASK = "Showing properties for task: {0}"
        self.CANCELLING_TASK = "Cancelling task: {0}"
        self.CANCELED = "Cancelled"
        self.DELETING_TASK = "Deleting task: {0}"
        self.LOADING_MODELS = "Loading models"
        self.LOADED_MODELS = "Loaded {0} models"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Browsing for models directory"
        self.SELECT_MODELS_DIRECTORY = "Select Models Directory"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Browsing for output directory"
        self.SELECT_OUTPUT_DIRECTORY = "Select Output Directory"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Browsing for logs directory"
        self.SELECT_LOGS_DIRECTORY = "Select Logs Directory"
        self.BROWSING_FOR_IMATRIX_FILE = "Browsing for IMatrix file"
        self.SELECT_IMATRIX_FILE = "Select IMatrix File"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU Usage: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = (
            "Validating quantisation inputs"  # Note the British spelling
        )
        self.MODELS_PATH_REQUIRED = "Models path is required"
        self.OUTPUT_PATH_REQUIRED = "Output path is required"
        self.LOGS_PATH_REQUIRED = "Logs path is required"
        self.STARTING_MODEL_QUANTIZATION = (
            "Starting model quantisation"  # Note the British spelling
        )
        self.INPUT_FILE_NOT_EXIST = "Input file '{0}' does not exist."
        self.QUANTIZING_MODEL_TO = "Quantizing {0} to {1}"
        self.QUANTIZATION_TASK_STARTED = (
            "Quantisation task started for {0}"  # Note the British spelling
        )
        self.ERROR_STARTING_QUANTIZATION = (
            "Error starting quantisation: {0}"  # Note the British spelling
        )
        self.UPDATING_MODEL_INFO = "Updating model info: {0}"
        self.TASK_FINISHED = "Task finished: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Showing task details for: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Browsing for IMatrix data file"
        self.SELECT_DATA_FILE = "Select Data File"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Browsing for IMatrix model file"
        self.SELECT_MODEL_FILE = "Select Model File"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Browsing for IMatrix output file"
        self.SELECT_OUTPUT_FILE = "Select Output File"
        self.STARTING_IMATRIX_GENERATION = "Starting IMatrix generation"
        self.BACKEND_PATH_NOT_EXIST = "Backend path does not exist: {0}"
        self.GENERATING_IMATRIX = "Generating IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Error starting IMatrix generation: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix generation task started"
        self.ERROR_MESSAGE = "Error: {0}"
        self.TASK_ERROR = "Task error: {0}"
        self.APPLICATION_CLOSING = "Application closing"
        self.APPLICATION_CLOSED = "Application closed"
        self.SELECT_QUANTIZATION_TYPE = (
            "Select the quantisation type"  # Note the British spelling
        )
        self.ALLOWS_REQUANTIZING = "Allows requantising tensors that have already been quantised"  # Note the British spelling
        self.LEAVE_OUTPUT_WEIGHT = "Will leave output.weight un(re)quantised"
        self.DISABLE_K_QUANT_MIXTURES = "Disable k-quant mixtures and quantise all tensors to the same type"  # Note the British spelling
        self.USE_DATA_AS_IMPORTANCE_MATRIX = "Use data in file as importance matrix for quant optimisations"  # Note the British spelling
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Use importance matrix for these tensors"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Don't use importance matrix for these tensors"
        )
        self.OUTPUT_TENSOR_TYPE = "Output Tensor Type:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Use this type for the output.weight tensor"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token Embedding Type:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Use this type for the token embeddings tensor"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = "Will generate quantised model in the same shards as input"  # Note the British spelling
        self.OVERRIDE_MODEL_METADATA = "Override model metadata"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Input data file for IMatrix generation"
        self.MODEL_TO_BE_QUANTIZED = (
            "Model to be quantised"  # Note the British spelling
        )
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Output path for the generated IMatrix"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "How often to save the IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Set GPU offload value (-ngl)"
        self.COMPLETED = "Completed"
        self.REFRESH_MODELS = "Refresh Models"


class _IndianEnglish(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (automated GGUF model quantizer)"
        self.RAM_USAGE = "RAM Usage:"
        self.CPU_USAGE = "CPU Usage:"
        self.BACKEND = "Llama.cpp Backend:"
        self.REFRESH_BACKENDS = "Refresh Backends"
        self.MODELS_PATH = "Models Path:"
        self.OUTPUT_PATH = "Output Path:"
        self.LOGS_PATH = "Logs Path:"
        self.BROWSE = "Browse"
        self.AVAILABLE_MODELS = "Available Models:"
        self.QUANTIZATION_TYPE = "Quantization Type:"
        self.ALLOW_REQUANTIZE = "Allow Requantize"
        self.LEAVE_OUTPUT_TENSOR = "Leave Output Tensor"
        self.PURE = "Pure"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Include Weights:"
        self.EXCLUDE_WEIGHTS = "Exclude Weights:"
        self.USE_OUTPUT_TENSOR_TYPE = "Use Output Tensor Type"
        self.USE_TOKEN_EMBEDDING_TYPE = "Use Token Embedding Type"
        self.KEEP_SPLIT = "Keep Split"
        self.KV_OVERRIDES = "KV Overrides:"
        self.ADD_NEW_OVERRIDE = "Add new override"
        self.QUANTIZE_MODEL = "Quantize Model"
        self.SAVE_PRESET = "Save Preset"
        self.LOAD_PRESET = "Load Preset"
        self.TASKS = "Tasks:"
        self.DOWNLOAD_LLAMACPP = "Download llama.cpp"
        self.SELECT_RELEASE = "Select Release:"
        self.SELECT_ASSET = "Select Asset:"
        self.EXTRACT_CUDA_FILES = "Extract CUDA files"
        self.SELECT_CUDA_BACKEND = "Select CUDA Backend:"
        self.DOWNLOAD = "Download"
        self.IMATRIX_GENERATION = "IMatrix Generation"
        self.DATA_FILE = "Data File:"
        self.MODEL = "Model:"
        self.OUTPUT = "Output:"
        self.OUTPUT_FREQUENCY = "Output Frequency:"
        self.GPU_OFFLOAD = "GPU Offload:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Generate IMatrix"
        self.ERROR = "Error"
        self.WARNING = "Warning"
        self.PROPERTIES = "Properties"
        self.CANCEL = "Cancel"
        self.RESTART = "Restart"
        self.DELETE = "Delete"
        self.CONFIRM_DELETION = "Are you sure you want to delete this task?"
        self.TASK_RUNNING_WARNING = (
            "Some tasks are still running. Are you sure you want to quit?"
        )
        self.YES = "Yes"
        self.NO = "No"
        self.DOWNLOAD_COMPLETE = "Download Complete"
        self.CUDA_EXTRACTION_FAILED = "CUDA Extraction Failed"
        self.PRESET_SAVED = "Preset Saved"
        self.PRESET_LOADED = "Preset Loaded"
        self.NO_ASSET_SELECTED = "No asset selected"
        self.DOWNLOAD_FAILED = "Download failed"
        self.NO_BACKEND_SELECTED = "No backend selected"
        self.NO_MODEL_SELECTED = "No model selected"
        self.REFRESH_RELEASES = "Refresh Releases"
        self.NO_SUITABLE_CUDA_BACKENDS = "No suitable CUDA backends found"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp binary downloaded and extracted to {0}\nCUDA files extracted to {1}"
        self.CUDA_FILES_EXTRACTED = "CUDA files extracted to"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "No suitable CUDA backend found for extraction"
        )
        self.ERROR_FETCHING_RELEASES = "Error fetching releases: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirm Deletion"
        self.LOG_FOR = "Log for {0}"
        self.ALL_FILES = "All Files (*)"
        self.GGUF_FILES = "GGUF Files (*.gguf)"
        self.DAT_FILES = "DAT Files (*.dat)"
        self.JSON_FILES = "JSON Files (*.json)"
        self.FAILED_LOAD_PRESET = "Failed to load preset: {0}"
        self.INITIALIZING_AUTOGGUF = "Initializing AutoGGUF application"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF initialization complete"
        self.REFRESHING_BACKENDS = "Refreshing backends"
        self.NO_BACKENDS_AVAILABLE = "No backends available"
        self.FOUND_VALID_BACKENDS = "Found {0} valid backends"
        self.SAVING_PRESET = "Saving preset"
        self.PRESET_SAVED_TO = "Preset saved to {0}"
        self.LOADING_PRESET = "Loading preset"
        self.PRESET_LOADED_FROM = "Preset loaded from {0}"
        self.ADDING_KV_OVERRIDE = "Adding KV override: {0}"
        self.SAVING_TASK_PRESET = "Saving task preset for {0}"
        self.TASK_PRESET_SAVED = "Task Preset Saved"
        self.TASK_PRESET_SAVED_TO = "Task preset saved to {0}"
        self.RESTARTING_TASK = "Restarting task: {0}"
        self.IN_PROGRESS = "In Progress"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download finished. Extracted to: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp binary downloaded and extracted to {0}\nCUDA files extracted to {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "No suitable CUDA backend found for extraction"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binary downloaded and extracted to {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Refreshing llama.cpp releases"
        self.UPDATING_ASSET_LIST = "Updating asset list"
        self.UPDATING_CUDA_OPTIONS = "Updating CUDA options"
        self.STARTING_LLAMACPP_DOWNLOAD = "Starting llama.cpp download"
        self.UPDATING_CUDA_BACKENDS = "Updating CUDA backends"
        self.NO_CUDA_BACKEND_SELECTED = "No CUDA backend selected for extraction"
        self.EXTRACTING_CUDA_FILES = "Extracting CUDA files from {0} to {1}"
        self.DOWNLOAD_ERROR = "Download error: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Showing task context menu"
        self.SHOWING_PROPERTIES_FOR_TASK = "Showing properties for task: {0}"
        self.CANCELLING_TASK = "Cancelling task: {0}"
        self.CANCELED = "Cancelled"
        self.DELETING_TASK = "Deleting task: {0}"
        self.LOADING_MODELS = "Loading models"
        self.LOADED_MODELS = "Loaded {0} models"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Browsing for models directory"
        self.SELECT_MODELS_DIRECTORY = "Select Models Directory"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Browsing for output directory"
        self.SELECT_OUTPUT_DIRECTORY = "Select Output Directory"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Browsing for logs directory"
        self.SELECT_LOGS_DIRECTORY = "Select Logs Directory"
        self.BROWSING_FOR_IMATRIX_FILE = "Browsing for IMatrix file"
        self.SELECT_IMATRIX_FILE = "Select IMatrix File"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU Usage: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validating quantization inputs"
        self.MODELS_PATH_REQUIRED = "Models path is required"
        self.OUTPUT_PATH_REQUIRED = "Output path is required"
        self.LOGS_PATH_REQUIRED = "Logs path is required"
        self.STARTING_MODEL_QUANTIZATION = "Starting model quantization"
        self.INPUT_FILE_NOT_EXIST = "Input file '{0}' does not exist."
        self.QUANTIZING_MODEL_TO = "Quantizing {0} to {1}"
        self.QUANTIZATION_TASK_STARTED = "Quantization task started for {0}"
        self.ERROR_STARTING_QUANTIZATION = "Error starting quantization: {0}"
        self.UPDATING_MODEL_INFO = "Updating model info: {0}"
        self.TASK_FINISHED = "Task finished: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Showing task details for: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Browsing for IMatrix data file"
        self.SELECT_DATA_FILE = "Select Data File"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Browsing for IMatrix model file"
        self.SELECT_MODEL_FILE = "Select Model File"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Browsing for IMatrix output file"
        self.SELECT_OUTPUT_FILE = "Select Output File"
        self.STARTING_IMATRIX_GENERATION = "Starting IMatrix generation"
        self.BACKEND_PATH_NOT_EXIST = "Backend path does not exist: {0}"
        self.GENERATING_IMATRIX = "Generating IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Error starting IMatrix generation: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix generation task started"
        self.ERROR_MESSAGE = "Error: {0}"
        self.TASK_ERROR = "Task error: {0}"
        self.APPLICATION_CLOSING = "Application closing"
        self.APPLICATION_CLOSED = "Application closed"
        self.SELECT_QUANTIZATION_TYPE = "Select the quantization type"
        self.ALLOWS_REQUANTIZING = (
            "Allows requantizing tensors that have already been quantized"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Will leave output.weight un(re)quantized"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Disable k-quant mixtures and quantize all tensors to the same type"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "Use data in file as importance matrix for quant optimisations"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Use importance matrix for these tensors"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Don't use importance matrix for these tensors"
        )
        self.OUTPUT_TENSOR_TYPE = "Output Tensor Type:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Use this type for the output.weight tensor"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token Embedding Type:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Use this type for the token embeddings tensor"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Will generate quantized model in the same shards as input"
        )
        self.OVERRIDE_MODEL_METADATA = "Override model metadata"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Input data file for IMatrix generation"
        self.MODEL_TO_BE_QUANTIZED = "Model to be quantized"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Output path for the generated IMatrix"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "How often to save the IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Set GPU offload value (-ngl)"
        self.COMPLETED = "Completed"
        self.REFRESH_MODELS = "Refresh Models"


class _CanadianEnglish(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF (automated GGUF model quantizer)"
        self.RAM_USAGE = "RAM Usage:"
        self.CPU_USAGE = "CPU Usage:"
        self.BACKEND = "Llama.cpp Backend:"
        self.REFRESH_BACKENDS = "Refresh Backends"
        self.MODELS_PATH = "Models Path:"
        self.OUTPUT_PATH = "Output Path:"
        self.LOGS_PATH = "Logs Path:"
        self.BROWSE = "Browse"
        self.AVAILABLE_MODELS = "Available Models:"
        self.QUANTIZATION_TYPE = "Quantization Type:"
        self.ALLOW_REQUANTIZE = "Allow Requantize"
        self.LEAVE_OUTPUT_TENSOR = "Leave Output Tensor"
        self.PURE = "Pure"
        self.IMATRIX = "IMatrix:"
        self.INCLUDE_WEIGHTS = "Include Weights:"
        self.EXCLUDE_WEIGHTS = "Exclude Weights:"
        self.USE_OUTPUT_TENSOR_TYPE = "Use Output Tensor Type"
        self.USE_TOKEN_EMBEDDING_TYPE = "Use Token Embedding Type"
        self.KEEP_SPLIT = "Keep Split"
        self.KV_OVERRIDES = "KV Overrides:"
        self.ADD_NEW_OVERRIDE = "Add new override"
        self.QUANTIZE_MODEL = "Quantize Model"
        self.SAVE_PRESET = "Save Preset"
        self.LOAD_PRESET = "Load Preset"
        self.TASKS = "Tasks:"
        self.DOWNLOAD_LLAMACPP = "Download llama.cpp"
        self.SELECT_RELEASE = "Select Release:"
        self.SELECT_ASSET = "Select Asset:"
        self.EXTRACT_CUDA_FILES = "Extract CUDA files"
        self.SELECT_CUDA_BACKEND = "Select CUDA Backend:"
        self.DOWNLOAD = "Download"
        self.IMATRIX_GENERATION = "IMatrix Generation"
        self.DATA_FILE = "Data File:"
        self.MODEL = "Model:"
        self.OUTPUT = "Output:"
        self.OUTPUT_FREQUENCY = "Output Frequency:"
        self.GPU_OFFLOAD = "GPU Offload:"
        self.AUTO = "Auto"
        self.GENERATE_IMATRIX = "Generate IMatrix"
        self.ERROR = "Error"
        self.WARNING = "Warning"
        self.PROPERTIES = "Properties"
        self.CANCEL = "Cancel"
        self.RESTART = "Restart"
        self.DELETE = "Delete"
        self.CONFIRM_DELETION = "Are you sure you want to delete this task?"
        self.TASK_RUNNING_WARNING = (
            "Some tasks are still running. Are you sure you want to quit?"
        )
        self.YES = "Yes"
        self.NO = "No"
        self.DOWNLOAD_COMPLETE = "Download Complete"
        self.CUDA_EXTRACTION_FAILED = "CUDA Extraction Failed"
        self.PRESET_SAVED = "Preset Saved"
        self.PRESET_LOADED = "Preset Loaded"
        self.NO_ASSET_SELECTED = "No asset selected"
        self.DOWNLOAD_FAILED = "Download failed"
        self.NO_BACKEND_SELECTED = "No backend selected"
        self.NO_MODEL_SELECTED = "No model selected"
        self.REFRESH_RELEASES = "Refresh Releases"
        self.NO_SUITABLE_CUDA_BACKENDS = "No suitable CUDA backends found"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = "llama.cpp binary downloaded and extracted to {0}\nCUDA files extracted to {1}"
        self.CUDA_FILES_EXTRACTED = "CUDA files extracted to"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = (
            "No suitable CUDA backend found for extraction"
        )
        self.ERROR_FETCHING_RELEASES = "Error fetching releases: {0}"
        self.CONFIRM_DELETION_TITLE = "Confirm Deletion"
        self.LOG_FOR = "Log for {0}"
        self.ALL_FILES = "All Files (*)"
        self.GGUF_FILES = "GGUF Files (*.gguf)"
        self.DAT_FILES = "DAT Files (*.dat)"
        self.JSON_FILES = "JSON Files (*.json)"
        self.FAILED_LOAD_PRESET = "Failed to load preset: {0}"
        self.INITIALIZING_AUTOGGUF = "Initializing AutoGGUF application"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF initialization complete"
        self.REFRESHING_BACKENDS = "Refreshing backends"
        self.NO_BACKENDS_AVAILABLE = "No backends available"
        self.FOUND_VALID_BACKENDS = "Found {0} valid backends"
        self.SAVING_PRESET = "Saving preset"
        self.PRESET_SAVED_TO = "Preset saved to {0}"
        self.LOADING_PRESET = "Loading preset"
        self.PRESET_LOADED_FROM = "Preset loaded from {0}"
        self.ADDING_KV_OVERRIDE = "Adding KV override: {0}"
        self.SAVING_TASK_PRESET = "Saving task preset for {0}"
        self.TASK_PRESET_SAVED = "Task Preset Saved"
        self.TASK_PRESET_SAVED_TO = "Task preset saved to {0}"
        self.RESTARTING_TASK = "Restarting task: {0}"
        self.IN_PROGRESS = "In Progress"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "Download finished. Extracted to: {0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = "llama.cpp binary downloaded and extracted to {0}\nCUDA files extracted to {1}"
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = (
            "No suitable CUDA backend found for extraction"
        )
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp binary downloaded and extracted to {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "Refreshing llama.cpp releases"
        self.UPDATING_ASSET_LIST = "Updating asset list"
        self.UPDATING_CUDA_OPTIONS = "Updating CUDA options"
        self.STARTING_LLAMACPP_DOWNLOAD = "Starting llama.cpp download"
        self.UPDATING_CUDA_BACKENDS = "Updating CUDA backends"
        self.NO_CUDA_BACKEND_SELECTED = "No CUDA backend selected for extraction"
        self.EXTRACTING_CUDA_FILES = "Extracting CUDA files from {0} to {1}"
        self.DOWNLOAD_ERROR = "Download error: {0}"
        self.SHOWING_TASK_CONTEXT_MENU = "Showing task context menu"
        self.SHOWING_PROPERTIES_FOR_TASK = "Showing properties for task: {0}"
        self.CANCELLING_TASK = "Cancelling task: {0}"
        self.CANCELED = "Cancelled"
        self.DELETING_TASK = "Deleting task: {0}"
        self.LOADING_MODELS = "Loading models"
        self.LOADED_MODELS = "Loaded {0} models"
        self.BROWSING_FOR_MODELS_DIRECTORY = "Browsing for models directory"
        self.SELECT_MODELS_DIRECTORY = "Select Models Directory"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "Browsing for output directory"
        self.SELECT_OUTPUT_DIRECTORY = "Select Output Directory"
        self.BROWSING_FOR_LOGS_DIRECTORY = "Browsing for logs directory"
        self.SELECT_LOGS_DIRECTORY = "Select Logs Directory"
        self.BROWSING_FOR_IMATRIX_FILE = "Browsing for IMatrix file"
        self.SELECT_IMATRIX_FILE = "Select IMatrix File"
        self.RAM_USAGE_FORMAT = "{0:.1f}% ({1} MB / {2} MB)"
        self.CPU_USAGE_FORMAT = "CPU Usage: {0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "Validating quantization inputs"
        self.MODELS_PATH_REQUIRED = "Models path is required"
        self.OUTPUT_PATH_REQUIRED = "Output path is required"
        self.LOGS_PATH_REQUIRED = "Logs path is required"
        self.STARTING_MODEL_QUANTIZATION = "Starting model quantization"
        self.INPUT_FILE_NOT_EXIST = "Input file '{0}' does not exist."
        self.QUANTIZING_MODEL_TO = "Quantizing {0} to {1}"
        self.QUANTIZATION_TASK_STARTED = "Quantization task started for {0}"
        self.ERROR_STARTING_QUANTIZATION = "Error starting quantization: {0}"
        self.UPDATING_MODEL_INFO = "Updating model info: {0}"
        self.TASK_FINISHED = "Task finished: {0}"
        self.SHOWING_TASK_DETAILS_FOR = "Showing task details for: {0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "Browsing for IMatrix data file"
        self.SELECT_DATA_FILE = "Select Data File"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "Browsing for IMatrix model file"
        self.SELECT_MODEL_FILE = "Select Model File"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "Browsing for IMatrix output file"
        self.SELECT_OUTPUT_FILE = "Select Output File"
        self.STARTING_IMATRIX_GENERATION = "Starting IMatrix generation"
        self.BACKEND_PATH_NOT_EXIST = "Backend path does not exist: {0}"
        self.GENERATING_IMATRIX = "Generating IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = (
            "Error starting IMatrix generation: {0}"
        )
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix generation task started"
        self.ERROR_MESSAGE = "Error: {0}"
        self.TASK_ERROR = "Task error: {0}"
        self.APPLICATION_CLOSING = "Application closing"
        self.APPLICATION_CLOSED = "Application closed"
        self.SELECT_QUANTIZATION_TYPE = "Select the quantization type"
        self.ALLOWS_REQUANTIZING = (
            "Allows requantizing tensors that have already been quantized"
        )
        self.LEAVE_OUTPUT_WEIGHT = "Will leave output.weight un(re)quantized"
        self.DISABLE_K_QUANT_MIXTURES = (
            "Disable k-quant mixtures and quantize all tensors to the same type"
        )
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "Use data in file as importance matrix for quant optimisations"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Use importance matrix for these tensors"
        )
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = (
            "Don't use importance matrix for these tensors"
        )
        self.OUTPUT_TENSOR_TYPE = "Output Tensor Type:"
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = (
            "Use this type for the output.weight tensor"
        )
        self.TOKEN_EMBEDDING_TYPE = "Token Embedding Type:"
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = (
            "Use this type for the token embeddings tensor"
        )
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "Will generate quantized model in the same shards as input"
        )
        self.OVERRIDE_MODEL_METADATA = "Override model metadata"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "Input data file for IMatrix generation"
        self.MODEL_TO_BE_QUANTIZED = "Model to be quantized"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "Output path for the generated IMatrix"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "How often to save the IMatrix"
        self.SET_GPU_OFFLOAD_VALUE = "Set GPU offload value (-ngl)"
        self.COMPLETED = "Completed"
        self.REFRESH_MODELS = "Refresh Models"


class _TraditionalChinese(_Localization):
    def __init__(self):
        super().__init__()
        self.WINDOW_TITLE = "AutoGGUF（自動 GGUF 模型量化器）"
        self.RAM_USAGE = "RAM 使用量："
        self.CPU_USAGE = "CPU 使用率："
        self.BACKEND = "Llama.cpp 後端："
        self.REFRESH_BACKENDS = "重新整理後端"
        self.MODELS_PATH = "模型路徑："
        self.OUTPUT_PATH = "輸出路徑："
        self.LOGS_PATH = "日誌路徑："
        self.BROWSE = "瀏覽"
        self.AVAILABLE_MODELS = "可用模型："
        self.QUANTIZATION_TYPE = "量化類型："
        self.ALLOW_REQUANTIZE = "允許重新量化"
        self.LEAVE_OUTPUT_TENSOR = "保留輸出張量"
        self.PURE = "純粹"
        self.IMATRIX = "IMatrix："
        self.INCLUDE_WEIGHTS = "包含權重："
        self.EXCLUDE_WEIGHTS = "排除權重："
        self.USE_OUTPUT_TENSOR_TYPE = "使用輸出張量類型"
        self.USE_TOKEN_EMBEDDING_TYPE = "使用權杖嵌入類型"
        self.KEEP_SPLIT = "保持分割"
        self.KV_OVERRIDES = "KV 覆蓋："
        self.ADD_NEW_OVERRIDE = "新增覆蓋"
        self.QUANTIZE_MODEL = "量化模型"
        self.SAVE_PRESET = "儲存預設"
        self.LOAD_PRESET = "載入預設"
        self.TASKS = "任務："
        self.DOWNLOAD_LLAMACPP = "下載 llama.cpp"
        self.SELECT_RELEASE = "選擇版本："
        self.SELECT_ASSET = "選擇資源："
        self.EXTRACT_CUDA_FILES = "解壓縮 CUDA 檔案"
        self.SELECT_CUDA_BACKEND = "選擇 CUDA 後端："
        self.DOWNLOAD = "下載"
        self.IMATRIX_GENERATION = "IMatrix 產生"
        self.DATA_FILE = "資料檔案："
        self.MODEL = "模型："
        self.OUTPUT = "輸出："
        self.OUTPUT_FREQUENCY = "輸出頻率："
        self.GPU_OFFLOAD = "GPU 卸載："
        self.AUTO = "自動"
        self.GENERATE_IMATRIX = "產生 IMatrix"
        self.ERROR = "錯誤"
        self.WARNING = "警告"
        self.PROPERTIES = "屬性"
        self.CANCEL = "取消"
        self.RESTART = "重新啟動"
        self.DELETE = "刪除"
        self.CONFIRM_DELETION = "您確定要刪除此任務嗎？"
        self.TASK_RUNNING_WARNING = "某些任務仍在執行中。您確定要結束嗎？"
        self.YES = "是"
        self.NO = "否"
        self.DOWNLOAD_COMPLETE = "下載完成"
        self.CUDA_EXTRACTION_FAILED = "CUDA 解壓縮失敗"
        self.PRESET_SAVED = "預設已儲存"
        self.PRESET_LOADED = "預設已載入"
        self.NO_ASSET_SELECTED = "未選擇資源"
        self.DOWNLOAD_FAILED = "下載失敗"
        self.NO_BACKEND_SELECTED = "未選擇後端"
        self.NO_MODEL_SELECTED = "未選擇模型"
        self.REFRESH_RELEASES = "重新整理版本"
        self.NO_SUITABLE_CUDA_BACKENDS = "找不到合適的 CUDA 後端"
        self.LLAMACPP_DOWNLOADED_EXTRACTED = (
            "llama.cpp 二進位檔案已下載並解壓縮至 {0}\nCUDA 檔案已解壓縮至 {1}"
        )
        self.CUDA_FILES_EXTRACTED = "CUDA 檔案已解壓縮至"
        self.NO_SUITABLE_CUDA_BACKEND_EXTRACTION = "找不到合適的 CUDA 後端進行解壓縮"
        self.ERROR_FETCHING_RELEASES = "擷取版本時發生錯誤：{0}"
        self.CONFIRM_DELETION_TITLE = "確認刪除"
        self.LOG_FOR = "{0} 的日誌"
        self.ALL_FILES = "所有檔案 (*)"
        self.GGUF_FILES = "GGUF 檔案 (*.gguf)"
        self.DAT_FILES = "DAT 檔案 (*.dat)"
        self.JSON_FILES = "JSON 檔案 (*.json)"
        self.FAILED_LOAD_PRESET = "載入預設失敗：{0}"
        self.INITIALIZING_AUTOGGUF = "正在初始化 AutoGGUF 應用程式"
        self.AUTOGGUF_INITIALIZATION_COMPLETE = "AutoGGUF 初始化完成"
        self.REFRESHING_BACKENDS = "正在重新整理後端"
        self.NO_BACKENDS_AVAILABLE = "沒有可用的後端"
        self.FOUND_VALID_BACKENDS = "找到 {0} 個有效的後端"
        self.SAVING_PRESET = "正在儲存預設"
        self.PRESET_SAVED_TO = "預設已儲存至 {0}"
        self.LOADING_PRESET = "正在載入預設"
        self.PRESET_LOADED_FROM = "從 {0} 載入了預設"
        self.ADDING_KV_OVERRIDE = "正在新增 KV 覆蓋：{0}"
        self.SAVING_TASK_PRESET = "正在儲存 {0} 的任務預設"
        self.TASK_PRESET_SAVED = "任務預設已儲存"
        self.TASK_PRESET_SAVED_TO = "任務預設已儲存至 {0}"
        self.RESTARTING_TASK = "正在重新啟動任務：{0}"
        self.IN_PROGRESS = "處理中"
        self.DOWNLOAD_FINISHED_EXTRACTED_TO = "下載完成。已解壓縮至：{0}"
        self.LLAMACPP_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp 二進位檔案已下載並解壓縮至 {0}\nCUDA 檔案已解壓縮至 {1}"
        )
        self.NO_SUITABLE_CUDA_BACKEND_FOUND = "找不到合適的 CUDA 後端進行解壓縮"
        self.LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED = (
            "llama.cpp 二進位檔案已下載並解壓縮至 {0}"
        )
        self.REFRESHING_LLAMACPP_RELEASES = "正在重新整理 llama.cpp 版本"
        self.UPDATING_ASSET_LIST = "正在更新資源清單"
        self.UPDATING_CUDA_OPTIONS = "正在更新 CUDA 選項"
        self.STARTING_LLAMACPP_DOWNLOAD = "正在開始下載 llama.cpp"
        self.UPDATING_CUDA_BACKENDS = "正在更新 CUDA 後端"
        self.NO_CUDA_BACKEND_SELECTED = "未選擇要解壓縮的 CUDA 後端"
        self.EXTRACTING_CUDA_FILES = "正在從 {0} 解壓縮 CUDA 檔案至 {1}"
        self.DOWNLOAD_ERROR = "下載錯誤：{0}"
        self.SHOWING_TASK_CONTEXT_MENU = "正在顯示任務操作選單"
        self.SHOWING_PROPERTIES_FOR_TASK = "正在顯示任務的屬性：{0}"
        self.CANCELLING_TASK = "正在取消任務：{0}"
        self.CANCELED = "已取消"
        self.DELETING_TASK = "正在刪除任務：{0}"
        self.LOADING_MODELS = "正在載入模型"
        self.LOADED_MODELS = "已載入 {0} 個模型"
        self.BROWSING_FOR_MODELS_DIRECTORY = "正在瀏覽模型目錄"
        self.SELECT_MODELS_DIRECTORY = "選擇模型目錄"
        self.BROWSING_FOR_OUTPUT_DIRECTORY = "正在瀏覽輸出目錄"
        self.SELECT_OUTPUT_DIRECTORY = "選擇輸出目錄"
        self.BROWSING_FOR_LOGS_DIRECTORY = "正在瀏覽日誌目錄"
        self.SELECT_LOGS_DIRECTORY = "選擇日誌目錄"
        self.BROWSING_FOR_IMATRIX_FILE = "正在瀏覽 IMatrix 檔案"
        self.SELECT_IMATRIX_FILE = "選擇 IMatrix 檔案"
        self.RAM_USAGE_FORMAT = "{0:.1f}%（{1} MB / {2} MB）"
        self.CPU_USAGE_FORMAT = "CPU 使用率：{0:.1f}%"
        self.VALIDATING_QUANTIZATION_INPUTS = "正在驗證量化輸入"
        self.MODELS_PATH_REQUIRED = "需要模型路徑"
        self.OUTPUT_PATH_REQUIRED = "需要輸出路徑"
        self.LOGS_PATH_REQUIRED = "需要日誌路徑"
        self.STARTING_MODEL_QUANTIZATION = "正在開始模型量化"
        self.INPUT_FILE_NOT_EXIST = "輸入檔案 '{0}' 不存在。"
        self.QUANTIZING_MODEL_TO = "正在將 {0} 量化為 {1}"
        self.QUANTIZATION_TASK_STARTED = "已啟動 {0} 的量化任務"
        self.ERROR_STARTING_QUANTIZATION = "啟動量化時發生錯誤：{0}"
        self.UPDATING_MODEL_INFO = "正在更新模型資訊：{0}"
        self.TASK_FINISHED = "任務完成：{0}"
        self.SHOWING_TASK_DETAILS_FOR = "正在顯示任務詳細資訊：{0}"
        self.BROWSING_FOR_IMATRIX_DATA_FILE = "正在瀏覽 IMatrix 資料檔案"
        self.SELECT_DATA_FILE = "選擇資料檔案"
        self.BROWSING_FOR_IMATRIX_MODEL_FILE = "正在瀏覽 IMatrix 模型檔案"
        self.SELECT_MODEL_FILE = "選擇模型檔案"
        self.BROWSING_FOR_IMATRIX_OUTPUT_FILE = "正在瀏覽 IMatrix 輸出檔案"
        self.SELECT_OUTPUT_FILE = "選擇輸出檔案"
        self.STARTING_IMATRIX_GENERATION = "正在開始 IMatrix 產生"
        self.BACKEND_PATH_NOT_EXIST = "後端路徑不存在：{0}"
        self.GENERATING_IMATRIX = "正在產生 IMatrix"
        self.ERROR_STARTING_IMATRIX_GENERATION = "啟動 IMatrix 產生時發生錯誤：{0}"
        self.IMATRIX_GENERATION_TASK_STARTED = "IMatrix 產生任務已啟動"
        self.ERROR_MESSAGE = "錯誤：{0}"
        self.TASK_ERROR = "任務錯誤：{0}"
        self.APPLICATION_CLOSING = "應用程式正在關閉"
        self.APPLICATION_CLOSED = "應用程式已關閉"
        self.SELECT_QUANTIZATION_TYPE = "請選擇量化類型"
        self.ALLOWS_REQUANTIZING = "允許重新量化已量化的張量"
        self.LEAVE_OUTPUT_WEIGHT = "將保留 output.weight 不被（重新）量化"
        self.DISABLE_K_QUANT_MIXTURES = "停用 k-quant 混合並將所有張量量化為相同類型"
        self.USE_DATA_AS_IMPORTANCE_MATRIX = (
            "使用檔案中的資料作為量化最佳化的重要性矩陣"
        )
        self.USE_IMPORTANCE_MATRIX_FOR_TENSORS = "對這些張量使用重要性矩陣"
        self.DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS = "不要對這些張量使用重要性矩陣"
        self.OUTPUT_TENSOR_TYPE = "輸出張量類型："
        self.USE_THIS_TYPE_FOR_OUTPUT_WEIGHT = "對 output.weight 張量使用此類型"
        self.TOKEN_EMBEDDING_TYPE = "權杖嵌入類型："
        self.USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS = "對權杖嵌入張量使用此類型"
        self.WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS = (
            "將在與輸入相同的分片中產生量化模型"
        )
        self.OVERRIDE_MODEL_METADATA = "覆蓋模型中繼資料"
        self.INPUT_DATA_FILE_FOR_IMATRIX = "IMatrix 產生的輸入資料檔案"
        self.MODEL_TO_BE_QUANTIZED = "要量化的模型"
        self.OUTPUT_PATH_FOR_GENERATED_IMATRIX = "產生的 IMatrix 的輸出路徑"
        self.HOW_OFTEN_TO_SAVE_IMATRIX = "儲存 IMatrix 的頻率"
        self.SET_GPU_OFFLOAD_VALUE = "設定 GPU 卸載值（-ngl）"
        self.COMPLETED = "已完成"
        self.REFRESH_MODELS = "重新整理模型"


# fmt: off

# Dictionary to map language codes to classes
_languages = {
    "en-US": _English,               # American English
    "fr-FR": _French,                # Metropolitan French
    "zh-CN": _SimplifiedChinese,     # Simplified Chinese
    "es-ES": _Spanish,               # Spanish (Spain)
    "hi-IN": _Hindi,                 # Hindi (India)
    "ru-RU": _Russian,               # Russian (Russia)
    "uk-UA": _Ukrainian,             # Ukrainian (Ukraine)
    "ja-JP": _Japanese,              # Japanese (Japan)
    "de-DE": _German,                # German (Germany)
    "pt-BR": _Portuguese,            # Portuguese (Brazil)
    "ar-SA": _Arabic,                # Arabic (Saudi Arabia)
    "ko-KR": _Korean,                # Korean (Korea)
    "it-IT": _Italian,               # Italian (Italy)
    "tr-TR": _Turkish,               # Turkish (Turkey)
    "nl-NL": _Dutch,                 # Dutch (Netherlands)
    "fi-FI": _Finnish,               # Finnish (Finland)
    "bn-BD": _Bengali,               # Bengali (Bangladesh)
    "cs-CZ": _Czech,                 # Czech (Czech Republic)
    "pl-PL": _Polish,                # Polish (Poland)
    "ro-RO": _Romanian,              # Romanian (Romania)
    "el-GR": _Greek,                 # Greek (Greece)
    "pt-PT": _Portuguese_PT,         # Portuguese (Portugal)
    "hu-HU": _Hungarian,             # Hungarian (Hungary)
    "en-GB": _BritishEnglish,        # British English
    "fr-CA": _CanadianFrench,        # Canadian French
    "en-IN": _IndianEnglish,         # Indian English
    "en-CA": _CanadianEnglish,       # Canadian English
    "zh-TW": _TraditionalChinese,    # Traditional Chinese (Taiwan)
}

# fmt: on


def set_language(lang_code) -> None:
    # Globals
    global WINDOW_TITLE, RAM_USAGE, CPU_USAGE, BACKEND, REFRESH_BACKENDS, MODELS_PATH, OUTPUT_PATH, LOGS_PATH
    global BROWSE, AVAILABLE_MODELS, QUANTIZATION_TYPE, ALLOW_REQUANTIZE, LEAVE_OUTPUT_TENSOR, PURE, IMATRIX
    global INCLUDE_WEIGHTS, EXCLUDE_WEIGHTS, USE_OUTPUT_TENSOR_TYPE, USE_TOKEN_EMBEDDING_TYPE, KEEP_SPLIT
    global KV_OVERRIDES, ADD_NEW_OVERRIDE, QUANTIZE_MODEL, SAVE_PRESET, LOAD_PRESET, TASKS, DOWNLOAD_LLAMACPP
    global SELECT_RELEASE, SELECT_ASSET, EXTRACT_CUDA_FILES, SELECT_CUDA_BACKEND, DOWNLOAD, IMATRIX_GENERATION
    global DATA_FILE, MODEL, OUTPUT, OUTPUT_FREQUENCY, GPU_OFFLOAD, AUTO, GENERATE_IMATRIX, ERROR, WARNING
    global PROPERTIES, CANCEL, RESTART, DELETE, CONFIRM_DELETION, TASK_RUNNING_WARNING, YES, NO, DOWNLOAD_COMPLETE
    global CUDA_EXTRACTION_FAILED, PRESET_SAVED, PRESET_LOADED, NO_ASSET_SELECTED, DOWNLOAD_FAILED, NO_BACKEND_SELECTED
    global NO_MODEL_SELECTED, REFRESH_RELEASES, NO_SUITABLE_CUDA_BACKENDS, LLAMACPP_DOWNLOADED_EXTRACTED, CUDA_FILES_EXTRACTED
    global NO_SUITABLE_CUDA_BACKEND_EXTRACTION, ERROR_FETCHING_RELEASES, CONFIRM_DELETION_TITLE, LOG_FOR, ALL_FILES
    global GGUF_FILES, DAT_FILES, JSON_FILES, FAILED_LOAD_PRESET, INITIALIZING_AUTOGGUF, AUTOGGUF_INITIALIZATION_COMPLETE
    global REFRESHING_BACKENDS, NO_BACKENDS_AVAILABLE, FOUND_VALID_BACKENDS, SAVING_PRESET, PRESET_SAVED_TO, LOADING_PRESET
    global PRESET_LOADED_FROM, ADDING_KV_OVERRIDE, SAVING_TASK_PRESET, TASK_PRESET_SAVED, TASK_PRESET_SAVED_TO, RESTARTING_TASK
    global IN_PROGRESS, DOWNLOAD_FINISHED_EXTRACTED_TO, LLAMACPP_DOWNLOADED_AND_EXTRACTED, NO_SUITABLE_CUDA_BACKEND_FOUND
    global LLAMACPP_BINARY_DOWNLOADED_AND_EXTRACTED, REFRESHING_LLAMACPP_RELEASES, UPDATING_ASSET_LIST, UPDATING_CUDA_OPTIONS
    global STARTING_LLAMACPP_DOWNLOAD, UPDATING_CUDA_BACKENDS, NO_CUDA_BACKEND_SELECTED, EXTRACTING_CUDA_FILES, DOWNLOAD_ERROR
    global SHOWING_TASK_CONTEXT_MENU, SHOWING_PROPERTIES_FOR_TASK, CANCELLING_TASK, CANCELED, DELETING_TASK, LOADING_MODELS, LOADED_MODELS
    global BROWSING_FOR_MODELS_DIRECTORY, SELECT_MODELS_DIRECTORY, BROWSING_FOR_OUTPUT_DIRECTORY, SELECT_OUTPUT_DIRECTORY
    global BROWSING_FOR_LOGS_DIRECTORY, SELECT_LOGS_DIRECTORY, BROWSING_FOR_IMATRIX_FILE, SELECT_IMATRIX_FILE, RAM_USAGE_FORMAT
    global CPU_USAGE_FORMAT, VALIDATING_QUANTIZATION_INPUTS, MODELS_PATH_REQUIRED, OUTPUT_PATH_REQUIRED, LOGS_PATH_REQUIRED
    global STARTING_MODEL_QUANTIZATION, INPUT_FILE_NOT_EXIST, QUANTIZING_MODEL_TO, QUANTIZATION_TASK_STARTED, ERROR_STARTING_QUANTIZATION
    global UPDATING_MODEL_INFO, TASK_FINISHED, SHOWING_TASK_DETAILS_FOR, BROWSING_FOR_IMATRIX_DATA_FILE, SELECT_DATA_FILE
    global BROWSING_FOR_IMATRIX_MODEL_FILE, SELECT_MODEL_FILE, BROWSING_FOR_IMATRIX_OUTPUT_FILE, SELECT_OUTPUT_FILE
    global STARTING_IMATRIX_GENERATION, BACKEND_PATH_NOT_EXIST, GENERATING_IMATRIX, ERROR_STARTING_IMATRIX_GENERATION
    global IMATRIX_GENERATION_TASK_STARTED, ERROR_MESSAGE, TASK_ERROR, APPLICATION_CLOSING, APPLICATION_CLOSED, SELECT_QUANTIZATION_TYPE
    global ALLOWS_REQUANTIZING, LEAVE_OUTPUT_WEIGHT, DISABLE_K_QUANT_MIXTURES, USE_DATA_AS_IMPORTANCE_MATRIX, USE_IMPORTANCE_MATRIX_FOR_TENSORS
    global DONT_USE_IMPORTANCE_MATRIX_FOR_TENSORS, OUTPUT_TENSOR_TYPE, USE_THIS_TYPE_FOR_OUTPUT_WEIGHT, TOKEN_EMBEDDING_TYPE, USE_THIS_TYPE_FOR_TOKEN_EMBEDDINGS
    global WILL_GENERATE_QUANTIZED_MODEL_IN_SAME_SHARDS, OVERRIDE_MODEL_METADATA, INPUT_DATA_FILE_FOR_IMATRIX, MODEL_TO_BE_QUANTIZED
    global OUTPUT_PATH_FOR_GENERATED_IMATRIX, HOW_OFTEN_TO_SAVE_IMATRIX, SET_GPU_OFFLOAD_VALUE, COMPLETED, REFRESH_MODELS
    global CONTEXT_SIZE, CONTEXT_SIZE_FOR_IMATRIX, THREADS, NUMBER_OF_THREADS_FOR_IMATRIX, EXTRA_ARGUMENTS, EXTRA_ARGUMENTS_LABEL
    global LORA_CONVERSION, LORA_INPUT_PATH, LORA_OUTPUT_PATH, SELECT_LORA_INPUT_DIRECTORY, SELECT_LORA_OUTPUT_FILE
    global CONVERT_LORA, STARTING_LORA_CONVERSION, LORA_INPUT_PATH_REQUIRED, LORA_OUTPUT_PATH_REQUIRED, ERROR_STARTING_LORA_CONVERSION
    global LORA_CONVERSION_TASK_STARTED, BIN_FILES, BROWSING_FOR_LORA_INPUT_DIRECTORY, BROWSING_FOR_LORA_OUTPUT_FILE, CONVERTING_LORA
    global LORA_CONVERSION_FINISHED, LORA_FILE_MOVED, LORA_FILE_NOT_FOUND, ERROR_MOVING_LORA_FILE, EXPORT_LORA
    global MODEL_PATH_REQUIRED, AT_LEAST_ONE_LORA_ADAPTER_REQUIRED, INVALID_LORA_SCALE_VALUE, ERROR_STARTING_LORA_EXPORT, LORA_EXPORT_TASK_STARTED
    global GGML_LORA_ADAPTERS, SELECT_LORA_ADAPTER_FILES, ADD_ADAPTER, DELETE_ADAPTER, LORA_SCALE
    global ENTER_LORA_SCALE_VALUE, NUMBER_OF_THREADS_FOR_LORA_EXPORT, EXPORTING_LORA, BROWSING_FOR_EXPORT_LORA_MODEL_FILE, BROWSING_FOR_EXPORT_LORA_OUTPUT_FILE
    global ADDING_LORA_ADAPTER, DELETING_LORA_ADAPTER, LORA_FILES, SELECT_LORA_ADAPTER_FILE, STARTING_LORA_EXPORT
    global OUTPUT_TYPE, SELECT_OUTPUT_TYPE, GGUF_AND_BIN_FILES, BASE_MODEL, SELECT_BASE_MODEL_FILE
    global BASE_MODEL_PATH_REQUIRED, BROWSING_FOR_BASE_MODEL_FILE, SELECT_BASE_MODEL_FOLDER, BROWSING_FOR_BASE_MODEL_FOLDER
    global LORA_CONVERSION_FROM_TO, GENERATING_IMATRIX_FOR, MODEL_PATH_REQUIRED_FOR_IMATRIX, NO_ASSET_SELECTED_FOR_CUDA_CHECK, QUANTIZATION_COMMAND
    global IMATRIX_GENERATION_COMMAND, LORA_CONVERSION_COMMAND, LORA_EXPORT_COMMAND
    global NO_QUANTIZATION_TYPE_SELECTED, STARTING_HF_TO_GGUF_CONVERSION, MODEL_DIRECTORY_REQUIRED
    global HF_TO_GGUF_CONVERSION_COMMAND, CONVERTING_TO_GGUF, ERROR_STARTING_HF_TO_GGUF_CONVERSION
    global HF_TO_GGUF_CONVERSION_TASK_STARTED, HF_TO_GGUF_CONVERSION, MODEL_DIRECTORY, OUTPUT_FILE
    global VOCAB_ONLY, USE_TEMP_FILE, NO_LAZY_EVALUATION, MODEL_NAME, VERBOSE, SPLIT_MAX_SIZE
    global DRY_RUN, CONVERT_HF_TO_GGUF, SELECT_HF_MODEL_DIRECTORY, BROWSE_FOR_HF_MODEL_DIRECTORY
    global BROWSE_FOR_HF_TO_GGUF_OUTPUT, SHARDED

    loc = _languages.get(lang_code, _English)()
    english_loc = _English()  # Create an instance of English localization for fallback

    for key in dir(english_loc):
        if not key.startswith("_"):
            globals()[key] = getattr(loc, key, getattr(english_loc, key))


def load_language():
    if not os.path.exists(".env"):
        return None

    with open(".env", "r") as f:
        for line in f:
            match = re.match(
                r"AUTOGGUF_LANGUAGE=([a-zA-Z]{2}-[a-zA-Z]{2})", line.strip()
            )
            if match:
                return match.group(1)

    return os.getenv("AUTOGGUF_LANGUAGE", "en-US")


# Get the language from the AUTOGGUF_LANGUAGE environment variable, default to 'en'
language_code = load_language()

# Set default language
set_language(language_code)
