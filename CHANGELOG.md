# Changelog

## [v2.0.1] - 2025-05-24

### Added
- Human readable mappings from KV pairs into model properties
- certifi library for backend download and update checking
- Automated checksums in CI process

### Changed
- Updated llama.cpp backend
- Improved backend UI, logging, and task handling
- Enhanced display of model properties and cleaner formatting of KV pairs
- Updated tensor data formatting and removed redundant KV pairs property
- Updated CUDA backend check for latest llama.cpp release format
- Global urllib usage implementation
- Updated README with more information about patches and updates
- Edited quick start instructions
- Small file formatting improvements

### Fixed
- Type hints corrections
- Build errors in CI
- `@upload-artifact` updated to v4

## [v2.0.0] - 2025-01-27

### Added
- Clipboard support for save/load preset functionality with shift-click option
- Support for shift-clicking to get quantization command
- AUTOGGUF_BACKEND_REPO environment variable for custom GitHub repository fetching
- New HF to GGUF conversion types: `tq1_0` and `tq2_0`

### Changed
- Updated multiple dependencies:
  - PySide6, PyTorch, Transformers, FastAPI, uvicorn, and other core libraries to their latest compatible versions
- Adjusted monitoring intervals from 0.2s to 0.5s
- Updated copyright year to 2025
- Bundled llama.cpp licensing text in About menu
- Removed x86 build matrix from CI
- Removed Import Model confirmation dialog

### Fixed
- Resolved PySide6 segfault issue
- Fixed error when deleting models from list
- Corrected incorrect menu bar name for Load Preset

## [v1.9.1] - 2024-10-13

### Added
- Support for specifying log directory name using AUTOGGUF_LOG_DIR_NAME environment variable
- Work in progress GGUF merge window
- Support for repository types in HF Transfer utility
- New `dequantize_gguf.py` script
- Support for MiniCPM3, RWKVv6, OLMoE, IBM Granite, and Jamba in llama.cpp convert scripts (conversion only)
- Add Nuitka build script for Linux

### Changed
- Updated Finnish and Russian localizations using Claude 3 Opus
- Improved layout of HF Upload window
- Updated gguf library from upstream
- Refactored code to use localizations for menubar
- Renamed imports_and_globals.py to globals.py
- Moved general functions verify_gguf and process_args to globals.py
- Created Plugins class for extensibility
- Updated dependencies:
  - huggingface-hub
  - fastapi (~=0.115.0)
  - setuptools (~=75.1.0)
  - pyside6 (~=6.7.3)
  - uvicorn (~=0.31.0)

### Fixed
- Corrected localization strings and file select types for GGUF merging
- Fix minor errors in build scripts

## [v1.9.0] - 2024-09-15

### Added
- Implemented Hugging Face (HF) upload functionality with GUI definitions
- Added RAM and CPU usage graphs to UI
- Input validation using wraps added to UI
- Right-click context menu added to the models list in UI
- Support for iMatrix generation tracking
- GGUF splitting feature added
- Japanese and German localizations updated

### Changed
- Refactored to move functions out of `AutoGGUF` to reduce bloat
- Localized GGUF split strings
- Optimized GGUF imports and renamed related modules
- Removed old `HFTransfer` class
- Adjusted logging strings and updated French and Dutch localizations
- Improved startup time by optimizing default configuration, disabling network fetches for backends/updates
- Removed `requests` and `python-dotenv` to reduce size
- Updated `fastapi` requirement from `~=0.112.2` to `~=0.114.2`
- Updated `torch` requirement from `~=2.4.0` to `~=2.4.1`
- Updated `setuptools` requirement from `~=74.0.0` to `~=74.1.2`
- Updated `safetensors` requirement from `~=0.4.4` to `~=0.4.5`
- Updated `huggingface-hub` requirement from `~=0.24.6` to `~=0.24.7`

### Fixed
- Adjusted indeterminate progress bar behavior
- Removed comments in `requirements.txt` and updated its formatting

## [v1.8.1] - 2024-09-04

### Added
- AutoFP8 quantization classes and window (currently WIP)
- Minimize/maximize buttons to title bar
- API key authentication support for the local server
- HuggingFace upload/download class
- OpenAPI docs for endpoints
- Added new showcase image

### Changed
- Replaced Flask with FastAPI and Uvicorn for improved performance
- Moved functions out of AutoGGUF.py into utils.py and TaskListItem.py
- Updated llama.cpp convert scripts
- Improved LoRA conversion process:
  - Allow specifying output path in arguments
  - Removed shutil.move operation
  - Increased max number of LoRA layers
- Changed default port to 7001
- Now binding to localhost (127.0.0.1) instead of 0.0.0.0
- Upadted Spanish localizations
- Updated setuptools requirement from ~=68.2.0 to ~=74.0.0
- Updated .env.example with new configuration parameters

### Fixed
- Web page not found error
- Use of proper status in TaskListItem
- Passing of quant_threads and Logger to TaskListItem
- Improved window moving smoothness
- Prevention of moving window below taskbar
- Optimized imports in various files
- Remove aliased quant types

## [v1.8.0] - 2024-08-26

### Added
- .env.example file added
- Sha256 generation support added to build.yml
- Allow importing models from any directory on the system
- Added manual model import functionality
- Verification for manual imports and support for concatenated files
- Implemented plugins feature using importlib
- Configuration options for AUTOGGUF_MODEL_DIR_NAME, AUTOGGUF_OUTPUT_DIR_NAME, and AUTOGGUF_RESIZE_FACTOR added

### Changed
- Moved get helper functions to utils.py
- Added type hints
- Reformat TaskListItem.py for better readability
- Separate macOS and Linux runs in CI/CD
- Updated .gitignore for better file management
- Updated numpy requirement from <2.0.0 to <3.0.0

### Fixed
- Fixed sha256 file format and avoided overwriting
- Updated regex for progress tracking
- Arabic and French localizations fixed
- Only count valid backends instead of total backend combos
- Import missing modules

## [v1.7.2] - 2024-08-19

### Added
- Update checking support (controlled by AUTOGGUF_CHECK_UPDATE environment variable)
- Live update support for GPU monitor graphs
- Smoother usage bar changes in monitor
- Unicode X button in KV Overrides box
- PyPI setup script
- Inno Setup build file
- Missing requirements and dotenv file loading

### Changed
- Moved functions out of AutoGGUF.py
- Relocated CustomTitleBar to separate file
- Updated torch requirement from ~=2.2.0 to ~=2.4.0
- Updated showcase image
- Version bumped to v1.7.2 in Localizations.py

### Fixed
- setup.py issues

## [v1.7.1] - 2024-08-16

### Added
- Modern UI with seamless title bar
- Window resizing shortcuts (Ctrl+, Ctrl-, Ctrl+0)
- Theming support
- CPU usage bar
- Save Preset and Load Preset options in File menu
- Support for EXAONE model type
- Window size configuration through environment variables

### Changed
- Refactored window to be scrollable
- Moved save/load preset logic to presets.py
- Updated docstrings for AutoGGUF.py, lora_conversion.py, and Logger.py
- Adapted gguf library to project standards

### Fixed
- Updated version to v1.7.0
- Fixed IDE-detected code typos and errors

## [v1.7.0] - 2024-08-16

### Added
- Menu bar with Close and About options
- Program version in localizations.py
- Support for 32-bit builds
- Added dependency audit
- Implemented radon, dependabot, and pre-commit workflows

### Changed
- Updated torch requirement from `~=1.13.1` to `~=2.4.0`
- Updated psutil requirement from `~=5.9.8` to `~=6.0.0`
- Refactored functions out of AutoGGUF.py and moved to ui_update.py
- Changed filenames to follow PEP 8 conventions
- Disabled .md and .txt CodeQL analysis

### Fixed
- Optimized imports in AutoGGUF.py
- Updated README with new version and styled screenshot
- Fixed image blur in documentation

## [v1.6.2] - 2024-08-15

### Added
- Server functionality with new endpoints:
  - `/v1/backends`: Lists all backends and their paths
  - `/v1/health`: Heartbeat endpoint
  - `/v1/tasks`: Provides current task info (name, status, progress, log file)
  - `/v1/models`: Retrieves model details (name, type, path, shard status)
- Environment variable support for server configuration:
  - `AUTOGGUF_SERVER`: Enable/disable server (true/false)
  - `AUTOGGUF_SERVER_PORT`: Set server port (integer)

### Changed
- Updated AutoGGUF docstrings
- Refactored build scripts

### Fixed
- Set GGML types to lowercase in command builder

## [v1.6.1] - 2024-08-12

### Added
- Optimized build scripts
- Nuitka for building

### Changed
- Updated .gitignore

### Fixed
- Bug where deletion while a task is running crashes the program

### Notes
- Fast build: Higher unzipped size (97MB), smaller download (38MB)
- Standard build: Created with PyInstaller, medium download and unzipped size (50MB), potentially slower

## [v1.6.0] - 2024-08-08

### Changed
- Resolve licensing issues by using PySide6

### Added
- Add GPU monitoring support for NVIDIA GPUs

## [v1.5.1] - 2024-08-08

### Changed
- Refactor localizations to use them in HF conversion area
- Rename FAILED_LOAD_PRESET to FAILED_TO_LOAD_PRESET localization key

### Removed
- Remove Save Preset context menu action

### Added
- Support loading *.gguf file types

## [v1.5.0] - 2024-08-06

### Changed
- Refactor localizations to use them in HF conversion area
- Organize localizations

### Added
- Add sha256 and PGP signatures (same as commit ones)
- Add HuggingFace to GGUF conversion support

### Fixed
- Fix scaling on low resolution screens, interface now scrolls

## [v1.4.3] - 2024-08-05

### Changed
- Updated src file in release to be Black formatted
- Modifying the quantize_model function to process all selected types
- Updating preset saving and loading to handle multiple quantization types
- Use ERROR and IN_PROGRESS constants from localizations in QuantizationThread
- Minor repository changes

### Added
- Added model sharding management support
- Allow multiple quantization types to be selected and started simultaneously

## [v1.4.2] - 2024-08-04

### Fixed
- Resolves bug where Base Model text was shown even when GGML type was selected
- Improved alignment

### Changed
- Minor repository changes

## [v1.4.1] - 2024-08-04

### Added
- Dynamic KV Overrides (see wiki: AutoGGUF/wiki/Dynamic-KV-Overrides)
- Quantization commands are now printed and logged

## [v1.4.0] - 2024-08-04

### Added
- LoRA Conversion:
  - New section for converting HuggingFace PEFT LoRA adapters to GGML/GGUF
  - Output type selection (GGML or GGUF)
  - Base model selection for GGUF output
  - LoRA adapter list with individual scaling factors
  - Export LoRA section for merging adapters into base model
- UI Improvements:
  - Updated task names in task list
  - IMatrix generation check
  - Larger window size
  - Added exe favicon
- Localization:
  - French and Simplified Chinese support for LoRA and "Refresh Models" strings
- Code and Build:
  - Code organization improvements
  - Added build script
  - .gitignore file
- Misc:
  - Currently includes src folder with conversion tools
  - No console window popup

## [v1.3.1] - 2024-08-04

### Added
- AUTOGGUF_CHECK_BACKEND environment variable to disable backend check on start

### Changed
- --onefile build with PyInstaller, _internal directory is no longer required

## [v1.3.0] - 2024-08-03

### Added
- Support for new llama-imatrix parameters:
  - Context size (--ctx-size) input
  - Threads (--threads) control
- New parameters to IMatrix section layout
- Slider-spinbox combination for thread count selection
- QSpinBox for output frequency input (1-100 range with percentage suffix)

### Changed
- Converted context size input to a QSpinBox
- Updated generate_imatrix() method to use new UI element values
- Improved error handling in preset loading
- Enhanced localization support for new UI elements

### Fixed
- Error when loading presets containing KV overrides

### Removed
- Duplicated functions

## [v1.2.1] - 2024-08-03

### Added
- Refresh Models button
- Linux build (built on Ubuntu 24.04 LTS)

### Fixed
- iostream llama.cpp issue, quantized_models directory created on launch

## [v1.2.0] - 2024-08-03

### Added
- More robust logging (find logs at latest_<timestamp>.log in logs folder)
- Localizations with support for 28 languages (machine translated using Gemini Experimental 0801)

## [v1.1.0] - 2024-08-03

### Added
- Dynamic KV override functionality
- Improved CUDA checking ability and extraction to the backend folder
- Scrollable area for KV overrides with add/delete capabilities

### Changed
- Enhanced visibility and usability of Output Tensor Type and Token Embedding Type options
- Refactored code for better modularity and reduced circular dependencies

### Fixed
- Behavior of Output Tensor Type and Token Embedding Type dropdown menus
- Various minor UI inconsistencies

## [1.0.1] - 2024-08-02

### Added
- Windows binary (created using PyInstaller)

### Fixed
- Issue where quantization errored with "AutoGGUF does not have x attribute"

## [v1.0.0] - 2024-08-02

### Added
- Initial release
- GUI interface for automated GGUF model quantization
- System resource monitoring (RAM and CPU usage)
- Llama.cpp backend selection and management
- Automatic download of llama.cpp releases from GitHub
- Model selection from local directory
- Comprehensive quantization options
- Task list for managing multiple quantization jobs
- Real-time log viewing for quantization tasks
- IMatrix generation feature with customizable settings
- GPU offload settings for IMatrix generation
- Context menu for task management
- Detailed model information dialog
- Error handling and user notifications
- Confirmation dialogs for task deletion and application exit
