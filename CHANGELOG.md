# Changelog

All notable changes to this project will be documented in this file.

## [1.4.2] - 2024-08-04

### Fixed
- Resolves bug where Base Model text was shown even when GGML type was selected
- Improved alignment

### Changed
- Minor repository changes

## [1.4.1] - 2024-08-04

### Added
- Dynamic KV Overrides (see wiki: AutoGGUF/wiki/Dynamic-KV-Overrides)
- Quantization commands are now printed and logged

## [1.4.0] - 2024-08-04

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

## [1.3.1] - 2024-08-04

### Added
- AUTOGGUF_CHECK_BACKEND environment variable to disable backend check on start

### Changed
- --onefile build with PyInstaller, _internal directory is no longer required

## [1.3.0] - 2024-08-03

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

## [1.2.1] - 2024-08-03

### Added
- Refresh Models button
- Linux build (built on Ubuntu 24.04 LTS)

### Fixed
- iostream llama.cpp issue, quantized_models directory created on launch

## [1.2.0] - 2024-08-03

### Added
- More robust logging (find logs at latest_<timestamp>.log in logs folder)
- Localizations with support for 28 languages (machine translated using Gemini Experimental 0801)

## [1.1.0] - 2024-08-03

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

## [1.0.0] - 2024-08-02

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
