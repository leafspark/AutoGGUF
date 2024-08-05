![AutoGGUF-banner](https://github.com/user-attachments/assets/0f74b104-0541-46a7-9ac8-4a3fcb74b896)

# AutoGGUF - automated GGUF model quantizer

<!-- Project Status -->
[![GitHub release](https://img.shields.io/github/release/leafspark/AutoGGUF.svg)](https://github.com/leafspark/AutoGGUF/releases)
[![GitHub last commit](https://img.shields.io/github/last-commit/leafspark/AutoGGUF.svg)](https://github.com/leafspark/AutoGGUF/commits)
[![CI/CD Status](https://img.shields.io/badge/CI%2FCD-passing-brightgreen)]()

<!-- Project Info -->
[![Powered by llama.cpp](https://img.shields.io/badge/Powered%20by-llama.cpp-green.svg)](https://github.com/ggerganov/llama.cpp)
![GitHub top language](https://img.shields.io/github/languages/top/leafspark/AutoGGUF.svg)
[![Platform Compatibility](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)]()
[![GitHub license](https://img.shields.io/github/license/leafspark/AutoGGUF.svg)](https://github.com/leafspark/AutoGGUF/blob/main/LICENSE)

<!-- Repository Stats -->
![GitHub stars](https://img.shields.io/github/stars/leafspark/AutoGGUF.svg)
![GitHub forks](https://img.shields.io/github/forks/leafspark/AutoGGUF.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/leafspark/AutoGGUF.svg)

<!-- Contribution -->
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Issues](https://img.shields.io/github/issues/leafspark/AutoGGUF)](https://github.com/leafspark/AutoGGUF/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/leafspark/AutoGGUF/pulls)

AutoGGUF provides a graphical user interface for quantizing GGUF models using the llama.cpp library. It allows users to download different versions of llama.cpp, manage multiple backends, and perform quantization tasks with various options.

## Features

- Download and manage llama.cpp backends
- Select and quantize GGUF models
- Configure quantization parameters
- Monitor system resources during quantization

## Usage

### Cross-platform
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or
   ```
   pip install PyQt6 requests psutil shutil
   ```
2. Run the application:
   ```
   python src/main.py
   ```
   or use the `run.bat` script.

### Windows
1. Download the latest release
2. Extract all files to a folder
3. Run `AutoGGUF.exe`

## Building

### Cross-platform
```bash
cd src
pip install -U pyinstaller
pyinstaller main.py --onefile
cd dist/main
./main
```

### Windows
```bash
build RELEASE | DEV
```
Find the executable in `build/<type>/dist/AutoGGUF.exe`.

## Dependencies

- PyQt6
- psutil
- shutil
- numpy
- torch
- safetensors
- gguf (bundled)

## Localizations

View the list of supported languages at [AutoGGUF/wiki/Installation#configuration](https://github.com/leafspark/AutoGGUF/wiki/Installation#configuration) (LLM translated, except for English).

To use a specific language, set the `AUTOGGUF_LANGUAGE` environment variable to one of the listed language codes.

## Known Issues

- Saving preset while quantizing causes UI thread crash (planned fix: remove this feature)
- Cannot delete task while processing (planned fix: disallow deletion before cancelling or cancel automatically)
- ~~Base Model text still shows when GGML is selected as LoRA type (fix: include text in show/hide Qt layout)~~ (fixed in v1.4.2)

## Planned Features

- Actual progress bar tracking
- Download safetensors from HF and convert to unquantized GGUF
- Perplexity testing
- Managing shards (coming in the next release)
- Time estimation for quantization
- Dynamic values for KV cache (coming in the next release)
- Ability to select and start multiple quants at once (saved in presets, coming in the next release)

## Troubleshooting

- SSL module cannot be found error: Install OpenSSL or run from source using `python src/main.py` with the `run.bat` script (`pip install requests`)

## Contributing

Fork the repo, make your changes, and ensure you have the latest commits when merging. Include a changelog of new features in your pull request description. Read `CONTRIBUTING.md` for more information.

## User Interface

![image](https://github.com/user-attachments/assets/2660c841-07ba-4c3f-ae3a-e63c7068bdc1)

## Stargazers

[![Star History Chart](https://api.star-history.com/svg?repos=leafspark/AutoGGUF&type=Date)](https://star-history.com/#leafspark/AutoGGUF&Date)
