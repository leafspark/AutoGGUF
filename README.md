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
![GitHub release (latest by date)](https://img.shields.io/github/downloads/leafspark/AutoGGUF/latest/total?color=green)
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
- Parallel quantization + imatrix generation
- LoRA conversion and merging
- Preset saving and loading

## Usage

### Cross-platform
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the application:
   ```
   python src/main.py
   ```
   or use the `run.bat` script.

### Windows
Standard builds:
1. Download the latest release
2. Extract all files to a folder
3. Run `AutoGGUF-x64.exe`

Setup builds:
1. Download setup varient of latest release
2. Extract all files to a folder
3. Run the setup program
4. The .GGUF extension will be registered with the program automatically
5. Run the program from the Start Menu or desktop shortcuts

### Verifying Releases

#### Linux/macOS:
```bash
gpg --import AutoGGUF-v1.5.0-prerel.asc
gpg --verify AutoGGUF-v1.5.0-Windows-avx2-prerel.zip.sig AutoGGUF-v1.5.0-Windows-avx2-prerel.zip
sha256sum -c AutoGGUF-v1.5.0-prerel.sha256
```

#### Windows (PowerShell):
```powershell
# Import the public key
gpg --import AutoGGUF-v1.5.0-prerel.asc

# Verify the signature
gpg --verify AutoGGUF-v1.5.0-Windows-avx2-prerel.zip.sig AutoGGUF-v1.5.0-Windows-avx2-prerel.zip

# Check SHA256
$fileHash = (Get-FileHash -Algorithm SHA256 AutoGGUF-v1.5.0-Windows-avx2-prerel.zip).Hash.ToLower()
$storedHash = (Get-Content AutoGGUF-v1.5.0-prerel.sha256 | Select-String AutoGGUF-v1.5.0-Windows-avx2-prerel.zip).Line.Split()[0]
if ($fileHash -eq $storedHash) { "SHA256 Match" } else { "SHA256 Mismatch" }
```

Release keys are identical to ones used for commiting.

## Building

### Cross-platform
```bash
pip install -U pyinstaller
./build.sh RELEASE | DEV
cd build/<type>/dist/
./AutoGGUF
```

### Windows
```bash
build RELEASE | DEV
```
Find the executable in `build/<type>/dist/AutoGGUF.exe`.

You can also use the slower build but faster executable script (Nuitka):
```bash
build_optimized RELEASE | DEV
```

## Dependencies

Find them in `requirements.txt`.

## Localizations

View the list of supported languages at [AutoGGUF/wiki/Installation#configuration](https://github.com/leafspark/AutoGGUF/wiki/Installation#configuration) (LLM translated, except for English).

To use a specific language, set the `AUTOGGUF_LANGUAGE` environment variable to one of the listed language codes (note: some languages may not be fully supported yet, those will fall back to English).

## Known Issues

- None!

## Planned Features

- Time estimation for quantization
- Actual progress bar tracking 
- Perplexity testing
- Web API and management (partially implemented in v1.6.2)
- ~~Themes~~ (added in v1.7.1)
- ~~Sleek UI menubar~~ (added in v1.7.1)

## Troubleshooting

- SSL module cannot be found error: Install OpenSSL or run from source using `python src/main.py` with the `run.bat` script (`pip install requests`)

## Contributing

Fork the repo, make your changes, and ensure you have the latest commits when merging. Include a changelog of new features in your pull request description. Read `CONTRIBUTING.md` for more information.

## User Interface

![AutoGGUF-v1 7 1-showcase-blue](https://github.com/user-attachments/assets/4240437f-77d4-459b-924f-c80e5f672c4f)

## Stargazers

[![Star History Chart](https://api.star-history.com/svg?repos=leafspark/AutoGGUF&type=Date)](https://star-history.com/#leafspark/AutoGGUF&Date)
