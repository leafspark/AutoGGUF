# AutoGGUF - automated GGUF model quantizer

This application provides a graphical user interface for quantizing GGUF models
using the llama.cpp library. It allows users to download different versions of
llama.cpp, manage multiple backends, and perform quantization tasks with various
options.

## Features:
1. Download and manage llama.cpp backends
2. Select and quantize GGUF models
3. Configure quantization parameters
4. Monitor system resources during quantization

## Usage:

**Cross platform**:
  1. Install dependencies, either using the `requirements.txt` file or `pip install PyQt6 requests psutil`.
  2. Run the `run.bat` script to start the application, or run the command `python src/main.py`.

**Windows**:
  1. Download latest release, extract all to folder and run `AutoGGUF.exe`
  2. Enjoy!

## Building:

**Cross platform**:
```bash
cd src
pip install -U pyinstaller
pyinstaller main.py --onefile
cd dist/main
./main
```
**Windows**:
```bash
build RELEASE/DEV
```
Find exe in `build/<type>/dist/AutoGGUF.exe`.

## Dependencies:
- PyQt6
- requests
- psutil
- shutil
- OpenSSL

## Localizations:

The following languages are currently supported (machine translated, except for English):
```python
{
    'en-US': _English,              # American English
    'fr-FR': _French,               # Metropolitan French
    'zh-CN': _SimplifiedChinese,    # Simplified Chinese
    'es-ES': _Spanish,              # Spanish (Spain)
    'hi-IN': _Hindi,                # Hindi (India)
    'ru-RU': _Russian,              # Russian (Russia)
    'uk-UA': _Ukrainian,            # Ukrainian (Ukraine)
    'ja-JP': _Japanese,             # Japanese (Japan)
    'de-DE': _German,               # German (Germany)
    'pt-BR': _Portuguese,           # Portuguese (Brazil)
    'ar-SA': _Arabic,               # Arabic (Saudi Arabia)
    'ko-KR': _Korean,               # Korean (Korea)    
    'it-IT': _Italian,              # Italian (Italy)
    'tr-TR': _Turkish,              # Turkish (Turkey)
    'nl-NL': _Dutch,                # Dutch (Netherlands)
    'fi-FI': _Finnish,              # Finnish (Finland)
    'bn-BD': _Bengali,              # Bengali (Bangladesh) 
    'cs-CZ': _Czech,                # Czech (Czech Republic)
    'pl-PL': _Polish,               # Polish (Poland)
    'ro-RO': _Romanian,             # Romanian (Romania)
    'el-GR': _Greek,                # Greek (Greece)
    'pt-PT': _Portuguese_PT,        # Portuguese (Portugal)
    'hu-HU': _Hungarian,            # Hungarian (Hungary)
    'en-GB': _BritishEnglish,       # British English
    'fr-CA': _CanadianFrench,       # Canadian French
    'en-IN': _IndianEnglish,        # Indian English
    'en-CA': _CanadianEnglish,      # Canadian English
    'zh-TW': _TraditionalChinese,   # Traditional Chinese (Taiwan)
}
```
In order to use them, please set the `AUTOGGUF_LANGUAGE` environment variable to one of the listed language codes.

## Issues:
- Actual progress bar tracking
- Download safetensors from HF and convert to unquanted GGUF
- Perplexity testing
- ~~Cannot disable llama.cpp update check on startup~~ (fixed in v1.3.1)
- ~~`_internal` directory required, will see if I can package this into a single exe on the next release~~ (fixed in v1.3.1)
- ~~Custom command line parameters~~ (added in v1.3.0)
- ~~More iMatrix generation parameters~~ (added in v1.3.0)
- ~~Specify multiple KV overrides~~ (added in v1.1.0)
- ~~Better error handling~~ (added in v1.1.0)
- ~~Cannot select output/token embd type~~ (fixed in v1.1.0)
- ~~Importing presets with KV overrides causes UI thread crash~~ (fixed in v1.3.0)

## Prerelease issues:
- Base Model label persists even when GGML type is selected

## Troubleshooting:
- ~~llama.cpp quantizations errors out with an iostream error: create the `quantized_models` directory (or set a directory)~~ (fixed in v1.2.1, automatically created on launch)
- SSL module cannot be found error: Install OpenSSL or run from source `python src/main.py` using the `run.bat` script (`pip install requests`)

## User interface:
![image](https://github.com/user-attachments/assets/906bf9cb-38ed-4945-a32e-179acfdcc529)

## Stargazers:
[![Star History Chart](https://api.star-history.com/svg?repos=leafspark/AutoGGUF&type=Date)](https://star-history.com/#leafspark/AutoGGUF&Date)
