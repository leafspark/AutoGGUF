# AutoGGUF - automated GGUF model quantizer

This application provides a graphical user interface for quantizing GGUF models
using the llama.cpp library. It allows users to download different versions of
llama.cpp, manage multiple backends, and perform quantization tasks with various
options.

**Main features**:
1. Download and manage llama.cpp backends
2. Select and quantize GGUF models
3. Configure quantization parameters
4. Monitor system resources during quantization

**Usage**:

Cross platform:
  1. Install dependencies, either using the `requirements.txt` file or `pip install PyQt6 requests psutil`.
  2. Run the `run.bat` script to start the application, or run the command `python src/main.py`.

Windows:
  1. Download latest release, extract all to folder and run `AutoGGUF.exe`
  2. Enjoy!

**Building**:
```
cd src
pip install -U pyinstaller
pyinstaller main.py
cd dist/main
main
```

**Dependencies**:
- PyQt6
- requests
- psutil

**Localizations:**

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
In order to use them, please set the `AUTOGGUF_LANGUAGE` enviroment variable to one of the listed language codes.

**Issues:**
- Actual progress bar tracking
- Download safetensors from HF and convert to unquanted GGUF
- ~~Specify multiple KV overrides~~ (added in v1.1.0)
- ~~Better error handling~~ (added in v1.1.0)
- ~~Cannot select output/token embd type~~ (fixed in v1.1.0)

**Troubleshooting:**
- llama.cpp quantizations errors out with an iostream error: create the `quantized_models` directory (or set a directory)

**User interface:**
![image](https://github.com/user-attachments/assets/b1b58cba-4314-479d-a1d8-21ca0b5a8935)
