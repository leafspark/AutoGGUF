![AutoGGUF-banner](https://github.com/user-attachments/assets/0f74b104-0541-46a7-9ac8-4a3fcb74b896)

# AutoGGUF - automated GGUF model quantizer

AutoGGUF provides a graphical user interface for quantizing GGUF models
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

View the list of languages supported at [AutoGGUF/wiki/Installation#configuration](https://github.com/leafspark/AutoGGUF/wiki/Installation#configuration) (LLM translated, except for English)

In order to use them, please set the `AUTOGGUF_LANGUAGE` environment variable to one of the listed language codes.

## Issues:
- Saving preset while quantizing causes UI thread crash (planned fix: remove this feature)
- Cannot delete task while processing, you must cancel it first or the program crashes (planned fix: don't allow deletion before cancelling, or cancel automatically)
- Base Model text still shows when GGML is selected as LoRA type (fix: include text in show/hide Qt layout)
- ~~Cannot disable llama.cpp update check on startup~~ (fixed in v1.3.1)
- ~~`_internal` directory required, will see if I can package this into a single exe on the next release~~ (fixed in v1.3.1)
- ~~Custom command line parameters~~ (added in v1.3.0)
- ~~More iMatrix generation parameters~~ (added in v1.3.0)
- ~~Specify multiple KV overrides~~ (added in v1.1.0)
- ~~Better error handling~~ (added in v1.1.0)
- ~~Cannot select output/token embd type~~ (fixed in v1.1.0)
- ~~Importing presets with KV overrides causes UI thread crash~~ (fixed in v1.3.0)

## Planned features:
- Actual progress bar tracking
- Download safetensors from HF and convert to unquanted GGUF
- Perplexity testing
- Managing shards (coming in the next release)
- Time estimated for quantization
- Dynamic values for KV cache, e.g. autogguf.quantized.time=str:{system.time.milliseconds} (coming in the next release)
- Ability to select and start multiple quants at once (saved in presets) (coming in the next release)

## Troubleshooting:
- ~~llama.cpp quantizations errors out with an iostream error: create the `quantized_models` directory (or set a directory)~~ (fixed in v1.2.1, automatically created on launch)
- SSL module cannot be found error: Install OpenSSL or run from source `python src/main.py` using the `run.bat` script (`pip install requests`)

## Contributing:
Simply fork the repo and make your changes; when merging make sure to have the latest commits. Description should contain a changelog of what's new.

## User interface:
![image](https://github.com/user-attachments/assets/2660c841-07ba-4c3f-ae3a-e63c7068bdc1)

## Stargazers:
[![Star History Chart](https://api.star-history.com/svg?repos=leafspark/AutoGGUF&type=Date)](https://star-history.com/#leafspark/AutoGGUF&Date)
