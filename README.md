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

**To be implemented:**
- Actual progress bar tracking
- Download safetensors from HF and convert to unquanted GGUF
- Specify multiple KV overrides
- Better error handling
- Cannot select output/token embd type

**Troubleshooting:**
- llama.cpp quantizations errors out with an iostream error: create the `quantized_models` directory (or set a directory)

**User interface:**
![image](https://github.com/user-attachments/assets/b1b58cba-4314-479d-a1d8-21ca0b5a8935)
