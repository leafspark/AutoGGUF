"""
Convert PEFT LoRA adapters to GGML format.

This script converts Hugging Face PEFT LoRA adapter files to the GGML format
used by llama.cpp and related projects. It reads the adapter configuration
from 'adapter_config.json' and the model weights from 'adapter_model.bin'
or 'adapter_model.safetensors', then writes the converted model to
'ggml-adapter-model.bin' in the same directory.

Usage:
    python lora_to_gguf.py <path> [arch]

Arguments:
    path: Directory containing the PEFT LoRA files
    arch: Model architecture (default: llama)

The script supports various model architectures and handles both PyTorch
and safetensors formats for input weights. It performs necessary tensor
transformations and writes the output in the GGML binary format.

Requirements:
    - Python 3.6+
    - numpy
    - torch
    - safetensors (optional, for safetensors input)

The script also requires the GGUF Python module, which should be in the
'gguf-py/gguf' subdirectory relative to this script's location.

Note: This script is designed for use with llama.cpp and related projects.
Ensure compatibility with your target application when using the output.
"""
