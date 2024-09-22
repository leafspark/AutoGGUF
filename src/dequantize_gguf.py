import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

import gguf


def dequantize_tensor(tensor):
    if tensor.tensor_type in [
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    ]:
        return np.array(tensor.data)
    else:
        return tensor.data.astype(np.float32)


def gguf_to_safetensors(gguf_path, safetensors_path, metadata_path=None):
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception as e:
        print(f"Error reading GGUF file: {e}", file=sys.stderr)
        sys.exit(1)

    tensors = {}
    metadata = {}

    for tensor in reader.tensors:
        try:
            dequantized_data = dequantize_tensor(tensor)
            tensors[tensor.name] = torch.from_numpy(
                dequantized_data.reshape(tuple(reversed(tensor.shape)))
            )
        except Exception as e:
            print(f"Error processing tensor {tensor.name}: {e}", file=sys.stderr)
            continue

    for field_name, field in reader.fields.items():
        if field.data:
            metadata[field_name] = field.parts[field.data[0]].tolist()

    try:
        save_file(tensors, safetensors_path)
    except Exception as e:
        print(f"Error saving SafeTensors file: {e}", file=sys.stderr)
        sys.exit(1)

    decoded_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            decoded_value = ""
            for item in value:
                if 48 <= item <= 57:
                    decoded_value += str(item - 48)
                elif 32 <= item <= 126:
                    decoded_value += chr(item)
                else:
                    decoded_value += str(item)
            decoded_metadata[key] = decoded_value
        else:
            decoded_metadata[key] = value

    if metadata_path:
        try:
            with open(metadata_path, "w") as f:
                json.dump(decoded_metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata file: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Convert GGUF to SafeTensors format")
    parser.add_argument("gguf_path", type=str, help="Path to the input GGUF file")
    parser.add_argument(
        "safetensors_path", type=str, help="Path to save the SafeTensors file"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Optional path to save metadata as a JSON file",
    )

    args = parser.parse_args()

    gguf_path = Path(args.gguf_path)
    safetensors_path = Path(args.safetensors_path)
    metadata_path = Path(args.metadata_path) if args.metadata_path else None

    if not gguf_path.exists():
        print(f"Error: GGUF file '{gguf_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {gguf_path} to {safetensors_path}")
    gguf_to_safetensors(gguf_path, safetensors_path, metadata_path)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
