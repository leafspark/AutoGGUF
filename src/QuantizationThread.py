import os
import re
import signal
import subprocess

from PySide6.QtCore import Signal, QThread

from globals import open_file_safe
from Localizations import IN_PROGRESS, COMPLETED


class QuantizationThread(QThread):
    # Define custom signals for communication with the main thread
    output_signal = Signal(str)
    status_signal = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)
    model_info_signal = Signal(dict)

    def __init__(self, command, cwd, log_file) -> None:
        super().__init__()
        self.command = command
        self.cwd = cwd
        self.log_file = log_file
        self.process = None
        self.model_info = {}

    def run(self) -> None:
        try:
            # Start the subprocess
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
            )
            # Open log file and process output
            with open_file_safe(self.log_file, "w") as log:
                for line in self.process.stdout:
                    line = line.strip()
                    self.output_signal.emit(line)
                    log.write(line + "\n")
                    log.flush()
                    self.status_signal.emit(IN_PROGRESS)
                    self.parse_model_info(line)

            # Wait for process to complete
            self.process.wait()
            if self.process.returncode == 0:
                self.status_signal.emit(COMPLETED)
                self.model_info_signal.emit(self.model_info)
            else:
                self.error_signal.emit(
                    f"Process exited with code {self.process.returncode}"
                )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

    def parse_model_info(self, line) -> None:
        # Mapping of technical keys to human-readable names
        key_mappings = {
            "general.architecture": "Architecture",
            "general.name": "Model Name",
            "general.file_type": "File Type",
            "general.quantization_version": "Quantization Version",
            "llama.block_count": "Layers",
            "llama.context_length": "Context Length",
            "llama.embedding_length": "Embedding Size",
            "llama.feed_forward_length": "Feed Forward Length",
            "llama.attention.head_count": "Attention Heads",
            "llama.attention.head_count_kv": "Key-Value Heads",
            "llama.attention.layer_norm_rms_epsilon": "RMS Norm Epsilon",
            "llama.rope.freq_base": "RoPE Frequency Base",
            "llama.rope.dimension_count": "RoPE Dimensions",
            "llama.vocab_size": "Vocabulary Size",
            "tokenizer.ggml.model": "Tokenizer Model",
            "tokenizer.ggml.pre": "Tokenizer Preprocessing",
            "tokenizer.ggml.tokens": "Tokens",
            "tokenizer.ggml.token_type": "Token Types",
            "tokenizer.ggml.merges": "BPE Merges",
            "tokenizer.ggml.bos_token_id": "Begin of Sequence Token ID",
            "tokenizer.ggml.eos_token_id": "End of Sequence Token ID",
            "tokenizer.chat_template": "Chat Template",
            "tokenizer.ggml.padding_token_id": "Padding Token ID",
            "tokenizer.ggml.unk_token_id": "Unknown Token ID",
        }

        # Parse output for model information
        if "llama_model_loader: loaded meta data with" in line:
            parts = line.split()
            self.model_info["kv_pairs"] = parts[6]
            self.model_info["tensors"] = parts[9]
        elif "general.architecture" in line:
            self.model_info["architecture"] = line.split("=")[-1].strip()
        elif line.startswith("llama_model_loader: - kv") and "=" in line:
            # Split on '=' and take the parts
            parts = line.split("=", 1)  # Split only on first '='
            left_part = parts[0].strip()
            value = parts[1].strip()

            # Extract key and type from left part
            # Format: "llama_model_loader: - kv N: key type"
            kv_parts = left_part.split(":")
            if len(kv_parts) >= 3:
                key_type_part = kv_parts[2].strip()  # This is "key type"
                key = key_type_part.rsplit(" ", 1)[
                    0
                ]  # Everything except last word (type)

                # Use human-readable name if available, otherwise use original key
                display_key = key_mappings.get(key, key)

                self.model_info.setdefault("kv_data", {})[display_key] = value
        elif line.startswith("llama_model_loader: - type"):
            parts = line.split(":")
            if len(parts) > 1:
                quant_type = parts[1].strip()
                tensors = parts[2].strip().split()[0]
                self.model_info.setdefault("quantization_type", []).append(
                    f"{quant_type}: {tensors} tensors"
                )

    def parse_progress(self, line, task_item, imatrix_chunks=None) -> None:
        # Parses the output line for progress information and updates the task item.
        match = re.search(r"\[\s*(\d+)\s*/\s*(\d+)\s*].*", line)

        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            progress = int((current / total) * 100)
            task_item.update_progress(progress)
        else:
            imatrix_match = re.search(
                r"compute_imatrix: computing over (\d+) chunks with batch_size \d+",
                line,
            )
            if imatrix_match:
                imatrix_chunks = int(imatrix_match.group(1))
            elif imatrix_chunks is not None:
                if "save_imatrix: stored collected data" in line:
                    save_match = re.search(r"collected data after (\d+) chunks", line)
                    if save_match:
                        saved_chunks = int(save_match.group(1))
                        progress = int((saved_chunks / self.imatrix_chunks) * 100)
                        task_item.update_progress(progress)

    def terminate(self) -> None:
        # Terminate the subprocess if it's still running
        if self.process:
            os.kill(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=5)
            if self.process.poll() is None:
                os.kill(self.process.pid, signal.SIGKILL)
