import os
import re
import sys
from typing import Any, IO, List, TextIO, Union

from PySide6.QtWidgets import (
    QMessageBox,
)

from Localizations import (
    DOTENV_FILE_NOT_FOUND,
    COULD_NOT_PARSE_LINE,
    ERROR_LOADING_DOTENV,
    AUTOGGUF_VERSION,
)


def verify_gguf(file_path) -> bool:
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            return magic == b"GGUF"
    except (FileNotFoundError, IOError, OSError):
        return False


def process_args(args: List[str]) -> bool:
    try:
        i = 1
        while i < len(args):
            key = (
                args[i][2:].replace("-", "_").upper()
            )  # Strip the first two '--' and replace '-' with '_'
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                value = "enabled"
                i += 1
            os.environ[key] = value
        return True
    except Exception:
        return False


def load_dotenv(self=Any) -> None:
    if not os.path.isfile(".env"):
        self.logger.warning(DOTENV_FILE_NOT_FOUND)
        return

    try:
        with open(".env") as f:
            for line in f:
                # Strip leading/trailing whitespace
                line = line.strip()

                # Ignore comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Match key-value pairs (unquoted and quoted values)
                match = re.match(r"^([^=]+)=(.*)$", line)
                if not match:
                    self.logger.warning(COULD_NOT_PARSE_LINE.format(line))
                    continue

                key, value = match.groups()

                # Remove any surrounding quotes from the value
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]

                # Decode escape sequences
                value = bytes(value, "utf-8").decode("unicode_escape")

                # Set the environment variable
                os.environ[key.strip()] = value.strip()
    except Exception as e:
        self.logger.error(ERROR_LOADING_DOTENV.format(e))


def show_about(self) -> None:
    about_text = f"""AutoGGUF

Version: {AUTOGGUF_VERSION}
        
A tool for managing and converting GGUF models.
This application is licensed under the Apache License 2.0.
Copyright (c) 2024-2025 leafspark.
It also utilizes llama.cpp, licensed under the MIT License.
Copyright (c) 2023-2025 The ggml authors."""
    QMessageBox.about(self, "About AutoGGUF", about_text)


def ensure_directory(path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def open_file_safe(file_path, mode="r") -> IO[Any]:
    encodings = ["utf-8", "latin-1", "ascii", "utf-16"]
    for encoding in encodings:
        try:
            return open(file_path, mode, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"Unable to open file {file_path} with any of the encodings: {encodings}"
    )


def resource_path(relative_path) -> Union[str, str, bytes]:
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller path
        base_path = sys._MEIPASS
    elif "__compiled__" in globals():
        # Nuitka path
        base_path = os.path.dirname(sys.executable)
    else:
        # Regular Python path
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
