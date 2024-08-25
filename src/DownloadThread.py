import os
import zipfile

import requests
from PySide6.QtCore import QThread, Signal


class DownloadThread(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, url, save_path) -> None:
        super().__init__()
        self.url = url
        self.save_path = save_path

    def run(self) -> None:
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192
            downloaded = 0

            with open(self.save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    downloaded += size
                    if total_size:
                        progress = int((downloaded / total_size) * 100)
                        self.progress_signal.emit(progress)

            # Extract the downloaded zip file
            extract_dir = os.path.splitext(self.save_path)[0]
            with zipfile.ZipFile(self.save_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Remove the zip file after extraction
            os.remove(self.save_path)

            self.finished_signal.emit(extract_dir)
        except Exception as e:
            self.error_signal.emit(str(e))
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
