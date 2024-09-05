import os
import urllib.request
import urllib.error
import zipfile
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
            req = urllib.request.Request(self.url)

            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(
                        self.url, response.status, "HTTP Error", response.headers, None
                    )

                total_size = int(response.headers.get("Content-Length", 0))
                block_size = 8192
                downloaded = 0

                with open(self.save_path, "wb") as file:
                    while True:
                        data = response.read(block_size)
                        if not data:
                            break
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
