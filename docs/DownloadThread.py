import os
import zipfile

import requests
from PySide6.QtCore import QThread, Signal


class DownloadThread(QThread):
    """
    A QThread subclass for downloading and extracting zip files.

    This thread downloads a file from a given URL, saves it to a specified path,
    extracts its contents if it's a zip file, and then removes the original zip file.

    Signals:
        progress_signal (int): Emits the download progress as a percentage.
        finished_signal (str): Emits the path of the extracted directory upon successful completion.
        error_signal (str): Emits an error message if an exception occurs during the process.
    """

    def __init__(self, url: str, save_path: str) -> None:
        """
        Initialize the DownloadThread.

        Args:
            url (str): The URL of the file to download.
            save_path (str): The local path where the file will be saved.
        """

    def run(self) -> None:
        """
        Execute the download, extraction, and cleanup process.

        This method performs the following steps:
        1. Downloads the file from the specified URL.
        2. Saves the file to the specified path.
        3. Extracts the contents if it's a zip file.
        4. Removes the original zip file after extraction.
        5. Emits signals for progress updates, completion, or errors.

        Raises:
            Exception: Any exception that occurs during the process is caught
                       and emitted through the error_signal.
        """
