class QuantizationThread(QThread):
    """
    QuantizationThread is a PyQt6-based thread for managing model quantization processes.

    This class provides functionality for:
    - Running quantization commands as subprocesses
    - Parsing and emitting model information during quantization
    - Logging quantization output to a file
    - Communicating process status, output, and errors to the main thread

    The thread manages the execution of quantization commands, monitors their output,
    and parses relevant model information. It uses Qt signals to communicate various
    events and data back to the main application thread.

    Attributes:
        output_signal (pyqtSignal): Signal emitting subprocess output lines.
        status_signal (pyqtSignal): Signal for updating quantization status.
        finished_signal (pyqtSignal): Signal emitted when quantization is complete.
        error_signal (pyqtSignal): Signal for reporting errors during quantization.
        model_info_signal (pyqtSignal): Signal for sending parsed model information.

    Methods:
        run(): Executes the quantization process and manages its lifecycle.
        parse_model_info(line: str): Parses output lines for model information.
        terminate(): Safely terminates the running subprocess.
    """

    def __init__(self, command, cwd, log_file):
        """
        Initialize the QuantizationThread.

        Args:
            command (list): The command to execute for quantization.
            cwd (str): The working directory for the subprocess.
            log_file (str): Path to the file where output will be logged.
        """

    def run(self):
        """
        Execute the quantization process.

        This method runs the subprocess, captures its output, logs it,
        parses model information, and emits signals for status updates.
        It handles process completion and any exceptions that occur.
        """

    def parse_model_info(self, line):
        """
        Parse a line of subprocess output for model information.

        This method extracts various pieces of model information from
        the output lines and stores them in the model_info dictionary.

        Args:
            line (str): A line of output from the quantization process.
        """

    def terminate(self):
        """
        Terminate the running subprocess.

        This method safely terminates the quantization process if it's
        still running, using SIGTERM first and SIGKILL if necessary.
        """
