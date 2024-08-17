class Logger:
    """
    This module provides a custom logger class for logging messages to both the console and a rotating log file.

    The log file will be created in the specified `log_dir` with a timestamp in the filename.
    The file will rotate when it reaches 10MB, keeping a maximum of 5 backup files.
    """

    def __init__(self, name, log_dir):
        """
        Initializes the logger with a specified name and log directory.

        Args:
            name (str): The name of the logger.
            log_dir (str): The directory where log files will be stored.
        """

    def debug(self, message):
        """
        Logs a message with the DEBUG level.

        Args:
            message (str): The message to log.
        """

    def info(self, message):
        """
        Logs a message with the INFO level.

        Args:
            message (str): The message to log.
        """

    def warning(self, message):
        """
        Logs a message with the WARNING level.

        Args:
            message (str): The message to log.
        """

    def error(self, message):
        """
        Logs a message with the ERROR level.

        Args:
            message (str): The message to log.
        """

    def critical(self, message):
        """
        Logs a message with the CRITICAL level.

        Args:
            message (str): The message to log.
        """
