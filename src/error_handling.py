from PySide6.QtWidgets import QMessageBox
from Localizations import ERROR_MESSAGE, ERROR, TASK_ERROR


def show_error(logger, message) -> None:
    logger.error(message)
    QMessageBox.critical(None, ERROR, message)


def handle_error(logger, error_message, task_item) -> None:
    logger.error(TASK_ERROR.format(error_message))
    show_error(logger, error_message)
    task_item.update_status(ERROR)
