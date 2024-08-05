from PyQt6.QtWidgets import QMessageBox
from localizations import *

def show_error(logger, message):
    logger.error(ERROR_MESSAGE.format(message))
    QMessageBox.critical(None, ERROR, message)


def handle_error(logger, error_message, task_item):
    logger.error(TASK_ERROR.format(error_message))
    show_error(logger, error_message)
    task_item.update_status(ERROR)