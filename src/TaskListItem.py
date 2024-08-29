from PySide6.QtCore import *
from PySide6.QtGui import QAction
from PySide6.QtWidgets import *

from Localizations import (
    DELETING_TASK,
    CANCELLING_TASK,
    CONFIRM_DELETION_TITLE,
    CONFIRM_DELETION,
    SHOWING_TASK_CONTEXT_MENU,
    CANCELED,
    CANCEL,
    PROPERTIES,
    COMPLETED,
    SHOWING_PROPERTIES_FOR_TASK,
    DELETE,
    RESTART,
)
from ModelInfoDialog import ModelInfoDialog


class TaskListItem(QWidget):
    def __init__(
        self, task_name, log_file, show_progress_bar=True, parent=None
    ) -> None:
        super().__init__(parent)
        self.task_name = task_name
        self.log_file = log_file
        self.status = "Pending"
        layout = QHBoxLayout(self)
        self.task_label = QLabel(task_name)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel(self.status)
        layout.addWidget(self.task_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        # Hide progress bar if show_progress_bar is False
        self.progress_bar.setVisible(show_progress_bar)

        # Use indeterminate progress bar if not showing percentage
        if not show_progress_bar:
            self.progress_bar.setRange(0, 0)

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0

    def show_task_context_menu(self, position) -> None:
        self.logger.debug(SHOWING_TASK_CONTEXT_MENU)
        item = self.task_list.itemAt(position)
        if item is not None:
            context_menu = QMenu(self)

            properties_action = QAction(PROPERTIES, self)
            properties_action.triggered.connect(lambda: self.show_task_properties(item))
            context_menu.addAction(properties_action)

            task_item = self.task_list.itemWidget(item)
            if task_item.status != COMPLETED:
                cancel_action = QAction(CANCEL, self)
                cancel_action.triggered.connect(lambda: self.cancel_task(item))
                context_menu.addAction(cancel_action)

            if task_item.status == CANCELED:
                restart_action = QAction(RESTART, self)
                restart_action.triggered.connect(lambda: self.restart_task(task_item))
                context_menu.addAction(restart_action)

            delete_action = QAction(DELETE, self)
            delete_action.triggered.connect(lambda: self.delete_task(item))
            context_menu.addAction(delete_action)

            context_menu.exec(self.task_list.viewport().mapToGlobal(position))

    def show_task_properties(self, item) -> None:
        self.logger.debug(SHOWING_PROPERTIES_FOR_TASK.format(item.text()))
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                model_info_dialog = ModelInfoDialog(thread.model_info, self)

                model_info_dialog.exec()
                break

    def cancel_task_by_item(self, item) -> None:
        task_item = self.task_list.itemWidget(item)
        for thread in self.quant_threads:
            if thread.log_file == task_item.log_file:
                thread.terminate()
                task_item.update_status(CANCELED)
                self.quant_threads.remove(thread)
                break

    def cancel_task(self, item) -> None:
        self.logger.info(CANCELLING_TASK.format(item.text()))
        self.cancel_task_by_item(item)

    def delete_task(self, item) -> None:
        self.logger.info(DELETING_TASK.format(item.text()))

        # Cancel the task first
        self.cancel_task_by_item(item)

        reply = QMessageBox.question(
            self,
            CONFIRM_DELETION_TITLE,
            CONFIRM_DELETION,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            task_item = self.task_list.itemWidget(item)
            row = self.task_list.row(item)
            self.task_list.takeItem(row)

            if task_item:
                task_item.deleteLater()

    def update_status(self, status) -> None:
        self.status = status
        self.status_label.setText(status)
        if status == "In Progress":
            # Only start timer if showing percentage progress
            if self.progress_bar.isVisible():
                self.progress_bar.setRange(0, 100)
                self.progress_timer.start(100)
        elif status == "Completed":
            self.progress_timer.stop()
            self.progress_bar.setValue(100)
        elif status == "Canceled":
            self.progress_timer.stop()
            self.progress_bar.setValue(0)

    def set_error(self) -> None:
        self.status = "Error"
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: red;")
        self.progress_bar.setRange(0, 100)
        self.progress_timer.stop()

    def update_progress(self, value=None) -> None:
        if value is not None:
            # Update progress bar with specific value
            self.progress_value = value
            self.progress_bar.setValue(self.progress_value)
        else:
            # Increment progress bar for indeterminate progress
            self.progress_value = (self.progress_value + 1) % 101
            self.progress_bar.setValue(self.progress_value)
