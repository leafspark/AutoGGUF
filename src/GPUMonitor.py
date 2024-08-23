import pynvml
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QProgressBar,
    QLabel,
    QDialog,
    QTabWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsLineItem,
    QComboBox,
)

from Localizations import (
    GPU_USAGE_FORMAT,
    GPU_DETAILS,
    GPU_USAGE_OVER_TIME,
    VRAM_USAGE_OVER_TIME,
    NO_GPU_DETECTED,
    AMD_GPU_NOT_SUPPORTED,
)

from ui_update import animate_bar


class SimpleGraph(QGraphicsView):
    def __init__(self, title, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.setMinimumHeight(200)
        self.title = title
        self.data = []

    def update_data(self, data) -> None:
        self.data = data
        self.scene().clear()
        if not self.data:
            return

        width = self.width() - 40
        height = self.height() - 40
        max_value = 100  # Fixed to 100% for GPU usage

        # Draw axes
        self.scene().addLine(20, height + 20, width + 20, height + 20)
        self.scene().addLine(20, 20, 20, height + 20)

        # Draw title
        self.scene().addText(self.title).setPos(width // 2, 0)

        # Draw graph
        path = QPen(QColor(0, 120, 212), 2)  # Blue color, 2px width
        for i in range(1, len(self.data)):
            x1 = 20 + (i - 1) * width / (len(self.data) - 1)
            y1 = 20 + height - (self.data[i - 1] * height / max_value)
            x2 = 20 + i * width / (len(self.data) - 1)
            y2 = 20 + height - (self.data[i] * height / max_value)
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(path)
            self.scene().addItem(line)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_data(self.data)


class GPUMonitor(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gpu_selector = QComboBox()
        self.gpu_selector.setVisible(False)
        self.gpu_selector.currentIndexChanged.connect(self.change_gpu)
        layout.addWidget(self.gpu_selector)

        self.gpu_bar = QProgressBar()
        self.gpu_bar.setTextVisible(False)
        layout.addWidget(self.gpu_bar)

        self.gpu_label = QLabel()
        layout.addWidget(self.gpu_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gpu_info)
        self.timer.start(200)  # Update every 0.2 seconds

        self.gpu_data = []
        self.vram_data = []

        self.handles = []
        self.current_gpu = 0

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                # Handle both string and bytes cases
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                self.handles.append(handle)
                self.gpu_selector.addItem(f"NVIDIA GPU {i}: {name}")

            if device_count > 1:
                self.gpu_selector.setVisible(True)

            if device_count == 0:
                self.check_for_amd_gpu()

        except pynvml.NVMLError:
            self.check_for_amd_gpu()

        if not self.handles:
            self.gpu_label.setText(NO_GPU_DETECTED)

    def check_for_amd_gpu(self) -> None:
        # This is a placeholder. Implementing AMD GPU detection would require
        # platform-specific methods or additional libraries.
        self.gpu_label.setText(AMD_GPU_NOT_SUPPORTED)

    def change_gpu(self, index) -> None:
        self.current_gpu = index
        self.gpu_data.clear()
        self.vram_data.clear()

    def update_gpu_info(self) -> None:
        if self.handles:
            try:
                handle = self.handles[self.current_gpu]
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_usage = utilization.gpu
                vram_usage = (memory.used / memory.total) * 100

                animate_bar(self, self.gpu_bar, int(vram_usage))
                self.gpu_label.setText(
                    GPU_USAGE_FORMAT.format(
                        gpu_usage,
                        vram_usage,
                        memory.used // 1024 // 1024,
                        memory.total // 1024 // 1024,
                    )
                )

                self.gpu_data.append(gpu_usage)
                self.vram_data.append(vram_usage)

                if len(self.gpu_data) > 60:
                    self.gpu_data.pop(0)
                    self.vram_data.pop(0)
            except pynvml.NVMLError:
                self.gpu_bar.setValue(0)
                self.gpu_label.setText(GPU_USAGE_FORMAT.format(0, 0, 0, 0))

    def mouseDoubleClickEvent(self, event) -> None:
        if self.handles:
            self.show_detailed_stats()

    def show_detailed_stats(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(GPU_DETAILS)
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(dialog)

        if len(self.handles) > 1:
            gpu_selector = QComboBox()
            gpu_selector.addItems(
                [
                    self.gpu_selector.itemText(i)
                    for i in range(self.gpu_selector.count())
                ]
            )
            gpu_selector.setCurrentIndex(self.current_gpu)
            gpu_selector.currentIndexChanged.connect(self.change_gpu)
            layout.addWidget(gpu_selector)

        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        gpu_graph = SimpleGraph(GPU_USAGE_OVER_TIME)
        vram_graph = SimpleGraph(VRAM_USAGE_OVER_TIME)

        def update_graph_data() -> None:
            gpu_graph.update_data(self.gpu_data)
            vram_graph.update_data(self.vram_data)

        timer = QTimer(dialog)
        timer.timeout.connect(update_graph_data)
        timer.start(200)  # Update every 0.2 seconds

        tab_widget.addTab(gpu_graph, GPU_USAGE_OVER_TIME)
        tab_widget.addTab(vram_graph, VRAM_USAGE_OVER_TIME)

        dialog.exec()

    def closeEvent(self, event) -> None:
        if self.handles:
            pynvml.nvmlShutdown()
        super().closeEvent(event)
