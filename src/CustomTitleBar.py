from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenuBar, QPushButton, QWidget


class CustomTitleBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent = parent
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Add the favicon
        # TODO: uncomment this
        # self.icon_label = QLabel()
        # self.icon_label.setPixmap(QPixmap(resource_path("assets/favicon.ico")))
        # layout.addWidget(self.icon_label)

        # Add app title (bolded)
        self.title = QLabel("<b>AutoGGUF</b>")  # Use HTML tags for bolding
        layout.addWidget(self.title)

        # Add menubar here
        self.menubar = QMenuBar()
        layout.addWidget(self.menubar)  # Add menubar to the layout

        layout.addStretch(1)  # This pushes the buttons to the right

        # Add minimize and close buttons
        self.minimize_button = QPushButton("—")
        self.close_button = QPushButton("✕")

        for button in (self.minimize_button, self.close_button):
            button.setFixedSize(30, 30)
            button.setStyleSheet(
                """
                QPushButton {
                    border: none;
                    background-color: transparent;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
            """
            )

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)

        self.minimize_button.clicked.connect(self.parent.showMinimized)
        self.close_button.clicked.connect(self.parent.close)

        self.start = QPoint(0, 0)
        self.pressing = False

    def mousePressEvent(self, event) -> None:
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event) -> None:
        if self.pressing:
            end = self.mapToGlobal(event.pos())
            movement = end - self.start
            self.parent.setGeometry(
                self.parent.x() + movement.x(),
                self.parent.y() + movement.y(),
                self.parent.width(),
                self.parent.height(),
            )
            self.start = end

    def mouseReleaseEvent(self, event) -> None:
        self.pressing = False
