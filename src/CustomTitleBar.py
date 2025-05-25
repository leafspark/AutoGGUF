from PySide6.QtCore import QPoint, Qt
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

        # Enable mouse tracking for smoother movement
        self.setMouseTracking(True)

        # Add maximize button
        self.maximize_button = QPushButton("□")
        self.maximize_button.setFixedSize(30, 30)
        self.maximize_button.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
                padding: 2px;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """
        )
        self.maximize_button.clicked.connect(self.toggle_maximize)

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)

        self.minimize_button.clicked.connect(self.parent.showMinimized)
        self.close_button.clicked.connect(self.parent.close)

        self.start = QPoint(0, 0)
        self.pressing = False
        self.isMaximized = False  # Flag to track maximization state
        self.normal_size = None  # Store the normal window size

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.start = event.globalPos() - self.parent.frameGeometry().topLeft()
            self.pressing = True

    def mouseMoveEvent(self, event) -> None:
        if self.pressing:
            new_pos = event.globalPos() - self.start
            screen = self.parent.screen()
            screen_geo = screen.availableGeometry()

            # Check if the new position would put the titlebar below the taskbar
            if (
                new_pos.y() + self.parent.height() > screen_geo.bottom()
            ):  # Use screen_geo.bottom()
                new_pos.setY(screen_geo.bottom() - self.parent.height())

            self.parent.move(new_pos)

    def mouseReleaseEvent(self, event) -> None:
        self.pressing = False

    def toggle_maximize(self) -> None:
        if self.isMaximized:
            self.parent.showNormal()
            if self.normal_size:
                self.parent.resize(self.normal_size)
            self.maximize_button.setText("□")  # Change back to maximize symbol
            self.isMaximized = False
        else:
            self.normal_size = self.parent.size()  # Store the current size
            self.parent.showMaximized()
            self.maximize_button.setText("❐")  # Change to restore symbol
            self.isMaximized = True
