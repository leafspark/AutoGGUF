import sys
from PyQt6.QtWidgets import QApplication
from AutoGGUF import AutoGGUF

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoGGUF()
    window.show()
    sys.exit(app.exec())