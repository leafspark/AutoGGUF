from PyQt6.QtWidgets import *


class ModelInfoDialog(QDialog):
    def __init__(self, model_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Information")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(self.format_model_info(model_info))

        layout.addWidget(info_text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def format_model_info(self, model_info):
        html = "<h2>Model Information</h2>"
        html += f"<p><b>Architecture:</b> {model_info.get('architecture', 'N/A')}</p>"
        html += f"<p><b>Quantization Type:</b> {model_info.get('quantization_type', 'N/A')}</p>"
        html += f"<p><b>KV Pairs:</b> {model_info.get('kv_pairs', 'N/A')}</p>"
        html += f"<p><b>Tensors:</b> {model_info.get('tensors', 'N/A')}</p>"

        html += "<h3>Key-Value Pairs:</h3>"
        for key, value in model_info.get("kv_data", {}).items():
            html += f"<p><b>{key}:</b> {value}</p>"

        return html
