import sys
import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from AutoGGUF import AutoGGUF
from flask import Flask, jsonify

server = Flask(__name__)


@server.route("/v1/models", methods=["GET"])
def models():
    if window:
        return jsonify({"models": window.get_models_data()})
    return jsonify({"models": []})


@server.route("/v1/tasks", methods=["GET"])
def tasks():
    if window:
        return jsonify({"tasks": window.get_tasks_data()})
    return jsonify({"tasks": []})


@server.route("/v1/health", methods=["GET"])
def ping():
    return jsonify({"status": "alive"})


@server.route("/v1/backends", methods=["GET"])
def get_backends():
    backends = []
    for i in range(window.backend_combo.count()):
        backends.append(
            {
                "name": window.backend_combo.itemText(i),
                "path": window.backend_combo.itemData(i),
            }
        )
    return jsonify({"backends": backends})


def run_flask():
    server.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


app = QApplication(sys.argv)
window = AutoGGUF()
window.show()
# Start Flask in a separate thread after a short delay
timer = QTimer()
timer.singleShot(100, lambda: threading.Thread(target=run_flask, daemon=True).start())
sys.exit(app.exec())
