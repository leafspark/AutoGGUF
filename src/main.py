import os
import sys
import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from AutoGGUF import AutoGGUF
from flask import Flask, Response, jsonify

server = Flask(__name__)


def main() -> None:
    @server.route("/v1/models", methods=["GET"])
    def models() -> Response:
        if window:
            return jsonify({"models": window.get_models_data()})
        return jsonify({"models": []})

    @server.route("/v1/tasks", methods=["GET"])
    def tasks() -> Response:
        if window:
            return jsonify({"tasks": window.get_tasks_data()})
        return jsonify({"tasks": []})

    @server.route("/v1/health", methods=["GET"])
    def ping() -> Response:
        return jsonify({"status": "alive"})

    @server.route("/v1/backends", methods=["GET"])
    def get_backends() -> Response:
        backends = []
        for i in range(window.backend_combo.count()):
            backends.append(
                {
                    "name": window.backend_combo.itemText(i),
                    "path": window.backend_combo.itemData(i),
                }
            )
        return jsonify({"backends": backends})

    @server.route("/v1/plugins", methods=["GET"])
    def get_plugins() -> Response:
        if window:
            return jsonify(
                {
                    "plugins": [
                        {
                            "name": plugin_data["data"]["name"],
                            "version": plugin_data["data"]["version"],
                            "description": plugin_data["data"]["description"],
                            "author": plugin_data["data"]["author"],
                        }
                        for plugin_data in window.plugins.values()
                    ]
                }
            )
        return jsonify({"plugins": []})

    def run_flask() -> None:
        if os.environ.get("AUTOGGUF_SERVER", "").lower() == "enabled":
            server.run(
                host="0.0.0.0",
                port=int(os.environ.get("AUTOGGUF_SERVER_PORT", 5000)),
                debug=False,
                use_reloader=False,
            )

    app = QApplication(sys.argv)
    window = AutoGGUF(sys.argv)
    window.show()
    # Start Flask in a separate thread after a short delay
    timer = QTimer()
    timer.singleShot(
        100, lambda: threading.Thread(target=run_flask, daemon=True).start()
    )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
