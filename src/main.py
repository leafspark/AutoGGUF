import os
import sys
import threading
from enum import Enum
from typing import List, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from fastapi import FastAPI, Query, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field
from uvicorn import Config, Server

from AutoGGUF import AutoGGUF
from Localizations import AUTOGGUF_VERSION

app = FastAPI(
    title="AutoGGUF",
    description="API for AutoGGUF - automatically quant GGUF models",
    version=AUTOGGUF_VERSION,
    license_info={
        "name": "Apache 2.0",
        "url": "https://raw.githubusercontent.com/leafspark/AutoGGUF/main/LICENSE",
    },
)

# Global variable to hold the window reference
window = None


class ModelType(str, Enum):
    single = "single"
    sharded = "sharded"


class Model(BaseModel):
    name: str = Field(..., description="Name of the model")
    type: str = Field(..., description="Type of the model")
    path: str = Field(..., description="Path to the model file")
    size: Optional[int] = Field(None, description="Size of the model in bytes")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Llama-3.1-8B-Instruct.fp16.gguf",
                "type": "single",
                "path": "Llama-3.1-8B-Instruct.fp16.gguf",
                "size": 13000000000,
            }
        }


class Task(BaseModel):
    # id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    progress: float = Field(..., description="Progress of the task as a percentage")

    class Config:
        json_json_schema_extra = {
            "example": {"id": "task_123", "status": "running", "progress": 75.5}
        }


class Backend(BaseModel):
    name: str = Field(..., description="Name of the backend")
    path: str = Field(..., description="Path to the backend executable")


class Plugin(BaseModel):
    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: str = Field(..., description="Description of the plugin")
    author: str = Field(..., description="Author of the plugin")


# API Key configuration
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(
    api_key_header: str = Security(api_key_header),
) -> Optional[str]:
    api_key_env = os.getenv("AUTOGGUF_SERVER_API_KEY")
    if not api_key_env:
        return None  # No API key restriction if not set

    api_keys = [
        key.strip() for key in api_key_env.split(",") if key.strip()
    ]  # Split by comma and strip whitespace

    if api_key_header and api_key_header.startswith("Bearer "):
        api_key = api_key_header[len("Bearer ") :]
        if api_key in api_keys:
            return api_key

    raise HTTPException(status_code=403, detail="Could not validate API key")


@app.get(
    "/v1/models",
    response_model=List[Model],
    tags=["Models"],
    dependencies=[Depends(get_api_key)],
)
async def get_models(
    type: Optional[ModelType] = Query(None, description="Filter models by type")
) -> List[Model]:
    if window:
        models = window.get_models_data()
        if type:
            models = [m for m in models if m["type"] == type]

        return [Model(**m) for m in models]
    return []


@app.get(
    "/v1/tasks",
    response_model=List[Task],
    tags=["Tasks"],
    dependencies=[Depends(get_api_key)],
)
async def get_tasks() -> List[Task]:
    if window:
        return window.get_tasks_data()
    return []


@app.get("/v1/health", tags=["System"], dependencies=[Depends(get_api_key)])
async def health_check() -> dict:
    return {"status": "alive"}


@app.get(
    "/v1/backends",
    response_model=List[Backend],
    tags=["System"],
    dependencies=[Depends(get_api_key)],
)
async def get_backends() -> List[Backend]:
    backends = []
    if window:
        for i in range(window.backend_combo.count()):
            backends.append(
                Backend(
                    name=window.backend_combo.itemText(i),
                    path=window.backend_combo.itemData(i),
                )
            )
    return backends


@app.get(
    "/v1/plugins",
    response_model=List[Plugin],
    tags=["System"],
    dependencies=[Depends(get_api_key)],
)
async def get_plugins() -> List[Plugin]:
    if window:
        return [
            Plugin(**plugin_data["data"]) for plugin_data in window.plugins.values()
        ]
    return []


def run_uvicorn() -> None:
    if os.environ.get("AUTOGGUF_SERVER", "").lower() == "enabled":
        config = Config(
            app=app,
            host="127.0.0.1",
            port=int(os.environ.get("AUTOGGUF_SERVER_PORT", 7001)),
            log_level="info",
        )
        server = Server(config)
        server.run()


def main() -> None:
    global window
    qt_app = QApplication(sys.argv)
    window = AutoGGUF(sys.argv)
    window.show()

    # Start Uvicorn in a separate thread after a short delay
    timer = QTimer()
    timer.singleShot(
        100, lambda: threading.Thread(target=run_uvicorn, daemon=True).start()
    )

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
