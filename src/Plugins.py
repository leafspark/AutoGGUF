import importlib
import os
from typing import Any, Dict
from Localizations import *


class Plugins:

    def load_plugins(self) -> Dict[str, Dict[str, Any]]:
        plugins = {}
        plugin_dir = "plugins"

        if not os.path.exists(plugin_dir):
            self.logger.info(PLUGINS_DIR_NOT_EXIST.format(plugin_dir))
            return plugins

        if not os.path.isdir(plugin_dir):
            self.logger.warning(PLUGINS_DIR_NOT_DIRECTORY.format(plugin_dir))
            return plugins

        for file in os.listdir(plugin_dir):
            if file.endswith(".py") and not file.endswith(".disabled.py"):
                name = file[:-3]
                path = os.path.join(plugin_dir, file)

                try:
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if isinstance(item, type) and hasattr(item, "__data__"):
                            plugin_instance = item()
                            plugin_data = plugin_instance.__data__()

                            compatible_versions = plugin_data.get(
                                "compatible_versions", []
                            )
                            if (
                                "*" in compatible_versions
                                or AUTOGGUF_VERSION in compatible_versions
                            ):
                                plugins[name] = {
                                    "instance": plugin_instance,
                                    "data": plugin_data,
                                }
                                self.logger.info(
                                    PLUGIN_LOADED.format(
                                        plugin_data["name"], plugin_data["version"]
                                    )
                                )
                            else:
                                self.logger.warning(
                                    PLUGIN_INCOMPATIBLE.format(
                                        plugin_data["name"],
                                        plugin_data["version"],
                                        AUTOGGUF_VERSION,
                                        ", ".join(compatible_versions),
                                    )
                                )
                            break
                except Exception as e:
                    self.logger.error(PLUGIN_LOAD_FAILED.format(name, str(e)))

        return plugins

    def apply_plugins(self) -> None:
        if not self.plugins:
            self.logger.info(NO_PLUGINS_LOADED)
            return

        for plugin_name, plugin_info in self.plugins.items():
            plugin_instance = plugin_info["instance"]
            for attr_name in dir(plugin_instance):
                if not attr_name.startswith("__") and attr_name != "init":
                    attr_value = getattr(plugin_instance, attr_name)
                    setattr(self, attr_name, attr_value)

            if hasattr(plugin_instance, "init") and callable(plugin_instance.init):
                plugin_instance.init(self)
