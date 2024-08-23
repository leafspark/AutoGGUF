class ExamplePlugin:
    def init(self, autogguf_instance):
        # This gets called after the plugin is loaded
        print("Plugin initialized")

    def __data__(self):
        return {
            "name": "ExamplePlugin",
            "description": "This is an example plugin.",
            "compatible_versions": ["*"],
            "author": "leafspark",
            "version": "v1.0.0",
        }
