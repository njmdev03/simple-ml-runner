class BaseExtension:
    """
    Base class for all extensions.
    Can optionally define config schemas and/or callbacks.
    """
    def __init__(self, global_config, config=None):
        self.global_config = global_config
        self.config = config

    def create_callbacks(self):
        return []
