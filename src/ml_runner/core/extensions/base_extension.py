class BaseExtension:
    """
    Base class for all extensions.
    Can optionally define config schemas and/or callbacks.
    """
    def create_callbacks(self, run_config):
        return []
