class CLIRegistry:
    """
    Registry for CLI arguments, following the same class-method
    pattern as other registries in the codebase.
    """
    _arguments = []

    @classmethod
    def register(cls, arg):
        cls._arguments.append(arg)

    @classmethod
    def extend(cls, args):
        cls._arguments.extend(args)

    @classmethod
    def get_arguments(cls):
        return cls._arguments

    @classmethod
    def clear(cls):
        cls._arguments = []
