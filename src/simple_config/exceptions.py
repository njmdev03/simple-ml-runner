class SimpleConfigError(Exception):
    """Base error for simple_config."""
    pass

class ConfigNotFoundError(SimpleConfigError):
    """Raised when a config file or schema is not found."""
    pass

class ConfigTypeError(SimpleConfigError, TypeError):
    """Raised when a config value has the wrong type."""
    pass

class ConfigValidationError(SimpleConfigError, ValueError):
    """Raised when a config value is invalid or missing."""
    pass
