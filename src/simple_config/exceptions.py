class SimpleConfigError(Exception):
    """Base error for simple_config."""
    pass

class ConfigNotFoundError(SimpleConfigError):
    """Raised when a config file or schema is not found in the registry."""
    pass

class ConfigTypeError(SimpleConfigError, TypeError):
    """Raised when a configuration value has an incorrect type.

    This exception includes information about the field path and the expected vs actual types.
    """
    pass

class ConfigValidationError(SimpleConfigError, ValueError):
    """Raised when configuration data is missing required fields or otherwise invalid.

    This exception typically includes the field path that caused the validation failure.
    """
    pass
