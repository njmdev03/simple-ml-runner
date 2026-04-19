class Variant:
    """Represents a configuration variant mapping strings to types.

    Attributes:
        mapping: Dictionary mapping variant names to their respective types.
        default: Optional name of the default variant.
    """
    def __init__(self, mapping: dict[str, type], default: str = None):
        """Initializes Variant.

        Args:
            mapping: Dictionary mapping variant names to types.
            default: Name of default variant. Defaults to None.
        """
        self.mapping = mapping
        self.default = default

    def get_default(self):
        """Gets type of default variant.

        Returns:
            Type of default variant or None if no default set.
        """
        if self.default:
            return self.mapping[self.default]
        else:
            return None
