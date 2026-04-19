class Variant:
    def __init__(self, mapping: dict[str, type], default: str = None):
        self.mapping = mapping
        self.default = default

    def get_default(self):
        if self.default:
            return self.mapping[self.default]
        else:
            return None
