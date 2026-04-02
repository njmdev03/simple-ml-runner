from abc import ABC, abstractmethod

class ConfigParser(ABC):
    @abstractmethod
    def load(self, path: str) -> dict:
        pass