
from abc import ABC, abstractmethod

class SetupStrategy(ABC):

    @abstractmethod
    def setup(self):
        pass