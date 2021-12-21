
from abc import ABC, abstractmethod

"""
abstract operation model
"""
class AbstractOperationStrategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

"""
concrete reference operation model
"""
class RefOperationStrategy(AbstractOperationStrategy):
    pass




"""
concrete optimization operation model
"""
class OptOperationStrategy(AbstractOperationStrategy):
    pass



