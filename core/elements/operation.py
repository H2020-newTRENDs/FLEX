
from abc import ABC, abstractmethod

"""
abstract flex_operation_old model
"""
class AbstractOperationStrategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

"""
concrete reference flex_operation_old model
"""
class RefOperationStrategy(AbstractOperationStrategy):
    pass




"""
concrete optimization flex_operation_old model
"""
class OptOperationStrategy(AbstractOperationStrategy):
    pass



