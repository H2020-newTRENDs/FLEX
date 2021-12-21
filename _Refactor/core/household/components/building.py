
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class Building(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.var_1 = None
        self.var_2 = None
        self.set_params(params_dict)

