
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class Boiler(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.power_limit = None
        self.var_2 = None
        self.set_params(params_dict)


