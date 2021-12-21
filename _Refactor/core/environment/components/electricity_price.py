
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class ElectricityPrice(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.electricity_price = None
        self.set_params(params_dict)
