
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class FeedinTariff(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.feedin_tariff = None
        self.set_params(params_dict)
