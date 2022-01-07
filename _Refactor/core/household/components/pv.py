
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class PV(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.ID_PV: int
        self.PVPower: float
        self.PVPower_unit: str

        self.set_params(params_dict)

