
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class Battery(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.ID_Battery: str
        self.Capacity: float
        self.Capacity_unit: str
        self.ChargeEfficiency: float
        self.DischargeEfficiency: float
        self.MaxChargePower: float
        self.MaxChargePower_unit: str
        self.MaxDischargePower: float
        self.MaxDischargePower_unit: str

        self.set_params(params_dict)

