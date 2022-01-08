
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class Battery(Component):

    def __init__(self):
        self.ID_Battery: str = None
        self.Capacity: float = None
        self.Capacity_unit: str = None
        self.ChargeEfficiency: float = None
        self.DischargeEfficiency: float = None
        self.MaxChargePower: float = None
        self.MaxChargePower_unit: str = None
        self.MaxDischargePower: float = None
        self.MaxDischargePower_unit: str = None


