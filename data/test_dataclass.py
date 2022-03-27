from abc import ABC
import sqlalchemy
import pandas as pd
from typing import get_type_hints
from dataclasses import dataclass, field
from basic.db import DB


component_scenarios = {
    'capacity': [0, 7_000],
    'capacity_unit': ['Wh', 'Wh'],
    'charge_efficiency': [0, 0.95],
    'discharge_efficiency': [0, 0.95],
    'charge_power_max': [0, 4500],
    'charge_power_max_unit': ['W', 'W'],
    'discharge_power_max': [0, 4500],
    'discharge_power_max_unit': ['W', 'W'],
}

df = pd.DataFrame(component_scenarios)
print(df)