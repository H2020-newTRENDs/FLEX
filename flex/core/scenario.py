from abc import ABC
from enum import Enum
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DB


class Scenario(ABC):
    db: 'DB'

    def get_scenario_data_dict(self, table_name: str, scenario_filter: dict) -> Dict[str, Any]:
        table = self.db.read_dataframe_new(table_name, filter=scenario_filter)
        assert not table.empty, f'No data found in Table {table_name} for the selected scenario.'
        if len(table) == 1:  # if it's a single row, then the values should not be packed into lists
            table_dict = table.to_dict("records")[0]
        else:  # if it's an array the values of the dict will be a list
            table_dict = table.to_dict("list")
        return table_dict


class ScenarioEnum(Enum):

    @property
    def id(self):
        return "ID_" + self.name
