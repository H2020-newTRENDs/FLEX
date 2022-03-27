from abc import ABC
from typing import Dict, Any


class Scenario(ABC):

    def get_scenario_data(self, table_name: str, scenario_filter: dict) -> Dict[str, Any]:
        table = self.db.read_dataframe_new(table_name, filter=scenario_filter)
        if len(table) == 1:  # if it's a single row, then the values should not be packed into lists
            table_dict = table.to_dict("records")[0]
        else:  # if it's an array the values of the dict will be a list
            table_dict = table.to_dict("list")
        return table_dict
