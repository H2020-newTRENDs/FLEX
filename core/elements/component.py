from core.elements.component_setup import GeneralComponent
from typing import Dict, Any


class Component:

    def set_parameters(self, component_id: int):
        # get the data from the root database:
        params = GeneralComponent(row_id=component_id, table_name=self.__class__.__name__).get_complete_data()

        for paramName, paramValue in params.items():
            if paramName in self.__dict__.keys():
                setattr(self, paramName, paramValue)

