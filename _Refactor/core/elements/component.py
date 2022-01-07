
from typing import Dict, Any


class Component:

    def set_params(self, params: Dict[str, Any]):
        for paramName, paramValue in params.items():
            paramName_lower = paramName.lower()
            if paramName_lower in self.__dict__.keys():
                setattr(self, paramName_lower, paramValue)

