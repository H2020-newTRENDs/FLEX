
from abc import ABC, abstractmethod
from typing import Dict, Any
from os.path import join, dirname

from _Refactor.basic.db import DB

class GetComponentParamsDictStrategy(ABC):

    @abstractmethod
    def get_params_dict(self) -> Dict[str, Any]:
        pass

class GeneralComponent(GetComponentParamsDictStrategy):

    def __init__(self,
                 table_name: str,
                 row_id: int,
                 db_name: str = "root",
                 db_folder: str = join(dirname(dirname(dirname(__file__))), "data")):
        self.table_name = table_name
        self.row_id = row_id
        self.db_folder = db_folder
        self.db = DB(db_name, self.db_folder)

    def get_params_dict(self) -> Dict[str, Any]:
        table = self.db.read_dataframe(self.table_name)
        param_dict = table.iloc[self.row_id].to_dict()
        return param_dict

