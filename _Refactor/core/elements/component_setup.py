
from abc import ABC, abstractmethod
from typing import Dict, Any
from os.path import join, dirname
from pathlib import Path

from _Refactor.basic.db import DB
from _Refactor.basic.config import Config


class GetComponentParamsDictStrategy(ABC):

    @abstractmethod
    def get_params_dict(self) -> Dict[str, Any]:
        pass


class GeneralComponent(GetComponentParamsDictStrategy):

    def __init__(self,
                 row_id: int,
                 table_name: str,
                 db_name: str = "root",
                 db_folder: str = Path(__file__).parent / Path("data")):
        self.table_name = table_name
        self.row_id = row_id
        self.db = DB()

    def get_params_dict(self) -> Dict[str, Any]:
        table = self.db.read_dataframe(self.table_name)
        param_dict = table.iloc[self.row_id].to_dict()
        return param_dict
