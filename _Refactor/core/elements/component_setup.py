
from abc import ABC, abstractmethod
from typing import Dict, Any
from os.path import join, dirname
from pathlib import Path

import pandas as pd

from _Refactor.basic.db import DB
import _Refactor.basic.config as config


class GetComponentParamsDictStrategy(ABC):

    @abstractmethod
    def get_complete_data(self) -> Dict[str, Any]:
        pass


class GeneralComponent(GetComponentParamsDictStrategy):

    def __init__(self,
                 row_id: int,
                 table_name: str,
                 db_name: str = "root",
                 db_folder: str = Path(__file__).parent.parent.parent / Path("data")):
        data_base_connection = config.create_connection(db_folder=db_folder, db_name=db_name)
        self.table_name = table_name
        self.row_id = row_id
        self.db = DB(data_base_connection)

    def get_complete_data(self) -> pd.DataFrame:
        # reads only the data where the ID is equal to the row ID. This way we can read in profiles with 8760 values
        kwargs_dict = {"ID_" + self.table_name: self.row_id}
        table = self.db.read_dataframe(self.table_name, **kwargs_dict)
        return table



