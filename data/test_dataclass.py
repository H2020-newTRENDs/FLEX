from abc import ABC
import sqlalchemy
import pandas as pd
from typing import get_type_hints
from dataclasses import dataclass, field
from basic.db import DB


class AAA(ABC):

    def print_self(self):
        print(self.__class__.__name__)


@dataclass
class CCC(AAA):
    ccc_id: int
    attr1: float = 0.0
    attr2: int = 0
    attr3: str = ''

    def __post_init__(self):
        self.attr2 = self.get_attr2()

    def get_attr2(self):
        return self.ccc_id + 2

    @staticmethod
    def get_table_info():
        info = "table info"
        return info


print(CCC.__name__)
print(get_type_hints(CCC))

ccc = CCC(ccc_id=1)
print(ccc)
ccc.print_self()

# dtypes = get_type_hints(CCC)
# print(dtypes)
#
# _type_py2sql_dict = {
#     int: sqlalchemy.types.BigInteger,
#     str: sqlalchemy.types.Unicode,
#     float: sqlalchemy.types.Float,
# }
# for key, value in dtypes.items():
#     dtypes[key] = _type_py2sql_dict[value]
# print(dtypes)


# table_dict = {"attr1": 2.2,
#               "attr2": 3,
#               "attr3": "test"}
# table = pd.DataFrame(table_dict, index=[0])
# dtypes = get_type_hints(CCC)
# print(dtypes)
# for key, value in dtypes.items():
#     dtypes[key] = _type_py2sql_dict[value]
# print(dtypes)
#
# DB().write_dataframe(table_name="CCC",
#                      data_frame=table,
#                      data_types=dtypes,
#                      if_exists="replace")
