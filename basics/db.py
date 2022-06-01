import os
from typing import TYPE_CHECKING

import pandas as pd
import sqlalchemy

if TYPE_CHECKING:
    from .config import Config


class DB:

    def __init__(self, config: 'Config'):
        self.config = config
        self.connection = self.create_connection(config.project_name)

    def create_connection(self, database_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(f'sqlite:///{os.path.join(self.config.output, database_name + ".sqlite")}')

    def get_engine(self):
        return self.connection

    def close(self):
        self.connection.dispose()

    def clear_database(self):
        for table_name in self.connection.table_names():
            self.connection.execute(f"drop table {table_name}")

    def drop_table(self, table_name: str):
        self.connection.execute(f"drop table if exists {table_name}")

    def write_dataframe(self,
                        table_name: str,
                        data_frame: pd.DataFrame,
                        data_types: dict = None,
                        if_exists='append'):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(table_name, self.connection, index=False, dtype=data_types, if_exists=if_exists,
                          chunksize=10_000)

    def read_dataframe(self, table_name: str, filter: dict = None, column_names: list = None) -> pd.DataFrame:
        df = pd.read_sql(f'select * from {table_name}', self.connection)
        if filter is not None:
            df = df.loc[(df[list(filter)] == pd.Series(filter)).all(axis=1)]
        if column_names is not None:
            df = df[column_names]
        return df

    def delete_row_from_table(self, table_name: str, column_name_plus_value: dict) -> None:
        condition_temp = ""
        for key, value in column_name_plus_value.items():
            condition_temp += key + " == '" + str(value) + "' and "
        condition = " where " + condition_temp
        condition = condition[0:-5]  # deleting last "and"

        query = f"DELETE FROM {table_name}" + condition
        self.connection.execute(query)

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.connection)


def create_db_conn(config: 'Config') -> DB:
    return DB(config)
