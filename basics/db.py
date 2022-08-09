import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd
import sqlalchemy

if TYPE_CHECKING:
    from .config import Config


class DB:
    def __init__(self, config: "Config"):
        self.config = config
        self.connection = self.create_connection(config.project_name)

    def create_connection(self, database_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(
            f'sqlite:///{os.path.join(self.config.output, database_name + ".sqlite")}'
        )

    def get_engine(self):
        return self.connection

    def close(self):
        self.connection.dispose()

    def clear_database(self):
        for table_name in self.connection.table_names():
            self.connection.execute(f"drop table {table_name}")

    def drop_table(self, table_name: str):
        self.connection.execute(f"drop table if exists {table_name}")

    def write_dataframe(
        self,
        table_name: str,
        data_frame: pd.DataFrame,
        data_types: dict = None,
        if_exists="append",
    ):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(
            table_name,
            self.connection,
            index=False,
            dtype=data_types,
            if_exists=if_exists,
            chunksize=10_000,
        )

    def read_dataframe(self, table_name: str, filter: dict = None, column_names: List[str] = None) -> pd.DataFrame:
        """column_names will only extract certain columns (list)
        filter have to be dict with: column_name: value. The table where the column has that value is returned

        Returns:
            object: pandas Dataframe"""
        if filter is None:
            filter = {}
        if column_names is None:
            column_names = []
        if len(column_names) > 0:
            columns2extract = ""
            for name in column_names:
                columns2extract += name + ','
            columns2extract = columns2extract[:-1]  # delete last ","
        else:
            columns2extract = "*"  # select all columns
        if len(filter) > 0:
            condition_temp = ""
            for key, value in filter.items():
                condition_temp += key + " == '" + str(value) + "' and "
            condition = " where " + condition_temp
            condition = condition[0:-5]  # deleting last "and"
        else:
            condition = ''

        dataframe = pd.read_sql('select ' + columns2extract + ' from ' + table_name + condition, con=self.connection)
        return dataframe

    def delete_row_from_table(
        self, table_name: str, column_name_plus_value: dict
    ) -> None:
        condition_temp = ""
        for key, value in column_name_plus_value.items():
            condition_temp += key + " == '" + str(value) + "' and "
        condition = " where " + condition_temp
        condition = condition[0:-5]  # deleting last "and"

        query = f"DELETE FROM {table_name}" + condition
        self.connection.execute(query)

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.connection)

    def save_dataframe_to_operation_input(self, df: pd.DataFrame, table_name):
        path = self.config.input_operation / Path(table_name + ".xlsx")
        df.to_excel(path, index=False)

    def save_dataframe_to_investment_input(self, df: pd.DataFrame, table_name):
        path = self.config.input_investment / Path(table_name + ".xlsx")
        df.to_excel(path, index=False)


def create_db_conn(config: "Config") -> DB:
    return DB(config)
