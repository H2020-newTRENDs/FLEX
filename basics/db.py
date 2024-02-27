import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd
import sqlalchemy

if TYPE_CHECKING:
    from basics.config import Config


class DB:
    def __init__(self, config: "Config"):
        self.config = config
        self.connection = self.create_connection(config.project_name)

    def create_connection(self, database_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(
            f'sqlite:///{os.path.join(self.config.project_root, self.config.output, database_name + ".sqlite")}'
        )

    def if_exists(self, table_name: str) -> bool:
        return self.connection.has_table(table_name)

    def if_parquet_exists(self, table_name: str, scenario_ID: int) -> bool:
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        path_to_file = self.config.output / file_name
        return path_to_file.exists()

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
            self,
            table_name: str,
            column_name_plus_value: dict
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

    def write_to_parquet(self, table_name: str, scenario_ID: int, table: pd.DataFrame) -> None:
        """
        is used to save the results. Writes the dataframe into a parquet file with 'table_name_ID' as file name
        """
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        saving_path = self.config.output / file_name
        table.to_parquet(path=saving_path, engine="auto", compression='gzip', index=False)

    def read_parquet(self, table_name: str, scenario_ID: int, column_names: List[str] = None) -> pd.DataFrame:
        """
        Returns: dataframe containing the results for the table name and specific scenario ID
        """
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        path_to_file = self.config.output / file_name
        if column_names:
            df = pd.read_parquet(path=path_to_file, engine="auto", columns=column_names)
        else:
            df = pd.read_parquet(path=path_to_file, engine="auto")
        return df


def create_db_conn(config: "Config") -> DB:
    return DB(config)
