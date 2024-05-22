import os
from typing import TYPE_CHECKING, List
from pathlib import Path

import pandas as pd
import sqlalchemy

if TYPE_CHECKING:
    from flex.config import Config


class DB:

    def __init__(self, path):
        self.engine = sqlalchemy.create_engine(f'sqlite:///{path}')
        self.metadata = sqlalchemy.MetaData()
        self.metadata.reflect(bind=self.engine)

    def if_exists(self, table_name: str) -> bool:
        return table_name in self.get_table_names()

    def if_parquet_exists(self, table_name: str, scenario_ID: int) -> bool:
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        path_to_file = self.config.output / file_name
        return path_to_file.exists()

    def get_engine(self):
        return self.engine

    def dispose(self):
        self.engine.dispose()

    def get_table_names(self):
        return sqlalchemy.inspect(self.engine).get_table_names()

    def clear_database(self):
        for table_name in self.get_table_names():
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(f"drop table {table_name}"))

    def drop_table(self, table_name: str):
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(f"drop table if exists {table_name}"))

    def write_dataframe(
            self,
            table_name: str,
            data_frame: pd.DataFrame,
            data_types: dict = None,
            if_exists="append",
    ):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(
            table_name,
            self.engine,
            index=False,
            dtype=data_types,
            if_exists=if_exists,
            chunksize=10_000,
        )
    
    
    def write_to_parquet(self, table_name: str, scenario_ID: int, table: pd.DataFrame, folder: Path) -> None:
        """
        is used to save the results. Writes the dataframe into a parquet file with 'table_name_ID' as file name
        """
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        saving_path = folder / file_name
        table.to_parquet(path=saving_path, engine="auto", compression='gzip', index=False)

    def read_parquet(self, table_name: str, scenario_ID: int, folder: Path, column_names: List[str] = None) -> pd.DataFrame:
        """
        Returns: dataframe containing the results for the table name and specific scenario ID
        """
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        path_to_file = folder / file_name
        if column_names:
            df = pd.read_parquet(path=path_to_file, engine="auto", columns=column_names)
        else:
            df = pd.read_parquet(path=path_to_file, engine="auto")
        return df

    def read_dataframe(self, table_name: str, filter: dict = None, column_names: List[str] = None) -> pd.DataFrame:
        """Reads data from a database table with optional filtering and column selection.

                Args:
                    table_name (str): Name of the table to query.
                    filter (dict, optional): Dictionary with {column_name: value} to filter the data.
                    column_names (list of str, optional): List of column names to extract.

                Returns:
                    pd.DataFrame: Resulting dataframe.
                """
        table = self.metadata.tables[table_name]

        if column_names:
            if len(column_names) > 1:
                query = sqlalchemy.select([table.columns[name] for name in column_names])
            else:
                query = sqlalchemy.select([table.columns[name] for name in column_names][0])
        else:
            query = sqlalchemy.select(table)

        if filter:
            for key, value in filter.items():
                query = query.where(table.columns[key] == value)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

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
        with self.engine.connect() as connection:
            connection.execute(sqlalchemy.text(query))

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)



def create_db_conn(config: "Config") -> DB:
    if config.task_id is None:
        conn = DB(config.output / f"{config.project_name}.sqlite")
    else:
        conn = DB(os.path.join(config.task_output, f'{config.project_name}.sqlite'))
    return conn


def init_project_db(config: "Config"):
    db = create_db_conn(config)
    db.clear_database()

    def file_exists(table_name: str):
        df = None
        extensions = {".xlsx": pd.read_excel,
                      ".csv": pd.read_csv}
        for ext, pd_read_func in extensions.items():
            file_path = os.path.join(config.input, table_name + ext)
            if os.path.exists(file_path):
                df = pd_read_func(file_path)
                break
        return df
