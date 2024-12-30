import os
from typing import TYPE_CHECKING, List, Dict

import pandas as pd
import sqlalchemy
import csv
from utils.tables import InputTables
from sqlalchemy.sql import text

if TYPE_CHECKING:
    from utils.config import Config


class DB:

    def __init__(self, path):
        self.engine = sqlalchemy.create_engine(f'sqlite:///{path}')
        self.metadata = sqlalchemy.MetaData()
        self.metadata.reflect(bind=self.engine)

    def if_exists(self, table_name: str) -> bool:
        return table_name in self.get_table_names()

    def get_engine(self):
        return self.engine

    def close(self):
        self.engine.dispose()

    def get_table_names(self):
        return sqlalchemy.inspect(self.engine).get_table_names()

    def clear_database(self):
        for table_name in self.get_table_names():
            self.drop_table(table_name=table_name)

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
            query = sqlalchemy.select([table.columns[name] for name in column_names])
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
        with self.engine.connect() as conn:
            conn.execute(text(query))

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)


def create_db_conn(config: "Config") -> DB:
    if config.task_id is None:
        conn = DB(os.path.join(config.output, config.project_name + ".sqlite"))
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
                if ext == ".csv":
                    # Detect the separator for CSV
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample = f.read(1024)  # Read a sample to detect the delimiter
                        sniffer = csv.Sniffer()
                        try:
                            delimiter = sniffer.sniff(sample).delimiter
                        except:
                            delimiter = ";"
                    df = pd_read_func(file_path, delimiter=delimiter).dropna(how="all")
                else:
                    df = pd_read_func(file_path).dropna(how="all")
                break
        return df

    for input_table in InputTables:
        df = file_exists(input_table.name)
        if df is not None:
            print(f'Loading input table --> {input_table.name}')
            db.write_dataframe(
                table_name=input_table.name,
                data_frame=df.dropna(axis=1, how="all").dropna(axis=0, how="all"),
                if_exists="replace"
            )


def fetch_input_tables(config: "Config") -> Dict[str, pd.DataFrame]:
    input_tables = {}
    db = create_db_conn(config)
    for table_name in db.get_table_names():
        input_tables[table_name] = db.read_dataframe(table_name)
    return input_tables
