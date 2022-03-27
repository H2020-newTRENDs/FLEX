import os
import sqlalchemy
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class DB:

    def __init__(self, config: 'Config'):
        self.folder_path = config.database_folder
        self.connection = self.create_connection(config.project_name)

    def create_connection(self, database_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(f'sqlite:///{os.path.join(self.folder_path, database_name + ".sqlite")}')

    def get_engine(self):
        return self.connection

    def close(self):
        self.connection.dispose()

    def clear_table_from_database(self):
        for table_name in self.connection.table_names():
            self.connection.execute(f"drop table {table_name}")

    def write_dataframe(self,
                        table_name: str,
                        data_frame: pd.DataFrame,
                        data_types: dict = None,
                        if_exists='append'):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(table_name, self.connection, index=False, dtype=data_types, if_exists=if_exists,
                          chunksize=10_000)

    def read_dataframe(self, table_name: str, *column_names: list, **kwargs: dict) -> pd.DataFrame:
        """column_names will only extract certain columns (list)
        kwargs have to be dict with: column_name: value. The table where the column has that value is returned"""
        if len(column_names) > 0:
            columns2extract = ""
            for name in column_names:
                columns2extract += name + ','
            columns2extract = columns2extract[:-1]  # delete last ","
        else:
            columns2extract = "*"  # select all columns
        if len(kwargs) > 0:
            condition_temp = ""
            for key, value in kwargs.items():
                condition_temp += key + " == '" + str(value) + "' and "
            condition = " where " + condition_temp
            condition = condition[0:-5]  # deleting last "and"
        else:
            condition = ''

        DataFrame = pd.read_sql('select ' + columns2extract + ' from ' + table_name + condition, con=self.connection)
        return DataFrame

    def read_dataframe_new(self, table_name: str, filter: dict = None, column_names: list = None) -> pd.DataFrame:
        df = pd.read_sql(f'select * from {table_name}', self.connection)
        if filter is not None:
            df = df.loc[(df[list(filter)] == pd.Series(filter)).all(axis=1)]
        if column_names is not None:
            df = df[column_names]
        return df

    def drop_table(self, table_name: str) -> None:
        """

        Args:
            table_name: name of table that should be deleted

        """
        self.connection.execute(f'drop table if exists  {table_name} ;')

    def delete_row_from_table(self, table_name: str, column_name_plus_value: dict) -> None:
        """

        Args:
            table_name: name of the table (str)
            column_name_plus_value: dictionary with column names (keys) and values of rows that should be dropped

        """
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
    """
    create a Database by current config
    :return:
    """
    return DB(config)

