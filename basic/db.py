import sqlalchemy
import pandas as pd

import basic.config as config


class DB:

    def __init__(self,
                 connection: sqlalchemy.engine.Engine = config.root_connection):
        """the default connection will be the root connection, for a different connection,
        the "create_connection function from the config file can be used"""
        self.connection = connection

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
                        data_types: dict,
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
