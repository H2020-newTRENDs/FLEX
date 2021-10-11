import sqlite3
import pandas as pd
from A_Infrastructure.A1_CONS import CONS


class DB:

    def create_Connection(self, database_name):
        conn = sqlite3.connect(CONS().DatabasePath / database_name)
        return conn

    def read_DataFrame(self, table_name, conn, *column_names, **kwargs):
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

        DataFrame = pd.read_sql('select ' + columns2extract + ' from ' + table_name + condition, con=conn)
        return DataFrame

    def read_DataFrameRow(self, table_name, row_id, conn):
        DataFrame = self.read_DataFrame(table_name, conn)
        Series = DataFrame.iloc[row_id]
        return Series

    def write_DataFrame(self, table, table_name, column_names, conn, **kwargs):
        table_DataFrame = pd.DataFrame(table, columns=column_names)
        if "dtype" in kwargs:
            table_DataFrame.to_sql(table_name, conn, index=False,
                                   if_exists='replace', chunksize=1000, dtype=kwargs["dtype"])
        else:
            table_DataFrame.to_sql(table_name, conn, index=False, if_exists='replace', chunksize=1000)
        return None

    def add_to_dataframe(self, row, table_name, column_names, conn):

        pass

    def revise_DataType(self, table_name, data_type, conn):
        table_DataFrame = self.read_DataFrame(table_name, conn)
        table_DataFrame.to_sql(table_name, conn, index=False, if_exists='replace', chunksize=1000,
                               dtype=data_type)

    def copy_DataFrame(self, table_name_from, conn_from, table_name_to, conn_to):
        table = self.read_DataFrame(table_name_from, conn_from)
        table.to_sql(table_name_to, conn_to, index=False, if_exists='replace', chunksize=1000)
        return None
