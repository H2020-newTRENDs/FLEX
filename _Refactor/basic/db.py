import sqlalchemy
import pandas as pd

import _Refactor.basic.config as config


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

    def clear_database(self):
        for table_name in self.connection.table_names():
            self.connection.execute(f"drop table {table_name}")

    def write_dataframe(self,
                        table_name: str,
                        data_frame: pd.DataFrame,
                        data_types: dict,
                        if_exists='append'):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(table_name, self.connection, index=False, dtype=data_types, if_exists=if_exists,
                          chunksize=10_000)

    def read_dataframe(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql(f'select * from {table_name}', self.connection)

    def drop_table(self, table_name: str):
        self.connection.execute(f'drop table if exists  {table_name} ;')

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.connection)

