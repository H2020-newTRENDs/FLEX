import os
import sqlalchemy
import pandas as pd


class DB:

    def __init__(self,
                 db_name: str,
                 db_path: str):

        self.db_name = db_name
        self.db_path = db_path
        self.connection = self.create_connection(self.db_name)

    def get_engine(self):
        return self.connection

    def create_connection(self, db_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(f'sqlite:///{os.path.join(self.db_path, db_name + ".sqlite")}')

    def close(self):
        self.connection.dispose()

    def clear_database(self):
        for table_name in self.connection.table_names():
            self.connection.execute(f"drop table {table_name}")

    def write_dataframe(self,
                        table_name: str,
                        data_frame: pd.DataFrame,
                        data_types,
                        if_exists='append'): # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(table_name, self.connection, index=False, dtype=data_types, if_exists=if_exists)

    def read_dataframe(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql(f'select * from {table_name}', self.connection)

    def drop_table(self, table_name: str):
        self.connection.execute(f'drop table if exists  {table_name} ;')

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.connection)

