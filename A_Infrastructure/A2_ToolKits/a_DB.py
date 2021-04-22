
import sqlite3
import pandas as pd
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A1_Config.b_Register import REG

class DB:

    def create_Connection(self, database_name):
        conn = sqlite3.connect(CONS().DatabasePath + database_name + ".sqlite")
        return conn

    def read_DataFrame(self, table_name, conn, **kwargs):
        if len(kwargs) > 0:
            condition_temp = " where "
            for key, value in kwargs.items():
                condition_temp = condition_temp + key + " == '" + str(value) + "' and "
            condition = condition_temp[0:-5]
            DataFrame = pd.read_sql('select * from ' + table_name + condition, con=conn)
        else:
            DataFrame = pd.read_sql('select * from ' + table_name, con=conn)
        return DataFrame

    def read_GlobalParameterValue(self, parameter_name):
        Conn = sqlite3.connect(CONS().DatabasePath + CONS().RootDB + ".sqlite")
        ParameterValueTable = self.read_DataFrame(REG().Exo_GlobalParameterValue, Conn, Parameter=parameter_name)
        ParameterValue = ParameterValueTable.iloc[0]["Value"]
        return ParameterValue

    # def read_ExoTableValue(self, table_name, ):

    def write_DataFrame(self, table, table_name, column_names, conn):
        table_DataFrame = pd.DataFrame(table, columns=column_names)
        table_DataFrame.to_sql(table_name, conn, index=False, if_exists='replace', chunksize=1000)
        return None

    def copy_DataFrame(self, table_name_from, conn_from, table_name_to, conn_to):
        table = self.read_DataFrame(table_name_from, conn_from)
        table.to_sql(table_name_to, conn_to, index=False, if_exists='replace', chunksize=1000)
        return None




