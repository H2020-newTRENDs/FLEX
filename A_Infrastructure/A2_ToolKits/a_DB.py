
import sqlite3
import pandas as pd
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A1_Config.b_Register import REG

class DB:

    """
    As shown below, all the functions (except create_Connection) demands a connection to the target database.
    This is because, in the later stage, we may have multiple databases, for example, one as ROOT database,
    the others as SCENARIO databases.
    """

    def create_Connection(self, database_name):
        conn = sqlite3.connect(CONS().DatabasePath + database_name + ".sqlite")
        return conn

    def read_DataFrame(self, table_name, conn, **kwargs):
        """
        Use this function instead of read_ExoTableValue when:
        you want to read the whole table or multiple rows from it.

        If the table is huge, it takes long time to load it to memory.
        So, it is good to use **kwargs, for example, you add ID_Country=5 when using this function.
        Besides, you should also read enough rows one time and do not too many times.
        Because it is much faster to process the selection in DataFrame in memory.
        So, there is a trade-off in "how many parameters to put in **kwargs".
        """
        if len(kwargs) > 0:
            condition_temp = " where "
            for key, value in kwargs.items():
                condition_temp = condition_temp + key + " == '" + str(value) + "' and "
            condition = condition_temp[0:-5]
            DataFrame = pd.read_sql('select * from ' + table_name + condition, con=conn)
        else:
            DataFrame = pd.read_sql('select * from ' + table_name, con=conn)
        return DataFrame

    def read_ExoTableValue(self, table_name, conn, **kwargs):
        """
        Use this function instead of read_DataFrame when:
        you are sure that there is only one row that satisfies the criteria in **kwargs,
        and you only need ONE VALUE from that row.
        """
        DataFrame = self.read_DataFrame(table_name, conn, **kwargs)
        Value = DataFrame.iloc[0]["Value"]
        return Value

    def read_GlobalParameterValue(self, parameter_name, conn):
        ParameterValue = self.read_ExoTableValue(REG().Exo_GlobalParameterValue, conn, Parameter=parameter_name)
        return ParameterValue

    def write_DataFrame(self, table, table_name, column_names, conn):
        table_DataFrame = pd.DataFrame(table, columns=column_names)
        table_DataFrame.to_sql(table_name, conn, index=False, if_exists='replace', chunksize=1000)
        return None

    def copy_DataFrame(self, table_name_from, conn_from, table_name_to, conn_to):
        table = self.read_DataFrame(table_name_from, conn_from)
        table.to_sql(table_name_to, conn_to, index=False, if_exists='replace', chunksize=1000)
        return None




