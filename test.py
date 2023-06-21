from typing import List
from config import cfg
from flex.config import Config
from flex.db import create_db_conn
from flex_operation.constants import OperationTable, OperationResultVar
import pandas as pd


class MonthConverter:

    def __init__(self, config: "Config"):
        self.config = config
        self.db = create_db_conn(config)
        self.df_opt = self.db.read_dataframe(OperationTable.ResultOptHour)
        self.df_ref = self.db.read_dataframe(OperationTable.ResultRefHour)

    @staticmethod
    def convert_h2m(id_scenario: int, df: pd.DataFrame) -> List[dict]:
        month_results = []
        hours_per_month = {
            1: (1, 744),
            2: (745, 1416),
            3: (1417, 2160),
            4: (2161, 2880),
            5: (2881, 3624),
            6: (3625, 4344),
            7: (4345, 5088),
            8: (5089, 5832),
            9: (5833, 6552),
            10: (6553, 7296),
            11: (7297, 8016),
            12: (8017, 8760)
        }
        for month, hour_range in hours_per_month.items():
            month_df = df[(df['Hour'] >= hour_range[0]) & (df['Hour'] <= hour_range[1])]
            month_df = month_df.drop(['ID_Scenario', 'Hour', 'DayHour'], axis=1)
            month_sum = month_df.sum().to_dict()
            month_sum["Month"] = month
            month_sum["ID_Scenario"] = id_scenario
            month_results.append(month_sum)
        return month_results

    @staticmethod
    def drop_vars(df: pd.DataFrame):
        for variable_name, variable_type in OperationResultVar.__dict__.items():
            if variable_type == "hour":
                df = df.drop([variable_name], axis=1)
        return df

    def process_tables(self):
        opt_month = []
        ref_month = []
        for id_scenario in range(1, 4321):
            print(f'id_scenario --> {id_scenario}')
            df_opt = self.df_opt.loc[self.df_opt["ID_Scenario"] == id_scenario]
            df_opt = self.drop_vars(df_opt)
            opt_month += self.convert_h2m(id_scenario, df_opt)

            df_ref = self.df_ref.loc[self.df_ref["ID_Scenario"] == id_scenario]
            df_ref = self.drop_vars(df_ref)
            ref_month += self.convert_h2m(id_scenario, df_ref)
        self.db.write_dataframe(table_name=OperationTable.ResultOptMonth, data_frame=pd.DataFrame(opt_month))
        self.db.write_dataframe(table_name=OperationTable.ResultRefMonth, data_frame=pd.DataFrame(ref_month))


if __name__ == "__main__":
    mc = MonthConverter(cfg)
    mc.process_tables()

