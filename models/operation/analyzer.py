from typing import TYPE_CHECKING
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from basics.db import create_db_conn
from basics import kit
from models.operation.enums import TableEnum

if TYPE_CHECKING:
    from basics.config import Config


logger = kit.get_logger(__name__)


class Analyzer:

    def __init__(self, config: 'Config'):
        self.db = create_db_conn(config)
        self.opt_hour_df = None
        self.opt_year_df = None
        self.ref_hour_df = None
        self.ref_year_df = None

    @property
    def opt_hour(self):
        if self.opt_hour_df is None:
            self.opt_hour_df = self.db.read_dataframe(TableEnum.ResultOptHour.value)
        return self.opt_hour_df

    @property
    def opt_year(self):
        if self.opt_year_df is None:
            self.opt_year_df = self.db.read_dataframe(TableEnum.ResultOptYear.value)
        return self.opt_year_df

    @property
    def ref_hour(self):
        if self.ref_hour_df is None:
            self.ref_hour_df = self.db.read_dataframe(TableEnum.ResultRefHour.value)
        return self.ref_hour_df

    @property
    def ref_year(self):
        if self.ref_year_df is None:
            self.ref_year_df = self.db.read_dataframe(TableEnum.ResultRefYear.value)
        return self.ref_year_df

    def compare_opt(self, id1, id2) -> None:
        df1 = kit.filter_df(self.opt_hour, filter_dict={"ID_Scenario": id1})
        df2 = kit.filter_df(self.opt_hour, filter_dict={"ID_Scenario": id2})
        name1 = f'opt_{id1}'
        name2 = f'opt_{id2}'
        self.gen_html(df1, df2, name1, name2)

    def compare_ref(self, id1, id2) -> None:
        df1 = kit.filter_df(self.ref_hour, filter_dict={"ID_Scenario": id1})
        df2 = kit.filter_df(self.ref_hour, filter_dict={"ID_Scenario": id2})
        name1 = f'ref_{id1}'
        name2 = f'ref_{id2}'
        self.gen_html(df1, df2, name1, name2)

    def compare_opt_ref(self, scenario_id: int) -> None:
        opt_df = kit.filter_df(self.opt_hour, filter_dict={"ID_Scenario": scenario_id})
        ref_df = kit.filter_df(self.ref_hour, filter_dict={"ID_Scenario": scenario_id})
        name1 = f'opt_{scenario_id}'
        name2 = f'ref_{scenario_id}'
        self.gen_html(opt_df, ref_df, name1, name2)

    @staticmethod
    def gen_html(df1, df2, name1: str, name2: str):
        assert sorted(list(df1.columns)) == sorted(list(df2.columns))
        # determine how many subplots are needed by excluding profiles that are zero in both modes
        for column_name in df1.columns:
            if (df1[column_name] == 0).all() and (df2[column_name] == 0).all():
                df1 = df1.drop(columns=[column_name])
                df2 = df2.drop(columns=[column_name])
                continue
        # count the columns which will be the number of subplots:
        subplots_number = len(list(df1.columns))
        fig = make_subplots(rows=subplots_number, cols=1,
                            subplot_titles=sorted(list(df1.columns)),
                            shared_xaxes=True)
        for i, column_name in enumerate(sorted(list(df1.columns))):
            fig.add_trace(go.Scatter(x=np.arange(8760), y=df1[column_name], name=name1), row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(8760), y=df2[column_name], name=name2), row=i + 1, col=1)
        fig.update_layout(height=400 * subplots_number, width=1600)
        fig.show()

