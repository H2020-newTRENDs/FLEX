from typing import TYPE_CHECKING
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from basics.db import create_db_conn

if TYPE_CHECKING:
    from basics.config import Config
    from models.operation.scenario import OperationScenario


class Analyzer:

    def __init__(self, scenario: 'OperationScenario', config: 'Config'):
        self.scenario = scenario
        self.db = create_db_conn(config)
        self.read_result_dfs()

    def read_result_dfs(self):
        scenario_id = self.scenario.scenario_id
        self.opt_hour = self.db.read_dataframe("Result_OptimizationHour", filter={"ID_Scenario": scenario_id})
        self.opt_year = self.db.read_dataframe("Result_OptimizationYear", filter={"ID_Scenario": scenario_id})
        self.ref_hour = self.db.read_dataframe("Result_ReferenceHour", filter={"ID_Scenario": scenario_id})
        self.ref_year = self.db.read_dataframe("Result_ReferenceYear", filter={"ID_Scenario": scenario_id})

    def hourly_comparison_SEMS_reference(self) -> None:
        reference_df = self.ref_hour
        optimization_df = self.opt_hour
        # check if both tables have same columns
        assert sorted(list(reference_df.columns)) == sorted(list(optimization_df.columns))
        # determine how many subplots are needed by excluding profiles that are zero in both modes
        for column_name in reference_df.columns:
            if (reference_df[column_name] == 0).all() and (optimization_df[column_name] == 0).all():
                reference_df = reference_df.drop(columns=[column_name])
                optimization_df = optimization_df.drop(columns=[column_name])
                continue
            # also exclude columns where all values are static (eg battery size) and the same:
            if (reference_df[column_name].to_numpy()[0] == reference_df[column_name].to_numpy()).all() and \
                    (optimization_df[column_name].to_numpy()[0] == optimization_df[column_name].to_numpy()).all():
                reference_df = reference_df.drop(columns=[column_name])
                optimization_df = optimization_df.drop(columns=[column_name])

        # count the columns which will be the number of subplots:
        column_number = len(list(reference_df.columns))
        # x-axis are the hours
        x_axis = np.arange(8760)
        # create plot
        fig = make_subplots(rows=column_number, cols=1,
                            subplot_titles=sorted(list(reference_df.columns)),
                            shared_xaxes=True)
        for i, column_name in enumerate(sorted(list(reference_df.columns))):
            fig.add_trace(go.Scatter(x=x_axis, y=reference_df[column_name], name="reference"), row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=x_axis, y=optimization_df[column_name], name="SEMS"), row=i + 1, col=1)

        fig.update_layout(height=400 * column_number, width=1600)
        fig.show()

    def run(self):
        self.hourly_comparison_SEMS_reference()
