from scipy.stats import norm
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Get the absolute path of the directory two levels up
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add this directory to sys.path
sys.path.insert(0, two_levels_up)
from models.operation.scenario import OperationScenario
from plotters.operation.Visualization_class import MotherVisualization
from utils.config import Config
from projects.main import get_config
from utils.db import fetch_input_tables


# -----------------------------------------------------------------------------------------------------------
class PlotlyVisualize(MotherVisualization):

    
    def show_yearly_comparison_of_SEMS_reference(self) -> None:
        """
        this function creates a plotly bar chart that instantly shows the differences in the yearly results. When bars
        have value = 1 they are the same in both modes, if they are both 0 they are not being used.
        """
        yearly_df = pd.concat(
            [self.yearly_results_optimization_df, self.yearly_results_reference_df],
            axis=0,
        )
        yearly_df.index = ["SEMS", "reference"]
        # normalize (min max normalization) the source so differences can be seen in the bar plot: min = Zero
        # columns not used will be nan and columns of equal value will be 1
        yearly_df_normalized = (yearly_df - 0) / (yearly_df.max() - 0)
        yearly_df_normalized = yearly_df_normalized.fillna(0)  # make nan to zero
        # melt the dataframe so it is easily usable for plotly
        yearly_df_plotly = pd.melt(
            yearly_df_normalized.T.reset_index(),
            id_vars="index",
            value_vars=["SEMS", "reference"],
        )
        # this plot is supposed to instantly show differences in input as well as output parameters:
        fig = px.bar(
            data_frame=yearly_df_plotly,
            x="index",
            y="value",
            color="variable",
            barmode="group",
        )
        fig.update_layout(title_text=self.create_header())
        fig.show()

    def hourly_comparison_SEMS_reference(self) -> None:
        reference_df = self.hourly_results_reference_df
        optimization_df = self.hourly_results_optimization_df
        # check if both tables have same columns
        assert sorted(list(reference_df.columns)) == sorted(
            list(optimization_df.columns)
        )
        # determine how many subplots are needed by excluding profiles that are zero in both modes
        for column_name in reference_df.columns:
            if (reference_df[column_name] == 0).all() and (
                    optimization_df[column_name] == 0
            ).all():
                reference_df = reference_df.drop(columns=[column_name])
                optimization_df = optimization_df.drop(columns=[column_name])
                continue
            # also exclude columns where all values are static (eg battery size) and the same:
            if (
                    reference_df[column_name].to_numpy()[0]
                    == reference_df[column_name].to_numpy()
            ).all() and (
                    optimization_df[column_name].to_numpy()[0]
                    == optimization_df[column_name].to_numpy()
            ).all():
                reference_df = reference_df.drop(columns=[column_name])
                optimization_df = optimization_df.drop(columns=[column_name])

        # count the columns which will be the number of subplots:
        column_number = len(list(reference_df.columns))
        # x-axis are the hours
        x_axis = np.arange(8760)
        # create plot
        fig = make_subplots(
            rows=column_number,
            cols=1,
            subplot_titles=sorted(list(reference_df.columns)),
            shared_xaxes=True,
        )
        for i, column_name in enumerate(sorted(list(reference_df.columns))):
            fig.add_trace(
                go.Scatter(x=x_axis, y=reference_df[column_name], name="reference"),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x_axis, y=optimization_df[column_name], name="SEMS"),
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            height=400 * column_number, width=1600, title_text=self.create_header()
        )
        fig.show()

    def investigate_resulting_load_profile(self):
        x_axis = np.arange(8760)
        reference_load = self.hourly_results_reference_df.Load.to_numpy() / 1000  # kW
        optimization_load = (
                self.hourly_results_optimization_df.Load.to_numpy() / 1000
        )  # kW
        try:
            electricity_price = (
                    self.hourly_results_reference_df.ElectricityPrice.to_numpy() * 1000
            )  # ct/kWh
        except:
            print("no energy price in hourly results saved")
            electricity_price = np.zeros(shape=reference_load.shape)

        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]]
        )
        fig.add_trace(
            go.Scatter(x=x_axis, y=reference_load, name="reference"), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=x_axis, y=optimization_load, name="SEMS"), secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=x_axis, y=electricity_price, name="electricity price"),
            secondary_y=True,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="hours")

        # Set y-axes titles
        fig.update_yaxes(title_text="total household load (kW)", secondary_y=False)
        fig.update_yaxes(title_text="electricity price (ct/kWh)", secondary_y=True)
        fig.update_layout(font={"size": 24})

        fig.show()

        # TODO make the figure nicer (quantilen erklären) and try to compare this results with a different household
        indices_price_10 = np.where(
            electricity_price < np.quantile(electricity_price, 0.1), True, False
        )
        indices_price_25 = np.where(
            electricity_price < np.quantile(electricity_price, 0.25), True, False
        )
        indices_price_75 = np.where(
            electricity_price > np.quantile(electricity_price, 0.75), True, False
        )
        indices_price_90 = np.where(
            electricity_price > np.quantile(electricity_price, 0.9), True, False
        )

        indices_price_below_50 = np.where(
            electricity_price < np.quantile(electricity_price, 0.5), True, False
        )
        indices_price_above_50 = np.where(
            electricity_price > np.quantile(electricity_price, 0.5), True, False
        )

        name_list = [
            "below 10%",
            "below 25%",
            "below 50%",
            "above 50%",
            "above 75%",
            "above 90%",
        ]
        load_dict_opt = {}
        load_dict_ref = {}
        for i, indices in enumerate(
                [
                    indices_price_10,
                    indices_price_25,
                    indices_price_below_50,
                    indices_price_above_50,
                    indices_price_75,
                    indices_price_90,
                ]
        ):
            ref_load_total = (indices * reference_load).sum()
            opt_load_total = (indices * optimization_load).sum()
            load_dict_ref[name_list[i]] = ref_load_total
            load_dict_opt[name_list[i]] = opt_load_total

        # show load difference in times where price signal is in certain quantile
        df_ref = pd.DataFrame(load_dict_ref, index=[0]).T
        df_opt = pd.DataFrame(load_dict_opt, index=[0]).T
        df = pd.concat([df_opt, df_ref], axis=1)

        df_loads = pd.concat(
            [pd.Series(reference_load), pd.Series(optimization_load)], axis=1
        ).set_index(electricity_price)
        df_loads.columns = ["Reference", "SEMS"]
        # df_loads = pd.melt(df_loads, id_vars="price", value_vars=["Reference", "SEMS"])
        # fig = px.histogram(data_frame=df_loads, x="price", color="variable")
        # fig.show()
        # create histograms that will be implemeted in subplots later:
        fig_z = ff.create_distplot(
            [electricity_price], group_labels=["electricity price"]
        )
        # fig_z.show()
        fig_ref = px.histogram(
            data_frame=df_loads, x=df_loads.index, y=df_loads["Reference"]
        )
        fig_sems = px.histogram(
            data_frame=df_loads, x=df_loads.index, y=df_loads["SEMS"]
        )

        # normal distribution curve of price source
        norm_distribution_elec_price = norm.pdf(
            electricity_price, np.mean(electricity_price), np.std(electricity_price)
        )
        norm_distribution_ref_load = norm.pdf(
            reference_load, np.mean(reference_load), np.std(reference_load)
        )
        norm_distribution_opt_load = norm.pdf(
            optimization_load, np.mean(optimization_load), np.std(optimization_load)
        )
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        # first subplot: the probability distribution of electric load over the electricity price
        fig.add_trace(
            go.Histogram(fig_sems["source"][0], opacity=0.3, name="SEMS load"),
            col=1,
            row=1,
        )
        fig.update_traces(col=1, row=1, marker_color="red")

        fig.add_trace(
            go.Histogram(fig_ref["source"][0], opacity=0.3, name="Reference load"),
            col=1,
            row=1,
        )
        fig.update_layout(barmode="overlay")

        # second subplot, the probability distribution of the electricity price:
        fig.add_trace(go.Histogram(fig_z["source"][0], autobinx=True), col=1, row=2)
        fig.add_trace(go.Scatter(fig_z["source"][1]), col=1, row=2)
        for percentage in [0.1, 0.25, 0.5, 0.75, 0.9]:
            fig.add_vline(
                x=np.quantile(electricity_price, percentage),
                line_dash="dash",
                col=1,
                row=1,
            )
            fig.add_vline(
                x=np.quantile(electricity_price, percentage),
                line_dash="dash",
                col=1,
                row=2,
            )
            fig.add_annotation(
                x=np.quantile(electricity_price, percentage),
                y=0,
                text="{:.0%}".format(percentage),
                col=1,
                row=2,
            )
        # x-axes
        fig.update_xaxes(title="electricity price (ct/kWh)", row=1, col=1)
        fig.update_xaxes(title="electricity price (ct/kWh)", row=2, col=1)
        # y-axes
        fig.update_yaxes(title="summed up electricity load (kWh)", row=1, col=1)
        fig.update_yaxes(title="probability", row=2, col=1)
        fig.update_layout(
            xaxis=dict(
                tickmode="linear", tick0=round(min(electricity_price)), dtick=0.5
            )
        )
        # save image to pdf or svg:
        image_name = "Electricity_price_and_Load_distribution.pdf"
        path_to_image_folder = (
                r"C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Refactor/project/PhilippTest/Figures/"
                + image_name
        )
        fig.write_image(path_to_image_folder)
        fig.show()

        # Distribution probability of load
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=reference_load,
                y=norm_distribution_ref_load,
                mode="markers",
                name="Reference load",
            ),
            col=1,
            row=1,
        )
        fig.add_trace(
            go.Scatter(
                x=optimization_load,
                y=norm_distribution_opt_load,
                mode="markers",
                name="SEMS load",
            ),
            col=1,
            row=1,
        )
        fig.update_xaxes(title="electricity load (kWh)", row=1, col=1)
        # y-axes
        fig.update_yaxes(title="probability", row=1, col=1)
        image_name = "Load_probability_distribution.pdf"
        # TODO make this path a variable that is dependent on the project
        path_to_image_folder = (
                r"C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Refactor/project/PhilippTest/Figures/"
                + image_name
        )
        fig.write_image(path_to_image_folder)
        fig.show()

def compare_variable_of_multiple_scenarios(var_name: str, cfg: Config, scenarios: list, input_tables, show_ref: bool, show_opt: bool):
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]], shared_xaxes=True)
    x_axis = np.arange(8760)
    colormap = plt.cm.get_cmap("tab10", len(scenarios))     
    color_map = {scen_id: f"rgba({int(colormap(i)[0]*255)}, {int(colormap(i)[1]*255)}, {int(colormap(i)[2]*255)}, 1)"
             for i, scen_id in enumerate(scenarios)}
    for scen_id in scenarios:
        scenario = OperationScenario(scenario_id=scen_id, config=cfg, input_tables=input_tables)

        (
            hourly_results_reference_df,
            yearly_results_reference_df,
            hourly_results_optimization_df,
            yearly_results_optimization_df,
        ) = MotherVisualization(scenario=scenario).fetch_results_for_specific_scenario_id()

        if show_ref:
            fig.add_trace(
                go.Scatter(x=x_axis, y=hourly_results_reference_df.loc[:, var_name], name=f"reference {scen_id}", line_dash="dash", line=dict(color=color_map[scen_id])), secondary_y=False
            )
        if show_opt:
            fig.add_trace(
                go.Scatter(x=x_axis, y=hourly_results_optimization_df.loc[:, var_name], name=f"optimized {scen_id}", line=dict(color=color_map[scen_id])), secondary_y=False
            )

        fig.add_trace(
                go.Scatter(x=x_axis, y=hourly_results_optimization_df.loc[:, "ElectricityPrice"], name=f"price {scen_id}", line=dict(color=color_map[scen_id]), line_dash="dot"), secondary_y=True
            )
        
    fig.update_yaxes(title="electricity prices", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title=var_name, secondary_y=False)
    fig.show()



if __name__ == "__main__":
    scenario_2_compare = [1, 2, 3, 4]
    cfg = get_config(project_name="electricity_price_check")
    input_tables = fetch_input_tables(config=cfg)

    compare_variable_of_multiple_scenarios(
        var_name="Q_RoomHeating", 
        cfg=cfg,
        scenarios=scenario_2_compare,
        input_tables=input_tables,
        show_opt=True,
        show_ref=True
        )

    # create scenario:
    scenario_id = 1  
    scenario = OperationScenario(scenario_id=scenario_id, config=cfg, input_tables=input_tables)
    # plotly_visualization = PlotlyVisualize(scenario=scenario)
    # plotly_visualization.show_yearly_comparison_of_SEMS_reference()
    # plotly_visualization.hourly_comparison_SEMS_reference()
    # plotly_visualization.investigate_resulting_load_profile()
    # ---------------------------------------------------------------------------------------------------------
