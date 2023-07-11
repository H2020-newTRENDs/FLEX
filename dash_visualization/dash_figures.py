import dash
import pandas as pd
import plotly.graph_objs
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import dash_visualization.ids as ids
import dash_visualization.buttons as buttons
from typing import List
from flex_operation.Country_Visualization import CountryResultPlots
from flex_operation.plotly_single_household import PlotlyVisualize
from flex.config import Config
from dash_visualization import data_extraction, main
from flex.db import DB
from flex_operation.constants import OperationTable, OperationScenarioComponent
from flex_operation.scenario import OperationScenario, MotherOperationScenario


def EU27_p2p(app: Dash) -> html.Div:
    @app.callback(Output(ids.EU27_P2P_CHART, "children"),
                  Input(ids.ALL_COUNTRIES_DROP_DOWN, "value"),
                  Input(ids.ALL_YEARS_DROP_DOWN, "value"))
    def update_Eu27_p2p(country_list: List[str], years: List[str]):
        year_list = [int(year) for year in years]
        fig = CountryResultPlots(countries=country_list, years=year_list, project_prefix=main.PROJECT_PREFIX).plotly_EU27_p2p()
        return html.Div(dcc.Graph(figure=fig), id=ids.EU27_P2P_CHART)

    return html.Div(id=ids.EU27_P2P_CHART)


def EU27_load_factor(app: Dash) -> html.Div:
    @app.callback(Output(ids.EU27_LOAD_FACTOR_CHART, "children"),
                  Input(ids.ALL_COUNTRIES_DROP_DOWN, "value"),
                  Input(ids.ALL_YEARS_DROP_DOWN, "value"))
    def update_Eu27_load_factor(country_list: List[str], years: List[str]):
        year_list = [int(year) for year in years]
        fig = CountryResultPlots(countries=country_list, years=year_list, project_prefix=main.PROJECT_PREFIX).plotly_EU27_load_factor()
        return html.Div(dcc.Graph(figure=fig), id=ids.EU27_LOAD_FACTOR_CHART)

    return html.Div(id=ids.EU27_LOAD_FACTOR_CHART)


def EU27_pv_self_consumption(app: Dash) -> html.Div:
    @app.callback(Output(ids.EU27_PV_SELF_CONSUMPTION, "children"),
                  Input(ids.ALL_COUNTRIES_DROP_DOWN, "value"),
                  Input(ids.ALL_YEARS_DROP_DOWN, "value"))
    def update_Eu27_load_factor(countries: List[str], years: List[str]):
        year_list = [int(year) for year in years]
        fig = CountryResultPlots(countries=countries, years=year_list, project_prefix=main.PROJECT_PREFIX).plotly_EU27_self_consumption()
        return html.Div(dcc.Graph(figure=fig), id=ids.EU27_PV_SELF_CONSUMPTION)

    return html.Div(id=ids.EU27_PV_SELF_CONSUMPTION)


def EU27_shifted_electricity(app: Dash) -> html.Div:
    @app.callback(Output(ids.EU27_SHIFTED_ELECTRICITY, "children"),
                  Input(ids.ALL_COUNTRIES_DROP_DOWN, "value"),
                  Input(ids.ALL_YEARS_DROP_DOWN, "value"))
    def update_Eu27_load_factor(countries: List[str], years: List[str]):
        year_list = [int(year) for year in years]
        fig = CountryResultPlots(countries=countries, years=year_list, project_prefix=main.PROJECT_PREFIX).plotly_EU27_shifted_electricity()
        return html.Div(dcc.Graph(figure=fig), id=ids.EU27_SHIFTED_ELECTRICITY)

    return html.Div(id=ids.EU27_SHIFTED_ELECTRICITY)


def var_to_list(item) -> list:
    if type(item) == list:
        return item
    else:
        return [item]


def hourly_profiles(app: Dash) -> html.Div:
    # the hourly profiles are loaded by the PlotlyVisualize cass which takes the OperationScenario and the config
    @app.callback(Output(ids.HOURLY_PROFILES_CHART, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"),
                  Input(ids.BATTERY_DROP_DOWN, "value"),
                  Input(ids.BUILDING_DROP_DOWN, "value"),
                  Input(ids.BOILER_DROP_DOWN, "value"),
                  Input(ids.ENERGY_PRICE_DROP_DOWN, "value"),
                  Input(ids.HEATING_ELEMENT_DROP_DOWN, "value"),
                  Input(ids.HOT_WATER_TANK_DROP_DOWN, "value"),
                  Input(ids.PV_DROP_DOWN, "value"),
                  Input(ids.SPACE_COOLING_DROP_DOWN, "value"),
                  Input(ids.SPACE_HEATING_TANK_DROP_DOWN, "value"),
                  Input(ids.VEHICLE_DROP_DOWN, "value"),
                  Input(ids.HOURLY_COLUMN_CHECK_LIST, "value"))
    def create_figure(country: str,
                      year: int,
                      battery_size: list,
                      building_ids: list,
                      boiler: list,
                      energy_prices: list,
                      heating_element: list,
                      hot_water_tank: list,
                      pv: list,
                      space_cooling: list,
                      space_heating_tank: list,
                      vehicle: list,
                      plot_values: list):
        selection_table = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )

        selected_df = selection_table.query(f"ID_Battery in {var_to_list(battery_size)} & "
                                            f"ID_Building in {var_to_list(building_ids)} &"
                                            f"ID_Boiler in {var_to_list(boiler)} &"
                                            f"ID_EnergyPrice in {var_to_list(energy_prices)} &"
                                            f"ID_HeatingElement in {var_to_list(heating_element)} &"
                                            f"ID_HotWaterTank in {var_to_list(hot_water_tank)} &"
                                            f"ID_PV in {var_to_list(pv)} &"
                                            f"ID_SpaceCoolingTechnology in {var_to_list(space_cooling)} &"
                                            f"ID_SpaceHeatingTank in {var_to_list(space_heating_tank)} &"
                                            f"ID_Vehicle in {var_to_list(vehicle)}")
        ids_to_plot = list(selected_df["ID_Scenario"])
        # grab the hourly load profiles for these ids:
        project_name = f"{main.PROJECT_PREFIX}_{country}_{year}"
        db = DB(config=Config(project_name))
        tables = []
        for scen in ids_to_plot:
            ref_table = db.read_parquet(table_name=OperationTable.ResultRefHour,
                                        scenario_ID=scen,
                                        column_names=["ID_Scenario", "Hour"] + plot_values)
            ref_table["mode"] = f"reference"
            opt_table = db.read_parquet(table_name=OperationTable.ResultOptHour,
                                        scenario_ID=scen,
                                        column_names=["ID_Scenario", "Hour"] + plot_values)
            opt_table["mode"] = f"optimized"
            tables.append(ref_table)
            tables.append(opt_table)
        large_df = pd.concat(tables, axis=0)
        long_df = large_df.melt(id_vars=["mode", "ID_Scenario", "Hour"])

        fig = px.line(
            data_frame=long_df,
            x="Hour",
            y="value",
            facet_row="variable",
            color="ID_Scenario",
            line_dash="mode",
            height=300 * len(plot_values)
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("variable=", "")))
        return html.Div(dcc.Graph(figure=fig), id=ids.HOURLY_PROFILES_CHART)

    return html.Div(id=ids.HOURLY_PROFILES_CHART)


def number_of_buildings(app: Dash) -> html.Div:
    @app.callback(Output(ids.NUMBER_OF_BUILDINGS_CHART, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"),
                  Input(ids.YEAR_BUILDING_NUMBER_DROP_DOWN, "value"),
                  Input(ids.BATTERY_BUTTON, "n_clicks", ),
                  Input(ids.BOILER_BUTTON, "n_clicks"),
                  Input(ids.HEATING_ELEMENT_BUTTON, "n_clicks"),
                  Input(ids.HOT_WATER_TANK_BUTTON, "n_clicks"),
                  Input(ids.PV_BUTTON, "n_clicks"),
                  Input(ids.SPACE_COOLING_BUTTON, "n_clicks"),
                  Input(ids.SPACE_HEATING_TANK_BUTTON, "n_clicks"),
                  Input(ids.VEHICLE_BUTTON, "n_clicks"),
                  )
    def button_update(country: str, year: int, battery, boiler, heating_element, hot_water_tank, pv, space_cooling,
                      space_heating_tank, vehicle):
        # Get the button that was clicked last
        button_clicked = dash.callback_context.triggered_id

        df = data_extraction.grab_scneario_table_with_numbers(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
        long_table = df.melt(id_vars=["ID_Building", "number"], value_vars=button_clicked)
        # transform the "value" column into strings so it is correctly showed in the plot as category
        long_table["value"] = long_table["value"].astype(str)
        # rename the column:
        long_table = long_table

        # figure
        fig = px.bar(
            data_frame=long_table,
            x="ID_Building",
            y="number",
            color="value",
            barmode="group"
        )
        return html.Div(dcc.Graph(figure=fig), id=ids.NUMBER_OF_BUILDINGS_CHART, )

    return html.Div(id=ids.NUMBER_OF_BUILDINGS_CHART)


def building_age_distribution(app: Dash) -> html.Div:
    @app.callback(Output(ids.BUILDING_AGE_DISTRIBUTION_CHART, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"),
                  )
    def update_figure(country: str) -> html.Div:
        large_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                age_table = data_extraction.load_building_age_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                age_table["year"] = year
                large_df = pd.concat([large_df, age_table], axis=0)
            except:
                print(f"building age table for {country} {year} could not be loaded!")
        fig = px.histogram(
            data_frame=large_df,
            x="construction_period",
            y="number",
            color="year",
            barmode="group"  # stack, group, overlay, relative
        )
        fig.update_layout(yaxis_title="number of buildings with heat pumps",
                          xaxis_title="construction period of the buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.BUILDING_AGE_DISTRIBUTION_CHART)

    return html.Div(id=ids.BUILDING_AGE_DISTRIBUTION_CHART)


def share_MFH_SFH(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_MFH_SFH, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                building_table = data_extraction.load_building_age_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = building_table.groupby("type")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"building age table for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="type",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_MFH_SFH)

    return html.Div(id=ids.SHARE_MFH_SFH)


def share_battery(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_BATTERY, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:

                battery_table = data_extraction.create_scenario_selection_table_with_numbers(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = battery_table.groupby("ID_Battery")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["ID_Battery"] = plot_df["ID_Battery"] / 1_000  # Wh to kWh
                plot_df["Battery capacity (kWh)"] = plot_df["ID_Battery"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"battery table for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="Battery capacity (kWh)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_BATTERY)

    return html.Div(id=ids.SHARE_BATTERY)


def share_dhw(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_DHW_STORAGE, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                dhw_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = dhw_table.groupby("ID_HotWaterTank")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["DHW tank size (l)"] = plot_df["ID_HotWaterTank"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="DHW tank size (l)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_DHW_STORAGE)

    return html.Div(id=ids.SHARE_DHW_STORAGE)


def share_heating_tank(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_HEATING_STORAGE, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                dhw_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = dhw_table.groupby("ID_SpaceHeatingTank")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["Space heating tank size (l)"] = plot_df["ID_SpaceHeatingTank"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="Space heating tank size (l)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_HEATING_STORAGE)

    return html.Div(id=ids.SHARE_HEATING_STORAGE)


def share_pv(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_PV, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                pv_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = pv_table.groupby("ID_PV")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["PV peak power (kWp)"] = plot_df["ID_PV"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="PV peak power (kWp)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_PV)

    return html.Div(id=ids.SHARE_PV)


def share_boiler(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_BOILER, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                dhw_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = dhw_table.groupby("ID_Boiler")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["Boiler type"] = plot_df["ID_Boiler"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="Boiler type",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_BOILER)

    return html.Div(id=ids.SHARE_BOILER)


def share_heating_element(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_HEATING_ELEMENT, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                dhw_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = dhw_table.groupby("ID_HeatingElement")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["ID_HeatingElement"] = plot_df["ID_HeatingElement"] / 1_000  # W in kW
                plot_df["Heating element power (kW)"] = plot_df["ID_HeatingElement"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="Heating element power (kW)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_HEATING_ELEMENT)

    return html.Div(id=ids.SHARE_HEATING_ELEMENT)


def share_space_cooling(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_SPACE_COOLING, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                dhw_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = dhw_table.groupby("ID_SpaceCoolingTechnology")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["ID_SpaceCoolingTechnology"] = plot_df["ID_SpaceCoolingTechnology"] / 1_000  # W in kW
                plot_df["AC power (kW)"] = plot_df["ID_SpaceCoolingTechnology"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="AC power (kW)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_SPACE_COOLING)

    return html.Div(id=ids.SHARE_SPACE_COOLING)


def share_vehicle(app: Dash) -> html.Div:
    @app.callback(Output(ids.SHARE_VEHICLE, "children"),
                  Input(ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN, "value"))
    def update_figure(country: str):
        big_df = pd.DataFrame()
        for year in main.YEARS:
            try:
                dhw_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                plot_df = dhw_table.groupby("ID_Vehicle")["number"].sum().reset_index()
                plot_df["year"] = str(year)
                plot_df["ID_Vehicle"] = plot_df["ID_Vehicle"] / 1_000  # Wh in kWh
                plot_df["E-car capacity (kWh)"] = plot_df["ID_Vehicle"].astype(str)  # convert to string
                big_df = pd.concat([big_df, plot_df], axis=0)
            except:
                print(f"scenario selection table with numbers for {country} {year} could not be loaded!")

        fig = px.bar(
            data_frame=big_df,
            x="year",
            y="number",
            color="E-car capacity (kWh)",
            barmode="relative"
        )
        fig.update_layout(yaxis_title="number of buildings")
        return html.Div(dcc.Graph(figure=fig), id=ids.SHARE_VEHICLE)

    return html.Div(id=ids.SHARE_VEHICLE)


def percentage_share_of_PV_all_countries():
    big_df = pd.DataFrame()
    for country in main.COUNTRIES:
        for year in main.YEARS:
            try:
                pv_table = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )
                # calculate the share of PV
                pv_share = pv_table.query("ID_PV != 0")["number"].sum() / pv_table["number"].sum() * 100
                big_df.loc[country, year] = pv_share
            except:
                print(f"PV data is not available for {country} {year}")


if __name__ == "__main__":
    percentage_share_of_PV_all_countries()
