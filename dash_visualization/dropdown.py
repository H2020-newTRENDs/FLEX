from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from typing import List
from flex_operation.constants import OperationResultVar
from dash_visualization import ids, buttons, data_extraction, main



def all_countries_dropdown(app: Dash) -> html.Div:
    all_nations = main.COUNTRIES

    # Button to select all countries at once
    @app.callback(Output(ids.ALL_COUNTRIES_DROP_DOWN, "value"), Input(ids.SELECT_ALL_COUNTRIES_BUTTON, "n_clicks"))
    def select_all_countries(_: int) -> List[str]:  # the number of clicks is the input
        return all_nations

    return html.Div(
        children=[
            html.H6("Country"),
            dcc.Dropdown(
                id=ids.ALL_COUNTRIES_DROP_DOWN,
                options=[{"label": country, "value": country} for country in all_nations],
                value="AUT",
                multi=True,
            ),
            html.Button(
                className="country-dropdown-button",
                children=["Select All"],
                id=ids.SELECT_ALL_COUNTRIES_BUTTON,
            ),
        ]
    )


def all_years_dropdown(app: Dash) -> html.Div:
    # Button to select all years at once
    @app.callback(Output(ids.ALL_YEARS_DROP_DOWN, "value"), Input(ids.SELECT_ALL_YEARS_BUTTON, "n_clicks"))
    def select_all_years(_: int) -> List[str]:
        return main.YEARS

    return html.Div(
        children=[
            html.H6("Years"),
            dcc.Dropdown(
                id=ids.ALL_YEARS_DROP_DOWN,
                options=[{"label": year, "value": year} for year in main.YEARS],
                value="2020",
                multi=True,
            ),
            html.Button(
                className="year-dropdown-button",
                children=["Select All"],
                id=ids.SELECT_ALL_YEARS_BUTTON,
            ),
        ]
    )


def country_drop_down(app: Dash) -> html.Div:
    return html.Div(
        children=[
            html.H6("Country"),
            dcc.Dropdown(
                id=ids.COUNTRY_DROP_DOWN,
                options=[{"label": country, "value": country} for country in main.COUNTRIES],
                value="AUT",
                multi=False,
            ),
        ]
    )


def year_drop_down(app: Dash) -> html.Div:
    return html.Div(
        children=[
            html.H6("Years"),
            dcc.Dropdown(
                id=ids.YEAR_DROP_DOWN,
                options=[{"label": year, "value": year} for year in main.YEARS],
                value="2020",
                multi=False,
            ),
        ]
    )


def year_dropdown_for_building_numbers(app: Dash) -> html.Div:  # currently not used
    return html.Div(
        children=[
            html.H6("Years"),
            dcc.Dropdown(
                id=ids.YEAR_BUILDING_NUMBER_DROP_DOWN,
                options=[{"label": year, "value": year} for year in main.YEARS],
                value="2020",
                multi=False,
            ),
        ]
    )


def country_drop_down_for_building_numbers(app: Dash) -> html.Div:
    return html.Div(
        children=[
            html.H6("Country"),
            dcc.Dropdown(
                id=ids.COUNTRY_BUILDING_NUMBER_DROP_DOWN,
                options=[{"label": country, "value": country} for country in main.COUNTRIES],
                value="AUT",
                multi=False,
            ),
        ]
    )


def battery_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.BATTERY_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_battery_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_Battery"]
        return html.Div(
            children=[
                html.H6("Battery size"),
                dcc.Dropdown(
                    id=ids.BATTERY_DROP_DOWN,
                    options=[{"label": f"{round(size / 1_000, 1)} kWh", "value": size} for size in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.BATTERY_DROP_DOWN)


def building_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.BUILDING_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_Building"]
        return html.Div(
            children=[
                html.H6("Building ID"),
                dcc.Dropdown(
                    id=ids.BUILDING_DROP_DOWN,
                    options=[{"label": f"Building {ID}", "value": ID} for ID in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.BUILDING_DROP_DOWN)


def boiler_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.BOILER_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
         )["ID_Boiler"]
        return html.Div(
            children=[
                html.H6("Boiler"),
                dcc.Dropdown(
                    id=ids.BOILER_DROP_DOWN,
                    options=[{"label": ID, "value": ID} for ID in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.BOILER_DROP_DOWN)


def energy_price_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.ENERGY_PRICE_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_EnergyPrice"]
        return html.Div(
            children=[
                html.H6("Energy Price"),
                dcc.Dropdown(
                    id=ids.ENERGY_PRICE_DROP_DOWN,
                    options=[{"label": f"Price {ID}", "value": ID} for ID in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.ENERGY_PRICE_DROP_DOWN)


def heating_element_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.HEATING_ELEMENT_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_HeatingElement"]
        return html.Div(
            children=[
                html.H6("Heating Element power"),
                dcc.Dropdown(
                    id=ids.HEATING_ELEMENT_DROP_DOWN,
                    options=[{"label": f"{round(power / 1_000, 1)} kWh", "value": power} for power in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.HEATING_ELEMENT_DROP_DOWN)


def hot_water_tank_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.HOT_WATER_TANK_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_HotWaterTank"]
        return html.Div(
            children=[
                html.H6("Hot Water Tank"),
                dcc.Dropdown(
                    id=ids.HOT_WATER_TANK_DROP_DOWN,
                    options=[{"label": f"{size} l", "value": size} for size in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.HOT_WATER_TANK_DROP_DOWN)


def pv_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.PV_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_PV"]
        return html.Div(
            children=[
                html.H6("PV size"),
                dcc.Dropdown(
                    id=ids.PV_DROP_DOWN,
                    options=[{"label": f"{size} kW peak", "value": size} for size in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.PV_DROP_DOWN)


def space_cooling_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.SPACE_COOLING_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_SpaceCoolingTechnology"]
        return html.Div(
            children=[
                html.H6("AC power"),
                dcc.Dropdown(
                    id=ids.SPACE_COOLING_DROP_DOWN,
                    options=[{"label": f"{round(power / 1_000, 1)} kW", "value": power} for power in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.SPACE_COOLING_DROP_DOWN)


def space_heating_tank_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.SPACE_HEATING_TANK_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_SpaceHeatingTank"]
        return html.Div(
            children=[
                html.H6("Space heating tank size"),
                dcc.Dropdown(
                    id=ids.SPACE_HEATING_TANK_DROP_DOWN,
                    options=[{"label": f"{size} l", "value": size} for size in values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.SPACE_HEATING_TANK_DROP_DOWN)


def vehicle_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.VEHICLE_DROP_DOWN, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = data_extraction.create_scenario_selection_table(
            project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
        )["ID_Vehicle"]
        return html.Div(
            children=[
                html.H6("Vehicle"),
                dcc.Dropdown(
                    id=ids.VEHICLE_DROP_DOWN,
                    options=[{"label": f"{round(capacity / 1_000, 1)} kWh", "value": capacity} for capacity in
                             values.unique()],
                    multi=True,
                    value=values[0]
                )
            ]
        )

    return html.Div(id=ids.VEHICLE_DROP_DOWN)


def hourly_column_dropdown(app: Dash) -> html.Div:
    @app.callback(Output(ids.HOURLY_COLUMN_CHECK_LIST, "children"),
                  Input(ids.COUNTRY_DROP_DOWN, "value"),
                  Input(ids.YEAR_DROP_DOWN, "value"))
    def update_building_dropdown(country: str, year: str):  # needs to be updated because it changes for every project
        values = [variable_name for variable_name in OperationResultVar.__dict__.keys() if
                  not variable_name.startswith("_")]
        return html.Div(
            children=[
                html.H6("Data to plot"),
                dcc.Checklist(
                    id=ids.HOURLY_COLUMN_CHECK_LIST,
                    options=[{"label": item, "value": item} for item in values],
                    value=["E_Heating_HP_out", "ElectricityPrice", "T_outside"],
                    style={
                        'backgroundColor': '#f7f7f7',
                        'color': '#333333',
                        'display': 'flex',
                        'flexDirection': 'column'
                    }
                )
            ]
        )

    return html.Div(id=ids.HOURLY_COLUMN_CHECK_LIST)


def x_axis_selection_drop_down(app: Dash) -> html.Div:
    x_column_names = [variable_name for variable_name in OperationResultVar.__dict__.keys() if
                not variable_name.startswith("_")]
    return html.Div(
        children=[
            html.H6("x axis"),
            dcc.Dropdown(
                id=ids.X_AXIS_DROP_DOWN,
                options=[{"label": item, "value": item} for item in x_column_names],
                value="Q_RoomCooling",
                multi=False
            )
        ]
    )


def y_axis_selection_drop_down(app: Dash) -> html.Div:
    y_column_names = [variable_name for variable_name in OperationResultVar.__dict__.keys() if
                not variable_name.startswith("_")]
    return html.Div(
        children=[
            html.H6("y axis"),
            dcc.Dropdown(
                id=ids.Y_AXIS_DROP_DOWN,
                options=[{"label": item, "value": item} for item in y_column_names],
                value="Q_RoomHeating",
                multi=False
            )
        ]
    )