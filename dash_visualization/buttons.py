from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from typing import List
from dash_visualization import ids, data_extraction


def save_europe_map_gif_button(app: Dash) -> html.Button:
    return html.Button(
        "Safe as GIF",
        id=ids.SAVE_EUROPE_MAP_GIF_BUTTON,
        n_clicks=0,
    )


def save_europe_map_csv_button(app: Dash) -> html.Button:
    return html.Button(
        "Safe as csv",
        id=ids.SAVE_EUROPE_MAP_CSV_BUTTON,
        n_clicks=0,
    )


def save_demand_profiles_csv_button(app: Dash) -> html.Button:
    return html.Button(
        "Safe as csv",
        id=ids.SAVE_DEMAND_PROFILE_CSV_BUTTON,
        n_clicks=0,
    )


def save_slice_demand_shift_profiles_button(app: Dash) -> html.Button:
    return html.Button(
        "Save as svg",
        id=ids.SAVE_SLICE_GENERATION_SHIFT_BUTTON,
        n_clicks=0,
    )


def battery_button(app: Dash) -> html.Button:
    return html.Button(
        className="battery-button",
        children=["Battery"],
        id=ids.BATTERY_BUTTON,
        n_clicks=0,
        value="ID_Battery"
    )


def boiler_button(app: Dash) -> html.Button:
    return html.Button(
        className="boiler-button",
        children=["Boiler"],
        id=ids.BOILER_BUTTON,
        n_clicks=0,
    )


def heating_element_button(app: Dash) -> html.Button:
    return html.Button(
        className="heating-element-button",
        children=["Heating Element"],
        id=ids.HEATING_ELEMENT_BUTTON,
        n_clicks=0,
    )


def hot_water_tank_button(app: Dash) -> html.Button:
    return html.Button(
        className="hot-water-tank-button",
        children=["Hot Water Tank"],
        id=ids.HOT_WATER_TANK_BUTTON,
        n_clicks=0,
    )


def pv_button(app: Dash) -> html.Button:
    return html.Button(
        className="pv-button",
        children=["PV"],
        id=ids.PV_BUTTON,
        n_clicks=0,
    )


def space_cooling_button(app: Dash) -> html.Button:
    return html.Button(
        className="space-cooling-button",
        children=["Space Cooling"],
        id=ids.SPACE_COOLING_BUTTON,
        n_clicks=0,
    )


def space_heating_tank_button(app: Dash) -> html.Button:
    return html.Button(
        className="space-heating-tank-button",
        children=["Space Heating Tank"],
        id=ids.SPACE_HEATING_TANK_BUTTON,
        n_clicks=0,
    )


def vehicle_button(app: Dash) -> html.Button:
    return html.Button(
        className="vehicle-button",
        children=["Vehicle"],
        id=ids.VEHICLE_BUTTON,
        n_clicks=0,
    )
