from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from typing import List
from dash_visualization import ids, data_extraction


def create_generation_slice_start_input_field():
    return html.Div([
        dcc.Input(id=ids.SLICE_DEMAND_SHIFT_START_INPUT_OUTPUT,
                  type='number',
                  placeholder='Enter a number between 0 and 8759',
                  debounce=True
                  ),
    ])


def create_generation_slice_end_input_field():
    return html.Div([
        dcc.Input(id=ids.SLICE_DEMAND_SHIFT_END_INPUT_OUTPUT,
                  type='number',
                  placeholder='Enter a number between 0 and 8759',
                  debounce=True
                  ),

    ])



