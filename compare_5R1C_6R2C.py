import numpy as np
import pandas as pd
from basics.kit import get_logger
from config import config
from models.operation.data_collector import OptDataCollector
from models.operation.data_collector import RefDataCollector
from models.operation.model_opt import OptInstance
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.scenario import OperationScenario, MotherOperationScenario
from operation_init import ProjectOperationInit
from models.operation.enums import OperationScenarioComponent
from scipy.optimize import least_squares, minimize
from basics.db import DB
from projects.Philipp_5R1C.compare_results import CompareModels
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from optimize_buffer_size import rewritten_5R1C_optimziation


'''
In this script I will first compare how the 5R1C compares to the 5R1C-rewritten.
Then the 6R2C parameters for floor capacity and heat transfer to the room are calculated,
through an optimziation. In the end a comparison between 5R1C, 6R2C and IDA ICE is done.
'''


def plot_comparison_of_main_chars(orig_model, new_model):
    x_axis = np.arange(8760)
    Q_roomheating_orig = list(orig_model.Q_RoomHeating.extract_values().values())
    Q_roomheating_new = list(new_model.Q_RoomHeating.extract_values().values())
    
    E_Heating_HP_out_orig = list(orig_model.E_Heating_HP_out.extract_values().values())
    E_Heating_HP_out_new = list(new_model.E_Heating_HP_out.extract_values().values())

    T_Room_orig = list(orig_model.T_Room.extract_values().values())
    T_Room_new = list(new_model.T_Room.extract_values().values())

    T_BuildingMass_orig = list(orig_model.T_BuildingMass.extract_values().values())
    T_BuildingMass_new = list(new_model.T_BuildingMass.extract_values().values())

    ElectricityPrice_orig = list(orig_model.ElectricityPrice.extract_values().values())
    ElectricityPrice_new = list(new_model.ElectricityPrice.extract_values().values())
    # Create subplots: 3 rows, 1 column
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=["Q_RoomHeating", "E_Heating_HP_out", "T_Room", "T_BuildingMass", "ElectricityPrice"])

    # Add traces (each subplot gets two lines)
    fig.add_trace(go.Scatter(x=x_axis, y=Q_roomheating_orig, mode='lines', name="Orig"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=Q_roomheating_new, mode='lines', name="New"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=E_Heating_HP_out_orig, mode='lines', name="Orig"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=E_Heating_HP_out_new, mode='lines', name="New"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_Room_orig, mode='lines', name="Orig"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_Room_new, mode='lines', name="New"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_BuildingMass_orig, mode='lines', name="Orig"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_BuildingMass_new, mode='lines', name="New"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ElectricityPrice_orig, mode='lines', name="Orig"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ElectricityPrice_new, mode='lines', name="New"), row=5, col=1)

    # Update layout for clarity
    fig.update_layout(
        height=3000, width=2500, 
        title_text="Model Comparison",
        showlegend=True
    )

    # Show the plot
    fig.show()

def show_important_dual_variables(orig_model, new_model):
    x_axis = np.arange(8760)

    # duals of indoor temperature:
    dual_orig_indoor_temp = np.array([orig_model.dual[orig_model.room_temperature_rule[t]] for t in range(1, 8761)])
    dual_new_indoor_temp = np.array([new_model.dual[new_model.room_temperature_rule[t]] for t in range(1, 8761)])

    dual_orig_thermal_mass_temp = np.array([orig_model.dual[orig_model.thermal_mass_temperature_rule[t]] for t in range(1, 8761)])
    dual_new_thermal_mass_temp = np.array([new_model.dual[new_model.thermal_mass_temperature_rule[t]] for t in range(1, 8761)])

    dual_new_surface_temp = np.array([new_model.dual[new_model.surface_temperature_rule[t]] for t in range(1, 8761)])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["T_Room_Dual", "T_ThermalMass_Dual", "Sum of Duals (mass, air, surface temp)"])

    # Add traces (each subplot gets two lines)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_orig_indoor_temp, mode='lines', name="Orig"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_new_indoor_temp, mode='lines', name="New"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_orig_thermal_mass_temp, mode='lines', name="Orig"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_new_thermal_mass_temp+dual_new_surface_temp, mode='lines', name="New"), row=2, col=1)


    fig.add_trace(go.Scatter(x=x_axis, y=dual_orig_indoor_temp+dual_orig_thermal_mass_temp, mode='lines', name="New"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_new_indoor_temp+dual_new_thermal_mass_temp+dual_new_surface_temp, mode='lines', name="Orig"), row=3, col=1)



    # Update layout for clarity
    fig.update_layout(
        height=3000, width=2500, 
        title_text="Model Comparison",
        showlegend=True
    )

    # Show the plot
    fig.show()




if __name__ == "__main__":
    # init = ProjectOperationInit(
    #         config=config,
    #         input_folder=config.input_operation,
    #         scenario_components=OperationScenarioComponent,
    #     )
    # init.main()

    scenario_id = 5
    mother_operation = MotherOperationScenario(config=config)
    orig_opt_instance = OptInstance().create_instance()


    
    scenario = OperationScenario(scenario_id=scenario_id, config=config, tables=mother_operation)
    new_5R1C_model, solve_status = rewritten_5R1C_optimziation(model=OptOperationModel(scenario))

    orig_5R1C_model, solve_status = OptOperationModel(scenario).solve(orig_opt_instance)

    plot_comparison_of_main_chars(orig_model=orig_5R1C_model, new_model=new_5R1C_model)
    show_important_dual_variables(orig_model=orig_5R1C_model, new_model=new_5R1C_model)




