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
from optimize_buffer_size import rewritten_5R1C_optimziation, find_6R2C_params, calculate_6R2C_with_specific_params
from pathlib import Path


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
    # dual_orig_indoor_temp = np.array([orig_model.dual[orig_model.room_temperature_rule[t]] for t in range(1, 8761)])
    # dual_new_indoor_temp = np.array([new_model.dual[new_model.room_temperature_rule[t]] for t in range(1, 8761)])

    dual_orig_thermal_mass_temp = np.array([orig_model.dual[orig_model.thermal_mass_temperature_rule[t]] for t in range(1, 8761)])
    dual_new_thermal_mass_temp = np.array([new_model.dual[new_model.thermal_mass_temperature_rule[t]] for t in range(1, 8761)])

    # dual_new_surface_temp = np.array([new_model.dual[new_model.surface_temperature_rule[t]] for t in range(1, 8761)])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["T_Room_Dual", "T_ThermalMass_Dual", "Sum of Duals (mass, air, surface temp)"])

    # Add traces (each subplot gets two lines)
    # fig.add_trace(go.Scatter(x=x_axis, y=dual_orig_indoor_temp, mode='lines', name="Orig"), row=1, col=1)
    # fig.add_trace(go.Scatter(x=x_axis, y=dual_new_indoor_temp, mode='lines', name="New"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_orig_thermal_mass_temp, mode='lines', name="Orig"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=dual_new_thermal_mass_temp, mode='lines', name="New"), row=2, col=1)


    # fig.add_trace(go.Scatter(x=x_axis, y=dual_orig_indoor_temp+dual_orig_thermal_mass_temp, mode='lines', name="New"), row=3, col=1)
    # fig.add_trace(go.Scatter(x=x_axis, y=dual_new_indoor_temp+dual_new_thermal_mass_temp, mode='lines', name="Orig"), row=3, col=1)



    # Update layout for clarity
    fig.update_layout(
        height=3000, width=2500, 
        title_text="Model Comparison",
        showlegend=True
    )

    # Show the plot
    fig.show()


def load_IDA_ICE_indoor_temp(scenario_id: int):
    CM = CompareModels(config.project_name)
    df = CM.read_indoor_temp_daniel(price="price2", cooling=False, floor_heating=True)
    df.columns = [key  for key, value in CM.building_names.items() for x in df.columns if value==x ]
    return df[scenario_id].to_numpy()

def load_IDA_ICE_heating(scenario_id: int) -> np.array:
    CM = CompareModels(config.project_name)
    df = CM.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=True)
    df.columns = [key  for key, value in CM.building_names.items() for x in df.columns if value==x ]
    return df[scenario_id].to_numpy()


def compare_IDA_ICE_6R2C_5R1C(model_6R2C: OptOperationModel, model_5R1C: OptOperationModel, scenario_id: int):
    IDA_ICE_T_room = load_IDA_ICE_indoor_temp(scenario_id)
    IDA_ICE_heating = load_IDA_ICE_heating(scenario_id)

    T_room_5R1C = np.array(list(model_5R1C.T_Room.extract_values().values()))
    T_room_6R2C = np.array(list(model_6R2C.T_Room.extract_values().values()))
    
    Q_RoomHeating_5R1C = np.array(list(model_5R1C.Q_RoomHeating.extract_values().values()))
    Q_RoomHeating_6R2C = np.array(list(model_6R2C.Q_RoomHeating.extract_values().values()))

    T_BuildingMass_5R1C = np.array(list(model_5R1C.T_BuildingMass.extract_values().values()))
    T_BuildingMass_6R2C = np.array(list(model_6R2C.T_BuildingMass.extract_values().values()))
    T_Floor_6R2C = np.array(list(model_6R2C.T_floor.extract_values().values()))

    x_axis = np.arange(8760)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["T_Room", "Q_RoomHeating", "T_BuildingMass"])

    # Add traces (each subplot gets two lines)
    fig.add_trace(go.Scatter(x=x_axis, y=IDA_ICE_T_room, mode='lines', name="IDA ICE"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_room_5R1C, mode='lines', name="5R1C"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_room_6R2C, mode='lines', name="6R2C"), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_axis, y=IDA_ICE_heating, mode='lines', name="IDA ICE"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=Q_RoomHeating_5R1C, mode='lines', name="5R1C"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=Q_RoomHeating_6R2C, mode='lines', name="6R2C"), row=2, col=1)

    fig.add_trace(go.Scatter(x=x_axis, y=T_BuildingMass_6R2C, mode='lines', name="6R2C mass"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_BuildingMass_5R1C, mode='lines', name="5R1C mass"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=T_Floor_6R2C, mode='lines', name="6R2C floor temperature"), row=3, col=1)


    fig.show()

def check_results(orig_5R1C_model: OptOperationModel, target_indoor_temp: np.array, 
                  Q_dict, T_room_dict, T_surface_dict, T_mass_dict, T_floor_dict):
    
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=["T_Room", "Q_Heating", "T_Mass", "T_surface", "T_floor"])
    x_axis = np.arange(8760)

    # T_Room
    fig.add_trace(go.Scatter(x=x_axis, y=target_indoor_temp, mode='lines', name="IDA ICE"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=np.array(list(orig_5R1C_model.T_Room.extract_values().values())), mode='lines', name="5R1C"), row=1, col=1)
    for name, T_room in T_room_dict.items():
        fig.add_trace(go.Scatter(x=x_axis, y=T_room, mode='lines', name=name), row=1, col=1)


    # Q_RoomHeating
    fig.add_trace(go.Scatter(x=x_axis, y=np.array(list(orig_5R1C_model.Q_RoomHeating.extract_values().values())), mode='lines', name="5R1C"), row=2, col=1)
    for name, Q_RoomHeating in Q_dict.items():
        fig.add_trace(go.Scatter(x=x_axis, y=Q_RoomHeating, mode='lines', name=name), row=2, col=1)

    # T_BuildingMass
    fig.add_trace(go.Scatter(x=x_axis, y=np.array(list(orig_5R1C_model.T_BuildingMass.extract_values().values())), mode='lines', name="5R1C"), row=3, col=1)
    for name, T_mass in T_mass_dict.items():
        fig.add_trace(go.Scatter(x=x_axis, y=T_mass, mode='lines', name=name), row=3, col=1)

    # T_surface
    for name, T_surface in T_surface_dict.items():
        fig.add_trace(go.Scatter(x=x_axis, y=T_surface, mode='lines', name=name), row=4, col=1)

    # T_floor
    for name, T_floor in T_floor_dict.items():
        fig.add_trace(go.Scatter(x=x_axis, y=T_floor, mode='lines', name=name), row=5, col=1)
        
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

    target_indoor_temp = load_IDA_ICE_indoor_temp(scenario_id)
    # model_6R2C, solve_status = find_6R2C_params(model=OptOperationModel(scenario), target_indoor_temp=target_indoor_temp)
    
    orig_5R1C_model, solve_status = OptOperationModel(scenario).solve(orig_opt_instance)

    # water per square meter floor heating is asumed to be between 1 and 5 liters:
    Cf_list = [x*scenario.building.Af*list(orig_5R1C_model.CPWater.extract_values().values())[0]*3600 for x in range(1, 6)]
    # Heat transfer resistance (W/K) between floor and indoor air temp is assumed to be between 1 and 10 W/K per square meter of floor heating
    # In literature (https://doi.org/10.1016/j.enbuild.2013.07.065) H_f is between 8 and 11 (measured)
    H_f_list = [x*scenario.building.Af for x in range(5, 15)]
    solved_dict = {}
    Q_RoomHeating_6R2C_dict = {}
    T_surface_dict = {}
    T_Room_dict = {}
    T_mass_dict = {}
    T_floor_dict = {}

    
    for c, Cf in enumerate(Cf_list):
        for h, H_f in enumerate(H_f_list):

            model_6C2C, solved = calculate_6R2C_with_specific_params(model=OptOperationModel(scenario), Htr_f=H_f, Cf=Cf, target_indoor_temp=target_indoor_temp)
            if solved:
                solved_dict[f"Cf:{c}, Hf: {h}"] = (Cf, H_f)
                Q_RoomHeating_6R2C_dict[f"Cf:{c}, Hf: {H_f}"] = np.array(list(model_6C2C.Q_RoomHeating.extract_values().values()))
                T_surface_dict[f"Cf:{c}, Hf: {h}"] = np.array(list(model_6C2C.T_surface.extract_values().values()))
                T_Room_dict[f"Cf:{c}, Hf: {h}"] = np.array(list(model_6C2C.T_Room.extract_values().values()))
                T_mass_dict[f"Cf:{c}, Hf: {h}"] = np.array(list(model_6C2C.T_BuildingMass.extract_values().values()))
                T_floor_dict[f"Cf:{c}, Hf: {h}"] = np.array(list(model_6C2C.T_floor.extract_values().values()))
    
    # now iterate over Cf_list and H_f_list and see which results in the best results
    df = pd.DataFrame.from_dict({
    "Q_RoomHeating_6R2C": Q_RoomHeating_6R2C_dict,
    "T_surface": T_surface_dict,
    "T_Room": T_Room_dict,
    "T_mass": T_mass_dict,
    "T_floor": T_floor_dict
    })
    df.to_csv(path=Path(__file__).parent / "6R1C_results.csv", index=False)

    check_results(
        orig_5R1C_model=orig_5R1C_model,
        target_indoor_temp=target_indoor_temp,
        Q_dict=Q_RoomHeating_6R2C_dict,
        T_room_dict=T_Room_dict,
        T_surface_dict=T_surface_dict,
        T_mass_dict=T_mass_dict,
        T_floor_dict=T_floor_dict
    )







    # fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=["T_Room_Dual", "T_ThermalMass_Dual", "Sum of Duals (mass, air, surface temp)"])
    # x_axis = np.arange(8760)
    # dual_values = []  # Store dual values for this constraint

    # for constraint in orig_5R1C_model.component_objects(pyo.Constraint, active=True):
    #     print(f"\nConstraint: {constraint.name}")
        
    #     dual = np.array([orig_5R1C_model.dual[constraint[t]] for t in range(1, 8761)])
    #     dual_values.append(dual)
       
                    
    #     fig.add_trace(go.Scatter(
    #         x=x_axis, 
    #         y=dual, 
    #         mode='lines', name=constraint.name), row=1, col=1)
    # fig.add_trace(go.Scatter(
    # x=x_axis, 
    # y=np.array(dual_values).sum(axis=0), 
    # mode='lines', name="Total"), row=1, col=1)
    # fig.show()

    # plot_comparison_of_main_chars(orig_model=orig_5R1C_model, new_model=new_5R1C_model)
    # show_important_dual_variables(orig_model=orig_5R1C_model, new_model=new_5R1C_model)




