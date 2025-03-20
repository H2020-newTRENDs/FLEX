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
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from model_6R2C import rewritten_5R1C_optimziation, find_6R2C_params, calculate_6R2C_with_specific_params, optimize_6R2C
from pathlib import Path
from compare_results import CompareModels
import tqdm
import pickle  # Added for saving dictionary



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
    name_to_id = {value: key for key, value in CM.building_names.items()}
    df.columns = [name_to_id[col] for col in df.columns]
    return df[scenario_id].to_numpy()

def load_IDA_ICE_heating(scenario_id: int) -> np.array:
    CM = CompareModels(config.project_name)
    df = CM.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=True)
    name_to_id = {value: key for key, value in CM.building_names.items()}
    df.columns = [name_to_id[col] for col in df.columns]
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
    # T_Floor_6R2C = np.array(list(model_6R2C.T_floor.extract_values().values()))

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
    # fig.add_trace(go.Scatter(x=x_axis, y=T_Floor_6R2C, mode='lines', name="6R2C floor temperature"), row=3, col=1)


    fig.show()

def check_results(orig_5R1C_model: OptOperationModel, 
                  target_indoor_temp: np.array, 
                  target_heating: np.array,
                  df_dict: dict, 
                  best_param=None):

    
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=["T_Room", "Q_Heating", "T_Mass", "T_surface", "T_floor"])
    x_axis = np.arange(8760)

    # T_Room
    fig.add_trace(go.Scatter( x=x_axis, y=target_indoor_temp, mode='lines', name="IDA ICE", line=dict(color='red', width=2, dash='dash') ), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=np.array(list(orig_5R1C_model.T_Room.extract_values().values())), mode='lines', name="5R1C"), row=1, col=1)
    for name, df in df_dict.items():
        if name == best_param:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_Room"]].to_numpy().flatten(), mode='lines', name=f"{name} (Best)", 
                                    line=dict(color='black', width=2)), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_Room"]].to_numpy().flatten(), mode='lines', name=name), row=1, col=1)


    # Q_RoomHeating
    fig.add_trace(go.Scatter(x=x_axis, y=np.array(list(orig_5R1C_model.Q_RoomHeating.extract_values().values())), mode='lines', name="5R1C"), row=2, col=1)
    fig.add_trace(go.Scatter( x=x_axis, y=target_heating, mode='lines', name="IDA ICE", line=dict(color='red', width=2, dash='dash'), opacity=0.8 ), row=2, col=1)

    for name, df in df_dict.items():
        if name == best_param:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["Q_RoomHeating"]].to_numpy().flatten(), mode='lines', name=f"{name} (Best)", 
                                    line=dict(color='black', width=2)), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["Q_RoomHeating"]].to_numpy().flatten(), mode='lines', name=name), row=2, col=1)

    # T_BuildingMass
    fig.add_trace(go.Scatter(x=x_axis, y=np.array(list(orig_5R1C_model.T_BuildingMass.extract_values().values())), mode='lines', name="5R1C"), row=3, col=1)
    for name, df in df_dict.items():
        if name == best_param:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_BuildingMass"]].to_numpy().flatten(), mode='lines', name=f"{name} (Best)", 
                                    line=dict(color='black', width=2)), row=3, col=1)
        else:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_BuildingMass"]].to_numpy().flatten(), mode='lines', name=name), row=3, col=1)

    # T_surface
    for name, df in df_dict.items():
        if name == best_param:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_surface"]].to_numpy().flatten(), mode='lines', name=f"{name} (Best)", 
                                    line=dict(color='black', width=2)), row=4, col=1)
        else:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_surface"]].to_numpy().flatten(), mode='lines', name=name), row=4, col=1)

    # T_floor
    for name, df in df_dict.items():
        if name == best_param:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_floor"]].to_numpy().flatten(), mode='lines', name=f"{name} (Best)", 
                                    line=dict(color='black', width=2)), row=5, col=1)
        else:
            fig.add_trace(go.Scatter(x=x_axis, y=df[["T_floor"]].to_numpy().flatten(), mode='lines', name=name), row=5, col=1)
        
    fig.show()

def calc_mean_squared_error_temperature(T_room_array: np.array, scenario_id):
    IDA_ICE_indoor_temp = load_IDA_ICE_indoor_temp(scenario_id=scenario_id)
    squared_errors = (IDA_ICE_indoor_temp - T_room_array) ** 2
    mse = np.mean(squared_errors)
    return mse


def calc_mean_squared_error_heating(heating_array: np.array, scenario_id):
    IDA_ICE_heating = load_IDA_ICE_heating(scenario_id=scenario_id)
    squared_errors = (IDA_ICE_heating - heating_array) ** 2
    mse = np.mean(squared_errors)
    return mse



def read_6R2C_results(variable, scenario_id, results_dir=None):
    """Read results from 6R2C model comparison dataset"""
    if results_dir is None:
        results_dir = Path(__file__).parent / "model_results" / f"comparison_data_{scenario_id}"
    
    file_path = results_dir / f"{variable}.csv"
    if not file_path.exists():
        raise ValueError(f"No data found for {variable} in {results_dir}")
    
    df = pd.read_csv(file_path)
    return df

def extract_static_value_from_model(model, static_values_to_extract: str) -> dict:
    value = list(getattr(model, static_values_to_extract).extract_values().values())[0]
    return value

def extract_results_from_model(model, values_to_extract: list = ["Q_RoomHeating", "T_surface", "T_Room", "T_BuildingMass", "T_floor", "E_Heating_HP_out", "ElectricityPrice", "T_outside", "SpaceHeatingHourlyCOP"]) -> pd.DataFrame: 
    arrays = []
    for name in values_to_extract:
        arrays.append(np.array(list(getattr(model, name).extract_values().values())))
    df = pd.DataFrame(dict(zip(values_to_extract, arrays)))
    return df

def calculate_Heating_flow_6R2C(model):
    results = extract_results_from_model(model, values_to_extract=["T_floor", "T_Room"])
    Q_RoomHeating = (results["T_floor"] - results["T_Room"]) * extract_static_value_from_model(model, static_values_to_extract="Htr_f")
    return Q_RoomHeating

def run_for_different_Cf_and_Hf(Cf_list, H_f_list):
    cp_water =  4200 / 3600
    mse_heating_dict = {}
    heating_solved = {}
    df_heating_dict = {}
    for Cf in tqdm.tqdm(Cf_list):
        for h, H_f in enumerate(H_f_list):
            # first we give a target indoor temp and solve by havin heating as a variable:
            model_6R2C_temp, solved = calculate_6R2C_with_specific_params(model=OptOperationModel(scenario), Htr_f=H_f, Cf=Cf, 
                                                                          target_indoor_temp=target_indoor_temp)
            if solved:
                cf_per_square_meter = round(Cf / (scenario.building.Af*cp_water*3600))
                hf_per_square_meter = round(H_f / (scenario.building.Af))
                heating_solved[f"Cf:{round(cf_per_square_meter)}, Hf: {round(hf_per_square_meter)}"] = (Cf, H_f)
                df = extract_results_from_model(model_6R2C_temp)
                df["Heat_flow_to_room"] = calculate_Heating_flow_6R2C(model_6R2C_temp)
                df_heating_dict[f"Cf:{cf_per_square_meter}, Hf: {hf_per_square_meter}"] = df


                mse_heating = calc_mean_squared_error_heating(df["Q_RoomHeating"].to_numpy(), scenario_id=scenario_id)
                mse_heat_flow_to_room = calc_mean_squared_error_heating(df["Heat_flow_to_room"].to_numpy(), scenario_id=scenario_id)
                mse_heating_dict[f"Cf:{cf_per_square_meter}, Hf: {hf_per_square_meter}"] = mse_heating

    return mse_heating_dict, heating_solved, df_heating_dict

        
if __name__ == "__main__":
    init = ProjectOperationInit(
            config=config,
            input_folder=config.input_operation,
            scenario_components=OperationScenarioComponent,
        )
    init.main()

    # TODO find a way to determine the parameters for all buildings
    # TODO Run 6R2C with best parameters and compare results in cost savings, shifted energy etc.
    # add to diss and paper
    Cf_Hf_dict = {}
    for scenario_id in range(1, 10):
        mother_operation = MotherOperationScenario(config=config)
        orig_opt_instance = OptInstance().create_instance()
        scenario = OperationScenario(scenario_id=scenario_id, config=config, tables=mother_operation)

        target_indoor_temp = load_IDA_ICE_indoor_temp(scenario_id)
        target_heating = load_IDA_ICE_heating(scenario_id=scenario_id)
        # model_6R2C, solve_status = find_6R2C_params(model=OptOperationModel(scenario), target_indoor_temp=target_indoor_temp)
        
        model_5R1C = OptOperationModel(scenario)
        optimized_5R1C_model, solve_status = model_5R1C.solve(orig_opt_instance)

        # fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=["T_Room"])
        # x_axis = np.arange(8760)
        # # fig.add_trace(go.Scatter( x=x_axis, y=model_5R1C.T_Room, mode='lines', name="reference Troom", line=dict(color='black', width=2, dash='dash')), row=1, col=1)
        # fig.add_trace(go.Scatter( x=x_axis, y=np.array(list(optimized_5R1C_model.T_Room.extract_values().values())), mode='lines', name="5R1C Troom",  line=dict(color='black', width=2, dash='dash')), row=1, col=1)
        # fig.add_trace(go.Scatter( x=x_axis, y=target_indoor_temp, mode='lines', name=f"IDA ICE",  ), row=1, col=1)
        # fig.show()


        # water per square meter floor heating is asumed to be between 1 and 5 liters:
        Cf_list = [x*scenario.building.Af*4200 for x in np.arange(0.5, 11.5, 0.5)] #11
        # Heat transfer resistance (W/K) between floor and indoor air temp is assumed to be between 1 and 10 W/K per square meter of floor heating
        # In literature (https://doi.org/10.1016/j.enbuild.2013.07.065) H_f is between 8 and 11 (measured)
        H_f_list = [x*scenario.building.Af for x in np.arange(6, 15.5, 0.5)] #15

        mse_heating_dict, heating_solved, df_heating_dict = run_for_different_Cf_and_Hf(Cf_list, H_f_list)

        best_heating_key, min_mse_heating = min(mse_heating_dict.items(), key=lambda x: x[1])
        results_target_heating = df_heating_dict[best_heating_key]    
        check_results(orig_5R1C_model=optimized_5R1C_model, target_indoor_temp=target_indoor_temp, df_dict=df_heating_dict, best_param=best_heating_key, target_heating=target_heating)


        # use the results from runs with target room temperature instead
        best_key = best_heating_key
        best_Cf, best_Hf = heating_solved[best_key]
        Cf_Hf_dict[scenario_id] = (best_Cf, best_Hf)
        new_6R2C_model, solve_status = optimize_6R2C(
            model=OptOperationModel(scenario),
            Htr_f=best_Hf,
            Cf=best_Cf
        )
        results_6R2C_final = extract_results_from_model(new_6R2C_model)
        
        new_6R2C_simulation, solve_status = optimize_6R2C(
            model=OptOperationModel(scenario),
            Htr_f=best_Hf,
            Cf=best_Cf,
            simulation=True
        )
        results_6R2C_final_simulation = extract_results_from_model(new_6R2C_simulation, values_to_extract=["Q_RoomHeating", "T_surface", "T_Room", "T_BuildingMass", "T_floor", "T_outside"])
        # electricity demand of HP have to be recalculated because of the wrong static COP
        results_6R2C_final_simulation["ElectricityPrice"] = results_6R2C_final["ElectricityPrice"]
        results_6R2C_final_simulation["E_Heating_HP_out"] = results_6R2C_final_simulation["Q_RoomHeating"] / results_6R2C_final["SpaceHeatingHourlyCOP"]

        # save results to csv:
        results_6R2C_final_simulation.to_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C") / f"Scenario_{scenario_id}_reference.csv")
        results_6R2C_final.to_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C") / f"Scenario_{scenario_id}_optimzation.csv")
        
        
        compare_IDA_ICE_6R2C_5R1C(
            model_6R2C=new_6R2C_model,
            model_5R1C=optimized_5R1C_model,
            scenario_id=scenario_id
        )
    
        # Save the Cf_Hf_dict for future use
        output_dir = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C/")
        output_dir.mkdir(parents=True, exist_ok=True)
        cf_hf_dict_path = output_dir / "Cf_Hf_dict.pkl"
        
        # Save using pickle
        if cf_hf_dict_path.exists():
            try:
                with open(cf_hf_dict_path, 'rb') as f:
                    Cf_Hf_dict = pickle.load(f)
                print(f"Loaded existing Cf_Hf_dict from {cf_hf_dict_path}")
                Cf_Hf_dict[scenario_id] = (best_Cf, best_Hf)
            except Exception as e:
                print(f"{e}")
        else:
            Cf_Hf_dict[scenario_id] = (best_Cf, best_Hf)

        with open(cf_hf_dict_path, 'wb') as f:
            pickle.dump(Cf_Hf_dict, f)


    
    # check_results(
    #     orig_5R1C_model=orig_5R1C_model,
    #     target_indoor_temp=target_indoor_temp,
    #     Q_dict=Q_RoomHeating_6R2C_dict,
    #     T_room_dict=T_Room_dict,
    #     T_surface_dict=T_surface_dict,
    #     T_mass_dict=T_mass_dict,
    #     T_floor_dict=T_floor_dict,
    #     best_param=best_param
    # )