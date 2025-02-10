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

logger = get_logger(__name__)


def run_ref_operation(scenario: "OperationScenario"):
    ref_model = RefOperationModel(scenario).solve()
    RefDataCollector(ref_model, scenario.scenario_id, config, save_hour_results=True).run()


def run_opt_scenario(scenario: "OperationScenario", opt_instance):
    opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
    if solve_status:
        OptDataCollector(opt_model, scenario.scenario_id, config, save_hour_results=True).run()


def run_scenarios(scenario_ids):
    init = ProjectOperationInit(
        config=config,
        input_folder=config.input_operation,
        scenario_components=OperationScenarioComponent,
    )
    init.main()

    opt_instance = OptInstance().create_instance()
    mother_operation = MotherOperationScenario(config=config)
    config.input_operation

    for scenario_id in scenario_ids:
        logger.info(f"FlexOperation --> Scenario = {scenario_id}.")

        scenario = OperationScenario(scenario_id=scenario_id, config=config, tables=mother_operation)
        run_ref_operation(scenario)
        run_opt_scenario(scenario, opt_instance)


def load_indoor_set_temp_from_opt(scenario_id: int):
    db = DB(config.output / f"{config.project_name}.sqlite")
    t_room = db.read_dataframe(table_name="OperationResult_OptimizationHour", column_names=["T_Room", "ID_Scenario"])
    t_room = t_room.loc[t_room["ID_Scenario"] == scenario_id, "T_Room"].to_numpy()
    return t_room

def load_Tm_start_value(scenario_id: int):
    db = DB(config.output / f"{config.project_name}.sqlite")
    t_buildingmass = db.read_dataframe(table_name="OperationResult_OptimizationHour", column_names=["T_BuildingMass", "ID_Scenario"])
    t_start = t_buildingmass.loc[t_buildingmass["ID_Scenario"] == scenario_id, "T_BuildingMass"].to_numpy()[0]
    return t_start


def load_CM_factor(scenario_id: int):
    db = DB(config.output / f"{config.project_name}.sqlite")
    cm_factor = db.read_dataframe("OperationScenario_Component_Building")
    cm_factor = cm_factor.loc[cm_factor["ID_Building"] == scenario_id, "CM_factor"].values[0]
    return cm_factor



def calculate_heating_and_cooling_demand(
            Htr_w: float,
            Cm_factor: float,
            Hve: float,
            Hop: float,
            scenario: OperationScenario,
    ) -> (np.array, np.array, np.array, np.array):


        """
        if "static" is True, then the RC model will calculate a static heat demand calculation for the first hour of
        the year by using this hour 100 times. This way a good approximation of the thermal mass temperature of the
        building in the beginning of the calculation is achieved. Solar gains are set to 0.
        Returns: heating demand, cooling demand, indoor air temperature, temperature of the thermal mass
        """
        heating_power_10 = scenario.building.Af * 10
        T_air_max = np.full((8760,), 100)
        time = np.arange(8760)
        Tm_t = np.zeros(shape=(8760,))  # thermal mass temperature
        T_sup = np.zeros(shape=(8760,))
        heating_demand = np.zeros(shape=(8760,))
        cooling_demand = np.zeros(shape=(8760,))
        room_temperature = np.zeros(shape=(8760,))

        area_window_east_west = scenario.building.effective_window_area_west_east
        area_window_south = scenario.building.effective_window_area_south
        area_window_north = scenario.building.effective_window_area_north

        Q_solar_north = np.outer(np.array(scenario.region.radiation_north), area_window_north)
        Q_solar_east = np.outer(np.array(scenario.region.radiation_east), area_window_east_west / 2)
        Q_solar_south = np.outer(np.array(scenario.region.radiation_south), area_window_south)
        Q_solar_west = np.outer(np.array(scenario.region.radiation_west), area_window_east_west / 2)

        T_outside = scenario.region.temperature
        shading_threshold_temperature = scenario.behavior.shading_threshold_temperature
        shading_solar_reduction_rate = scenario.behavior.shading_solar_reduction_rate
        solar_gain_rate = np.ones(len(T_outside), )
        for i in range(0, len(T_outside)):
            if T_outside[i] >= shading_threshold_temperature:
                solar_gain_rate[i] = 1 - shading_solar_reduction_rate
        Q_solar = (Q_solar_north + Q_solar_south + Q_solar_east + Q_solar_west).squeeze() * solar_gain_rate

        Cm = Cm_factor * scenario.building.Af
        Af = scenario.building.Af
        # Hve = scenario.building.Hve
        Am = scenario.building.Am_factor * scenario.building.Af  # Effective mass related area [m^2]
        Atot = (4.5 * scenario.building.Af)  # 7.2.2.2: Area of all surfaces facing the building zone
        Qi = scenario.building.internal_gains * scenario.building.Af
        # Htr_w = scenario.building.Htr_w
        Htr_ms = np.float_(9.1) * Am  # from 12.2.2 Equ. (64)
        Htr_is = np.float_(3.45) * Atot
        Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        Htr_1 = np.float_(1) / (np.float_(1) / scenario.building.Hve + np.float_(1) / Htr_is)  # Equ. C.6
        Htr_2 = Htr_1 + scenario.building.Htr_w  # Equ. C.7
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8
        PHI_ia = 0.5 * Qi  # Equ. C.1

        thermal_start_temperature = load_Tm_start_value(scenario.scenario_id)
        T_air_min = load_indoor_set_temp_from_opt(scenario.scenario_id)

        # RC-Model
        for t in time:  # t is the index for each time step
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Qi + Q_solar[t])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (
                    0.5 * Qi + Q_solar[t]
            )

            # (T_sup = T_outside because incoming air is not preheated)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = (
                    PHI_m
                    + Htr_em * T_outside[t]
                    + Htr_3
                    * (
                            PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1 * (((PHI_ia + 0) / Hve) + T_sup[t])
                    )
                    / Htr_2
            )

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = (
                    PHI_m
                    + Htr_em * T_outside[t]
                    + Htr_3
                    * (
                            PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1
                            * (((PHI_ia + heating_power_10) / Hve) + T_sup[t])
                    )
                    / Htr_2
            )

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = (
                    PHI_m
                    + Htr_em * T_outside[t]
                    + Htr_3
                    * (
                            PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1
                            * (((PHI_ia - heating_power_10) / Hve) + T_sup[t])
                    )
                    / Htr_2
            )

            if t == 0:
                Tm_t_prev = thermal_start_temperature
            else:
                Tm_t_prev = Tm_t[t - 1]

            # Equ. C.4
            Tm_t_0 = (
                            Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em))
                            + PHI_mtot_0
                    ) / (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.4 for 10 W/m^2 heating
            Tm_t_10 = (
                            Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em))
                            + PHI_mtot_10
                    ) / (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.4 for 10 W/m^2 cooling
            Tm_t_10_c = (
                                Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em))
                                + PHI_mtot_10_c
                        ) / (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.9
            T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 heating
            T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 cooling
            T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

            # Euq. C.10
            T_s_0 = (
                            Htr_ms * T_m_0
                            + PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1 * (T_sup[t] + (PHI_ia + 0) / Hve)
                    ) / (Htr_ms + Htr_w + Htr_1)

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (
                            Htr_ms * T_m_10
                            + PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1 * (T_sup[t] + (PHI_ia + heating_power_10) / Hve)
                    ) / (Htr_ms + Htr_w + Htr_1)

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (
                            Htr_ms * T_m_10_c
                            + PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1 * (T_sup[t] + (PHI_ia - heating_power_10) / Hve)
                    ) / (Htr_ms + Htr_w + Htr_1)

            # Equ. C.11
            T_air_0 = (Htr_is * T_s_0 + Hve * T_sup[t] + PHI_ia + 0) / (
                    Htr_is + Hve
            )

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (
                            Htr_is * T_s_10
                            + Hve * T_sup[t]
                            + PHI_ia
                            + heating_power_10
                    ) / (Htr_is + Hve)

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (
                                Htr_is * T_s_10_c
                                + Hve * T_sup[t]
                                + PHI_ia
                                - heating_power_10
                        ) / (Htr_is + Hve)

            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if T_air_0 >= T_air_min[t] and T_air_0 <= T_air_max[t]:
                heating_demand[t] = 0
            elif T_air_0 < T_air_min[t]:  # heating is required
                heating_demand[t] = (
                        heating_power_10 * (T_air_min[t] - T_air_0) / (T_air_10 - T_air_0)
                )
            elif T_air_0 > T_air_max[t]:  # cooling is required
                cooling_demand[t] = (
                        heating_power_10 * (T_air_max[t] - T_air_0) / (T_air_10_c - T_air_0)
                )

            # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
            # Equ. C.5 with actual heating power
            PHI_mtot_real = (
                    PHI_m
                    + Htr_em * T_outside[t]
                    + Htr_3
                    * (
                            PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1
                            * (
                                    (
                                            (PHI_ia + heating_demand[t] - cooling_demand[t])
                                            / Hve
                                    )
                                    + T_sup[t]
                            )
                    )
                    / Htr_2
            )
            # Equ. C.4
            Tm_t[t] = (
                            Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em))
                            + PHI_mtot_real
                    ) / (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.9
            T_m_real = (Tm_t[t] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (
                            Htr_ms * T_m_real
                            + PHI_st
                            + Htr_w * T_outside[t]
                            + Htr_1
                            * (
                                    T_sup[t]
                                    + (PHI_ia + heating_demand[t] - cooling_demand[t]) / Hve
                            )
                    ) / (Htr_ms + Htr_w + Htr_1)

            # Equ. C.11 for 10 W/m^2 heating
            room_temperature[t] = (
                                        Htr_is * T_s_real
                                        + Hve * T_sup[t]
                                        + PHI_ia
                                        + heating_demand[t]
                                        - cooling_demand[t]
                                ) / (Htr_is + Hve)

        return heating_demand, cooling_demand, room_temperature, Tm_t

def load_target_heating(scenario_id: int):
    CM = CompareModels(config.project_name)
    df = CM.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=True)
    df.columns = [key  for key, value in CM.building_names.items() for x in df.columns if value==x ]
    return df[scenario_id].to_numpy()

def load_target_indoor_temp(scenario_id: int):
    CM = CompareModels(config.project_name)
    df = CM.read_indoor_temp(table_name="OperationResult_OptimizationHour", prize_scenario="price2", cooling=False)
    df.columns = [key  for key, value in CM.building_names.items() for x in df.columns if value==x ]
    return df[scenario_id].to_numpy()



def objective(vars, target_heating, scenario: OperationScenario):
    Cm, Htr_w, Hve, Hop = vars[0], vars[1], vars[2], vars[3]
    heating_demand,_,_,_ = calculate_heating_and_cooling_demand(Htr_w=Htr_w, Cm_factor=Cm, Hve=Hve, Hop=Hop, scenario=scenario)

    return np.sqrt(np.mean((heating_demand - target_heating)**2))  # RMSE


def run_ref_with_set_temp_from_opt_and_change_Cm_factor(scenario_ids):

    mother_operation = MotherOperationScenario(config=config)
    for scenario_id in scenario_ids:
        scenario_id = 1
        scenario = OperationScenario(scenario_id=scenario_id, config=config, tables=mother_operation)
        target_heating = load_target_heating(scenario_id)
        Htr_w_start = scenario.building.Htr_w
        start_Cm_factor = load_CM_factor(scenario_id)
        Hve_start = scenario.building.Hve
        Hop_start = scenario.building.Hop
        orig_heating, _, room_temp_orig, _ = calculate_heating_and_cooling_demand(Cm_factor=start_Cm_factor, Htr_w= Htr_w_start, Hve=Hve_start, Hop=Hop_start, scenario=scenario)


        logger.info(f"Optimizing Cm_factor for --> Scenario = {scenario_id}.")
        print("start Cm factor: ", start_Cm_factor)

        # swap variables from x0 to args and vice versa to change if they are optimized or not
        variables = [start_Cm_factor, Htr_w_start, Hve_start, Hop_start]
        parameters = [target_heating, scenario]
        result = minimize(objective, x0=variables, args=(target_heating, scenario), method="L-BFGS-B")
        result = minimize(objective, 
                               x0=variables, 
                               args=(target_heating, scenario),
                               verbose=2
                               )

        print(f"Cm factor: {result.x[0]}")
        print(f"Htr_w: {result.x[1]}")
        print(f"Hve: {result.x[2]}")
        print(f"Hop: {result.x[3]}")
        print(f"Least squares error: {result.cost}") 


        running_mean = np.convolve(scenario.region.temperature, np.ones(24)/24, mode='valid')
        plt.plot(running_mean, label="temperature (24h running mean)", color="blue")

        optimzied_cm_heating,_,room_temp_optimized,_ = calculate_heating_and_cooling_demand(Cm_factor=result.x[0], Htr_w=result.x[1], Hve=result.x[2], Hop=result.x[3], scenario=scenario)
        target_indoor_temp = load_target_indoor_temp(scenario_id)


        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Heating Demand", "Room Temperature"])

        # First Plot: Heating Demand
        fig.add_trace(go.Scatter(x=np.arange(len(orig_heating)), y=orig_heating, mode="lines", name="Original Heating", line=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(optimzied_cm_heating)), y=optimzied_cm_heating, mode="lines", name="Optimized Heating", line=dict(color="red")), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(target_heating)), y=target_heating, mode="lines", name="Target Heating", line=dict(color="green", dash="dash")), row=1, col=1)

        # Second Plot: Room Temperature
        fig.add_trace(go.Scatter(x=np.arange(len(target_indoor_temp)), y=target_indoor_temp, mode="lines", name="Target Room Temp", line=dict(color="green", dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(room_temp_orig)), y=room_temp_orig, mode="lines", name="Original Room Temp", line=dict(color="blue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(room_temp_optimized)), y=room_temp_optimized, mode="lines", name="Optimized Room Temp", line=dict(color="red")), row=2, col=1)

        # Update layout
        fig.update_layout(
            title="Comparison of Heating Demand and Room Temperature",
            xaxis_title="Time Step",
            yaxis_title="Heating Demand (kWh)",
            yaxis2_title="Room Temperature (°C)",
            height=2000,
            width=3500
        )

        # Show interactive Plotly plot
        fig.show()








        





if __name__ == "__main__":

    SCENARIO_IDS = np.arange(1, 10)
    # run_scenarios(SCENARIO_IDS)
    run_ref_with_set_temp_from_opt_and_change_Cm_factor(SCENARIO_IDS)

