import pyomo.environ as pyo
import numpy as np
from pathlib import Path
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from Core_rc_model import rc_heating_cooling
from pyomo.util.infeasible import log_infeasible_constraints
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A1_Config.A11_Constants import CONS

def create_building_dataframe():
    #%%
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    endungen =  {"_AUT": 20, "_GER": 5}
    for endung, id_country in endungen.items():
        dateiName_dynamic = "001__dynamic_calc_data_bc_2015" + endung + ".csv"
        dateiName_classes = "001_Building_Classes_2015" + endung + ".csv"
        dynamic_calc = pd.read_csv(base_input_path / dateiName_dynamic,
                                   sep=None,
                                   engine="python")\
                                    .dropna().drop(columns=["average_effective_area_wind_west_east_red_cool",
                                                            "average_effective_area_wind_south_red_cool",
                                                            "average_effective_area_wind_north_red_cool",
                                                            "spec_int_gains_cool_watt"])
        building_classes = pd.read_csv(base_input_path / dateiName_classes,
                                       sep=None,
                                       engine="python").dropna().reset_index(drop=True)
        building_df = pd.concat([building_classes, dynamic_calc], axis=1)
        # drop a lot of columns:
        building_df = building_df.drop(columns=[
                                        "building_life_time_index",
                                        "length_of_building",
                                        "width_of_building",
                                        "number_of_floors",
                                        "room_height",
                                        "building_geometry_special_index",
                                        "percentage_of_building_surface_attached_length",
                                        "percentage_of_building_surface_attached_width",
                                        "share_of_window_area_on_gross_surface_area",
                                        "share_of_windows_oriented_to_south",
                                        "share_of_windows_oriented_to_north",
                                        "gebaudebauweise_fbw",
                                        "renovation_facade_year_start",
                                        "renovation_facade_year_end",
                                        "f_x3_facade",
                                        "f_x2_facade",
                                        "f_x1_facade",
                                        "f_x0_facade",
                                        "renovation_windows_year_start",
                                        "renovation_windows_year_end",
                                        "f_x3_windows",
                                        "f_x2_windows",
                                        "f_x1_windows",
                                        "f_x0_windows",
                                        "envelope_quality_def_index",
                                        "mech_ventilation_system_index",
                                        "user_profiles_index",
                                        "climate_region_index",
                                        "agent_mixes_index",
                                        "bc_specific_userfactor",
                                        "INPUT_DATA_STOP",
                                        "grossfloor_area",  # ist = Af
                                        "heated_area",
                                        "total_vertical_surface_area",
                                        "aewd",
                                        "areafloor",
                                        "areadoors",
                                        "grossvolume",
                                        "heatedvolume",
                                        "heated_norm_volume",
                                        "characteristic_length",
                                        "A_V_ratio",
                                        "LEK",
                                        "hwb",
                                        "hlpb",
                                        "orig_hlpb",
                                        "hlpb_dhw",
                                        "t_indoor_eff_factor",
                                        "effective_indoor_temp_jan",
                                        "prev_index",
                                        "last_action",
                                        "last_action_year",
                                        "mean_construction_period",
                                        "uedh_sh_norm",
                                        "uedh_sh_effective",
                                        "uedh_sh_original",
                                        "climate_change_factor1",
                                        "uedh_sh_climate1",
                                        "climate_change_factor1a",
                                        "uedh_sh_climate1a",
                                        "climate_change_factor1b",
                                        "uedh_sh_climate1b",
                                        "climate_change_factor2",
                                        "uedh_sh_climate2",
                                        "ued_cool",
                                        "ued_cool_climate1",
                                        "ued_cool_climate1a",
                                        "ued_cool_climate1b",
                                        "ued_cool_climate2",
                                        "ued_cool_no_int_gains",
                                        "uedh_sh_savings_mech_vent",
                                        "effective_heating_hours",
                                        "effective_operation_hours_solar",
                                        "boiler_operation_hours_dhw",
                                        "distribution_operation_hours_dhw",
                                        "rand_number_1",
                                        "rand_number_2",
                                        "rand_number_3",
                                        "u_value_ceiling",
                                        "u_value_exterior_walls",
                                        "u_value_windows1",
                                        "u_value_windows2",
                                        "u_value_roof",
                                        "u_value_zangendecke",
                                        "u_value_floor",
                                        "seam_loss_windows",
                                        "n_50",
                                        "envelope_quality_add_data_idx",
                                        "add_building_life_time",
                                        "facade_life_time_factor",
                                        "windows_life_time_factor",
                                        "envelope_lifetime_index",
                                        "trans_loss_walls",
                                        "trans_loss_ceil",
                                        "trans_loss_wind",
                                        "trans_loss_doors",
                                        "trans_loss_floor",
                                        "trans_loss_therm_bridge",
                                        "trans_loss_ventilation",
                                        "total_heat_losses",
                                        "uedh_sh_effective_gains",
                                        "uedh_sh_effective_gains_solar",
                                        "uedh_sh_effective_gains_lighting",
                                        "uedh_sh_effective_gains_internal_other",
                                        "uedh_sh_effective_gains_internal_people",
                                        "uedh_sh_norm_gains_solar_transparent",
                                        "ued_cool_gains_solar_transparent",
                                        "ued_cool_gains_solar_opaque",
                                        "ued_cool_gains_lighting",
                                        "ued_cool_gains_internal_other",
                                        "ued_cool_gains_internal_people",
                                        "internal_cool_gains_lighting",
                                        "internal_gains_lighting",
                                        "internal_cool_gains_other_appliances",
                                        "internal_gains_other_appliances",
                                        "internal_cool_gains_people",
                                        "internal_gains_people",
                                        "uedh_sh_effective_losses_before_gains",
                                        "estimated_electricity_internal_gains",
                                        "estimated_electricity_not_internal_gains",
                                        "uedh_sh_effective_jan",
                                        "uedh_sh_effective_feb",
                                        "uedh_sh_effective_mar",
                                        "uedh_sh_effective_apr",
                                        "uedh_sh_effective_may",
                                        "uedh_sh_effective_jun",
                                        "uedh_sh_effective_jul",
                                        "uedh_sh_effective_aug",
                                        "uedh_sh_effective_sep",
                                        "uedh_sh_effective_oct",
                                        "uedh_sh_effective_nov",
                                        "uedh_sh_effective_dec",
                                        "uedh_sh_effective_gains_solar_jan",
                                        "uedh_sh_effective_gains_solar_feb",
                                        "uedh_sh_effective_gains_solar_mar",
                                        "uedh_sh_effective_gains_solar_apr",
                                        "uedh_sh_effective_gains_solar_may",
                                        "uedh_sh_effective_gains_solar_jun",
                                        "uedh_sh_effective_gains_solar_jul",
                                        "uedh_sh_effective_gains_solar_aug",
                                        "uedh_sh_effective_gains_solar_sep",
                                        "uedh_sh_effective_gains_solar_oct",
                                        "uedh_sh_effective_gains_solar_nov",
                                        "uedh_sh_effective_gains_solar_dec",
                                        "annual_mech_vent_volume_heat_recovery_mode",
                                        "annual_mech_vent_volume_non_heat_recovery_mode",
                                        "ued_cooling_jan",
                                        "ued_cooling_feb",
                                        "ued_cooling_mar",
                                        "ued_cooling_apr",
                                        "ued_cooling_may",
                                        "ued_cooling_jun",
                                        "ued_cooling_jul",
                                        "ued_cooling_aug",
                                        "ued_cooling_sep",
                                        "ued_cooling_oct",
                                        "ued_cooling_nov",
                                        "ued_cooling_dec",
                                        "renovation_inv_costs",
                                        "of_which_maintenance_inv_costs_part",
                                        "target_hwb",
                                        "target_hwb_asterix",
                                        "hwb_lc_factor",
                                        "target_hwb_u_value_adoption_factor",
                                        "hwb_original_u_values_using_days",
                                        "hwb_using_days",
                                        "immissionFlaeche_sommer_nachweis_B8110_3",
                                        "immissionsflaechenbezogenerLuftwechsel_sommer_nachweis_B8110_3",
                                        "erforderliche_speicherwirksameMasse_sommer_nachweis_B8110_3",
                                        "immissionsflaechenbezogene_speicherwirksameMasse_sommer_nachweis_B8110_3",
                                        "heatstoragemass_ratio_sommer_nachweis_B8110_3",
                                        "start_refurb_obligation_year",
                                        "is_unrefurbished",
                                        "deep_refurb_level",
                                        "delta_hwb_refurb",
                                        "average_hwb_unrefurbished_simulation_start",
                                        "average_hwb_unrefurbished_current_period",
                                        "hwb_ref_climate_zone",
                                        "is_unrefurbished_age_not_considered",
                                        "cum_inv_env_exist_build",
                                        "cum_sub_env_exist_build",
                                        "bc_index",
                                        "Tset",
                                        "hwb_norm",
                                        "user_profile",
                                        ])
        building_df["country_ID"] = id_country
        excel_name = "building_dataframe_2015" + endung + ".xlsx"
        building_df.to_excel(base_input_path / excel_name)
    #%%



def create_radiation_gains():
    data = DB().read_DataFrame(REG().Sce_Weather_Radiation, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    return data.loc[:, "Radiation"].to_numpy()
# radiation = create_radiation_gains()

def get_elec_profile():
    data = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, conn=DB().create_Connection(CONS().RootDB))
    return data.loc[:, "BaseElectricityProfile"].to_numpy()


def get_out_temp():
    data = DB().read_DataFrame(REG().Sce_Weather_Temperature, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    temperature = data.loc[:, "Temperature"].to_numpy()
    return temperature



def get_invert_data():
    #%%

    project_directory_path = Path(__file__).parent.resolve()
    base_results_path = project_directory_path / "inputdata"

    data = pd.read_excel(base_results_path / "building_dataframe_2015_AUT.xlsx", engine="openpyxl").drop(columns="Unnamed: 0")
    # take only single family houses:
    data = data.loc[data["building_categories_index"]==2]

    # konditionierte Nutzfläche
    Af = data.loc[:, "Af"].to_numpy()
    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af  # 7.2.2.2
    # Airtransfercoefficient
    Hve = data.loc[:, "Hve"].to_numpy()
    # Transmissioncoefficient wall
    Htr_w = data.loc[:, "Htr_w"].to_numpy()
    # Transmissioncoefficient opake Bauteile
    Hop = data.loc[:, "Hop"].to_numpy()
    # Speicherkapazität J/K
    Cm = data.loc[:, "CM_factor"].to_numpy() * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = data.loc[:, "Am_factor"].to_numpy() * Af
    # internal gains
    Qi = data.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
    # HWB_norm = data.loc[:, "hwb_norm"].to_numpy()

    # window areas in celestial directions
    Awindows_rad_east_west = data.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy()
    Awindows_rad_south = data.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy()
    Awindows_rad_north = data.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy()

    # solar gains through windows TODO berechne auf Fensterfläche!
    Qsol = create_radiation_gains()


    # electricity price for 24 hours:
    elec_price = pd.read_excel(base_results_path / "Elec_price_per_hour.xlsx", engine="openpyxl")
    elec_price = elec_price.loc[:, "Euro/MWh"].dropna().to_numpy() + 0.1


    return Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Qsol, elec_price, Af

def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary


Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Q_solar, price, Af = get_invert_data()
temperature = get_out_temp()
temp_outside = temperature


# fixed starting values:
tank_starting_temp = 50
thermal_mass_starting_temp = 20

# constants:
# water mass in storage
m_water = 10_000  # l
# thermal capacity water
cp_water = 4.2  # kJ/kgK
# constant surrounding temp for water tank
T_a = 20

At = 4.5  # 7.2.2.2

# Kopplung Temp Luft mit Temp Surface Knoten s
his = np.float_(3.45)  # 7.2.2.2
# kopplung zwischen Masse und  zentralen Knoten s (surface)
hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
# thermischer Kopplungswerte W/K
Htr_is = his * Atot
# Equ. C.1
PHI_ia = 0.5 * Qi

# Equ. C.6
Htr_1 = 1 / (1 / Hve + 1 / Htr_is)
# Equ. C.7
Htr_2 = Htr_1 + Htr_w
# Equ.C.8
Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)


def create_pyomo_model(elec_price, tout, Qsol, Am, Atot, Cm, Hop, Htr_1, Htr_2, Htr_3, Htr_em, Htr_is, Htr_ms,
                       Htr_w, Hve, PHI_ia, Qi):
    # model
    m = pyo.AbstractModel()

    # parameters
    m.time = pyo.RangeSet(len(elec_price))   # later just len(elec_price) for whole year
    # electricity price
    m.p = pyo.Param(m.time, initialize=create_dict(elec_price))
    # outside temperature
    m.T_outside = pyo.Param(m.time, initialize=create_dict(tout))
    # solar gains
    m.Q_sol = pyo.Param(m.time, initialize=create_dict(Qsol))


    # variables
    # energy used for heating
    m.Q_heating = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 15_000))
    # energy used for cooling
    m.Q_cooling = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 15_000))
    # real indoor temperature
    m.T_room = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(20, 25))
    # thermal mass temperature
    m.Tm_t = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 50))


    # objective
    def minimize_cost(m):
        return sum((m.Q_heating[t] + m.Q_cooling[t]) * m.p[t] for t in m.time)
    m.OBJ = pyo.Objective(rule=minimize_cost)


    # constraints
    def thermal_mass_temperature_rc(m, t):
        if t == 1:
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            # Equ. C.5
            PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                    PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve) +
                                                               m.T_outside[t])) / Htr_2

            # Equ. C.4
            return m.Tm_t[t] == (thermal_mass_starting_temp * ((Cm/3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                        (Cm/3600) + 0.5 * (Htr_3 + Htr_em))
        # if t == 168:
        #     # Equ. C.2
        #     PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
        #     # Equ. C.3
        #     PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
        #
        #     # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
        #     # Equ. C.5
        #     PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
        #             PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t]) / Hve) + m.T_outside[t])) / \
        #                Htr_2
        #
        #     # Equ. C.4
        #     return m.Tm_t[t] == 20
        else:
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            T_sup = m.T_outside[t]
            # Equ. C.5
            PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                    PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve) +
                                                               T_sup)) / Htr_2

            # Equ. C.4
            return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm/3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                                (Cm/3600) + 0.5 * (Htr_3 + Htr_em))
    m.thermal_mass_temperature_rule = pyo.Constraint(m.time, rule=thermal_mass_temperature_rc)


    def room_temperature_rc(m, t):
        if t == 1:
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + thermal_mass_starting_temp) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (
                    PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
            # Equ. C.11
            T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / (Htr_is + Hve)
            # Equ. C.12
            T_op = 0.3 * T_air + 0.7 * T_s
            # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
            return m.T_room[t] == T_air
        else:
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + m.Tm_t[t-1]) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (
                    PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
            # Equ. C.11
            T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / (Htr_is + Hve)
            # Equ. C.12
            T_op = 0.3 * T_air + 0.7 * T_s
            # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
            return m.T_room[t] == T_air
    m.room_temperature_rule = pyo.Constraint(m.time, rule=room_temperature_rc)


    instance = m.create_instance(report_timing=True)
    opt = pyo.SolverFactory("gurobi")
    results = opt.solve(instance, tee=True)
    print(results)
    instance.display("./log.txt")
    return instance, m


def calculate_cost_diff(instance, Q_HC, price):
    total_cost_optimized = instance.OBJ()/100/1_000  # # €
    # total cost not optimized:
    total_cost_normal = sum(Q_HC * price)/100/1_000  # €

    # difference in energy consumption
    total_energy_optimized = sum(np.array([instance.Q_heating[t]() for t in m.time])) / 1_000  # kWh
    total_energy_normal = sum(Q_HC) / 1_000  # kWh

    x_ticks = [1, 2, 3, 4]
    fig = plt.figure()
    ax = plt.gca()
    ax2 = ax.twinx()
    colors = ["blue", "skyblue", "darkred", "orangered"]
    labels = ["total_cost_normal", "total_cost_optimized", "total_energy_normal", "total_energy_optimized"]
    ax.bar(x_ticks, [total_cost_normal, total_cost_optimized, 0, 0], label=labels, color=colors)
    ax.bar(x_ticks, [0, total_cost_normal-total_cost_optimized, 0, 0], color="green", bottom=total_cost_optimized)
    ax.text(1.85, total_cost_optimized + (total_cost_normal-total_cost_optimized)/2,
            str(round(total_cost_normal-total_cost_optimized, 2)))

    ax2.bar(x_ticks, [0, 0, total_energy_normal, total_energy_optimized], color=colors)
    ax2.bar(x_ticks, [0, 0, total_energy_optimized-total_energy_normal, 0], color="darkorchid", bottom=total_energy_normal)
    ax2.text(2.85, total_energy_normal + (total_energy_optimized-total_energy_normal)/2,
             str(round(total_energy_optimized-total_energy_normal, 2)))

    ax.set_xticks(x_ticks)
    ax.set_ylabel("EUR")
    ax2.set_ylabel("kWh")
    ax.set_xticklabels(labels, rotation=15)
    plt.title("Cost and Energy difference")
    plt.show()

# create plots to visualize results
def show_results(instance, m, Q_HC, Tm_t, Q_solar, price, temp_outside):
    Q_a = np.array([instance.Q_heating[t]() for t in m.time])
    Q_cool = np.array([instance.Q_cooling[t]() for t in m.time])
    T_room = [instance.T_room[t]() for t in m.time]
    T_mass_mean = [instance.Tm_t[t]() for t in m.time]

    total_cost = instance.OBJ()

    # total_cost = instance.Objective()
    x_achse = np.arange(len(Q_a))

    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    # ax4 = ax3.twinx()

    # ax1.bar(x_achse, Q_e, label="boiler power")
    ax1.plot(x_achse, Q_a, label="heating power", color="red")
    ax1.plot(x_achse, Q_HC, label="HC", color="black", linestyle="--")
    ax1.plot(x_achse, Q_cool, label="cooling power", color="blue")
    ax1.plot(x_achse, Q_solar, label="solar power", color="green", alpha=0.5)
    ax2.plot(x_achse, price, color="orange", label="price")

    ax1.set_ylabel("energy Wh")
    # ax1.yaxis.label.set_color('green')
    ax2.set_ylabel("price per kWh")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="room temp", color="blue")
    ax3.plot(x_achse, T_mass_mean, label="structure temp", color="grey", alpha=0.2)
    ax3.plot(x_achse, temp_outside, label="outside temp", color="skyblue", alpha=0.5)

    ax3.set_ylabel("temperature °C")
    # ax3.yaxis.label.set_color('blue')
    ax3.legend()
    ax1.grid()
    ax3.grid()
    ax1.set_title("Total costs: " + str(round(total_cost / 1000, 3)))
    plt.show()


    fig2, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(x_achse, Q_a, label="heating optimized", color="red")
    ax2.plot(x_achse, T_mass_mean, label="temp thermal mass optimized", color="orange")
    ax1.plot(x_achse, get_elec_profile()[0:len(Q_a)]*1000, label="normal elec demand", color="green")
    ax1.plot(x_achse, Q_HC, label="heating normal", color="blue", alpha=0.5)
    ax2.plot(x_achse, Tm_t, label="temp thermal mass normal", color="skyblue", alpha=0.5)
    ax1.legend()
    ax2.legend()
    ax1.set_title("Heating + Elec demand")
    plt.show()


    # COP for heatpump
    COP = 3
    fig3 = plt.figure()
    plt.plot(x_achse, Q_a/3+get_elec_profile()[0:len(Q_a)]*1000, label="elec with optim heating")
    plt.plot(x_achse, Q_HC/3+get_elec_profile()[0:len(Q_a)]*1000, label="elec normal heating", linestyle="--")

    plt.title("compare elec demands with COP 3")
    plt.legend()
    plt.show()


number_hours_toplot = 8760
Q_HC_real, Tm_t = rc_heating_cooling(Q_solar[0:number_hours_toplot], Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Af,
                                     temp_outside[0:number_hours_toplot],
                                     initial_thermal_mass_temp=20, T_air_min=20, T_air_max=28)
for i in range(1):

    instance, m = create_pyomo_model(price[0:number_hours_toplot], temp_outside[0:number_hours_toplot],
                                     Q_solar[0:number_hours_toplot], Am[i], Atot[i], Cm[i], Hop[i],
                                     Htr_1[i], Htr_2[i], Htr_3[i], Htr_em[i], Htr_is[i], Htr_ms[i], Htr_w[i], Hve[i],
                                     PHI_ia[i], Qi[i])


    show_results(instance, m, Q_HC_real[:, i], Tm_t[:, i], Q_solar[0:number_hours_toplot],
                 price[0:number_hours_toplot], temp_outside[0:number_hours_toplot])

    calculate_cost_diff(instance, Q_HC_real[:, i], price[0:number_hours_toplot])