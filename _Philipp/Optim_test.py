import pyomo.environ as pyo
import numpy as np
from pathlib import Path
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A1_Config.A11_Constants import CONS


def create_radiation_gains():
    data = DB().read_DataFrame(REG().Sce_Weather_Radiation, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    return data.loc[:, "Radiation"].to_numpy()
# radiation = create_radiation_gains()



def get_navicat():
    data = DB().read_DataFrame(REG().Sce_Weather_Temperature, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    temperature = data.loc[:, "Temperature"].to_numpy()
    return temperature



def get_invert_data():
    project_directory_path = Path(__file__).parent.resolve()
    base_results_path = project_directory_path / "inputdata"
    scn = r'_scen_aut_a00_WAM_plus_v6_ren_mue'

    results_path = base_results_path / scn
    run_number_str = "001"
    results_path_rcm = results_path / "Dynamic_Calc_Input_Data"
    results_path_temperatures = results_path / "Annual_Climate_Data"
    results_path_FWKWK_data = results_path / "FWKWK_REGIONAL_DATA"

    # check if paths exist:
    print("rcm path exists: " + str(results_path_rcm.exists()))
    print("temperatures path exists: " + str(results_path_temperatures.exists()))
    print("FWKWK path exists: " + str(results_path_FWKWK_data.exists()))

    # load building stock data exportet by invert run:
    YEAR = 2020
    datei = str(run_number_str) + "__dynamic_calc_data_bc_" + str(YEAR) + ".csv"
    data = pd.read_csv(results_path_rcm / datei).drop(0, axis=0).reset_index(drop=True)
    haus_1 = data.iloc[0, :]
    haus_2 = data.iloc[14, :]
    haus_3 = data.iloc[25, :]
    haus_DataFrame = pd.concat([haus_1, haus_2, haus_3], axis=1).T  #komplett wertlos aber ich nehm die werte dieser drei häuser
    haus_DataFrame = haus_DataFrame.drop(columns={"climate_region_index", "Tset", "hwb_norm", "user_profile"})

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
    HWB_norm = data.loc[:, "hwb_norm"].to_numpy()

    # window areas in celestial directions
    Awindows_rad_east_west = data.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy()
    Awindows_rad_south = data.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy()
    Awindows_rad_north = data.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy()

    datei = run_number_str + '__climate_data_solar_rad_' + str(YEAR) + ".csv"
    sol_rad = pd.read_csv(results_path_rcm / datei)
    sol_rad = sol_rad.drop(columns=["climate_region_index"]).to_numpy()
    # climate region index - 1 because python starts to count at zero:
    climate_region_index = data.loc[:, "climate_region_index"].to_numpy().astype(int) - 1
    unique_climate_region_index = np.unique(climate_region_index).astype(int)
    sol_rad_north = np.empty((1, 12), float)
    sol_rad_east_west = np.empty((1, 12))
    sol_rad_south = np.empty((1, 12))
    for climate_region in unique_climate_region_index:
        anzahl = len(np.where(climate_region_index == climate_region)[0])
        sol_rad_north = np.append(sol_rad_north,
                                  np.tile(sol_rad[int(climate_region), np.arange(12)], (anzahl, 1)),
                                  axis=0)  # weil von 0 gezählt
        sol_rad_east_west = np.append(sol_rad_east_west,
                                      np.tile(sol_rad[int(climate_region), np.arange(12, 24)], (anzahl, 1)), axis=0)
        sol_rad_south = np.append(sol_rad_south,
                                  np.tile(sol_rad[int(climate_region), np.arange(24, 36)], (anzahl, 1)), axis=0)

    # delete first row in solar radiation as its just zeros
    sol_rad_north = np.delete(sol_rad_north, 0, axis=0)
    sol_rad_east_west = np.delete(sol_rad_east_west, 0, axis=0)
    sol_rad_south = np.delete(sol_rad_south, 0, axis=0)

    # hours per day
    h = np.arange(1, 25)
    # sol_hx = min(14, max(0, h + 0.5 - 6.5))
    # Verteilung der Sonneneinstrahlung auf die einzelnen Stunden mit fixem Profil:
    # Minimalwert = 0 und Maximalwert = 14
    sol_hx = []
    for i in h:
        if h[i - 1] + 0.5 - 6.5 < 0:
            sol_hx.append(0)
        elif h[i - 1] + 0.5 - 6.5 > 14:
            sol_hx.append(14)
        else:
            sol_hx.append(h[i - 1] + 0.5 - 6.5)

    # sol_rad_norm = max(0, np.sin(3.1415 * sol_hx / 13)) / 8.2360
    # Sinusprofil für die Sonneneinstrahlung:
    sol_rad_norm = []
    for i in sol_hx:
        if np.sin(3.1415 * i / 13) / 8.2360 < 0:
            sol_rad_norm.append(0)
        else:
            sol_rad_norm.append(np.sin(3.1415 * i / 13) / 8.2360)

    sol_rad_norm = np.array(sol_rad_norm)
    # solar radiation: Norm profile multiplied by typical radiation of month times 24 hours for one day
    # TODO create 8760 vector not only for first 24 hours
    sol_rad_n = sol_rad_norm * sol_rad_north[0, 0] * 24
    sol_rad_ea = sol_rad_norm * sol_rad_east_west[0, 0] * 24
    sol_rad_s = sol_rad_norm * sol_rad_south[0, 0] * 24

    # solar gains through windows
    Qsol = Awindows_rad_north[0] * sol_rad_n + Awindows_rad_east_west[0] * sol_rad_ea + Awindows_rad_south[0] * sol_rad_s
    Qsol = np.tile(Qsol, (1, 7)).squeeze()

    # electricity price for 24 hours:
    elec_price = pd.read_excel(base_results_path / "Elec_price_per_hour.xlsx", engine="openpyxl")
    elec_price = elec_price.loc[:, "Euro/MWh"].dropna().to_numpy()


    return Atot[[0, 14, 25]], Hve[[0, 14, 25]], Htr_w[[0, 14, 25]], Hop[[0, 14, 25]], Cm[[0, 14, 25]], Am[[0, 14, 25]], Qi[[0, 14, 25]], Qsol, elec_price[0:168], haus_DataFrame



def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary

Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Q_solar, price, haus_DataFrame = get_invert_data()
temperature = get_navicat()
temp_outside = temperature[0:168]

# fixed starting values:
tank_starting_temp = 50
indoor_starting_temp = 22

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

# given values changing over time:
# electricity price
elec_price = create_dict(price)
# outside temperature
tout = create_dict(temp_outside)
# solar gains
Qsol = create_dict(Q_solar)


# TODO Dimensionskontrolle!
# model
m = pyo.AbstractModel()

# parameters
m.indices = pyo.RangeSet(3)    # number of different houses, later len(Atot)
m.time = pyo.RangeSet(168)   # later just len(elec_price) for whole year
# electricity price
m.p = pyo.Param(m.time, initialize=elec_price)
# outside temperature
m.T_outside = pyo.Param(m.time, initialize=tout)
# solar gains
m.Q_sol = pyo.Param(m.time, initialize=Qsol)



# variables
# energy used for heating
m.Q_heating = pyo.Var(m.indices, m.time, within=pyo.NonNegativeReals)
# real indoor temperature
m.T_room = pyo.Var(m.indices, m.time, within=pyo.NonNegativeReals, bounds=(20, 28))
# # mean indoor temperature
m.Tm_t = pyo.Var(m.indices, m.time, within=pyo.NonNegativeReals, bounds=(0, 50))


# objective
def minimize_cost(m, i):
    return sum(m.Q_heating[i, t] * m.p[t] for t in m.time)

m.OBJ = pyo.Objective(m.indices, rule=minimize_cost)


# constraints
def mean_room_temperature_rc(m, t, i):
    if t == 1:
        # Equ. C.2
        PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

        # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
        # Equ. C.5
        PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[i, t]) / Hve) + m.T_outside[t])) / \
                   Htr_2

        # Equ. C.4
        return m.Tm_t[i, t] == (indoor_starting_temp * ((Cm) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                    (Cm ) + 0.5 * (Htr_3 + Htr_em))
    else:
        # Equ. C.2
        PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

        # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
        T_sup = m.T_outside[t]
        # Equ. C.5
        PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[i, t]) / Hve) + T_sup)) / \
                   Htr_2

        # Equ. C.4
        return m.Tm_t[i, t] == (m.Tm_t[i, t - 1] * ((Cm) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                    (Cm ) + 0.5 * (Htr_3 + Htr_em))
m.mean_room_temperature = pyo.Constraint(m.time, m.indices, rule=mean_room_temperature_rc)


def room_temperature_rc(m, t, i):
    if t == 1:
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.9
        T_m = (m.Tm_t[i, t] + indoor_starting_temp) / 2
        T_sup = m.T_outside[t]
        # Euq. C.10
        T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (PHI_ia + m.Q_heating[i, t]) / Hve)) / (
                Htr_ms + Htr_w + Htr_1)
        # Equ. C.11
        T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[i, t]) / (Htr_is + Hve)
        # Equ. C.12
        T_op = 0.3 * T_air + 0.7 * T_s
        # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
        return m.T_room[i, t] == T_air
    else:
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.9
        T_m = (m.Tm_t[i, t] + m.Tm_t[i, t-1]) / 2
        T_sup = m.T_outside[t]
        # Euq. C.10
        T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (PHI_ia + m.Q_heating[i, t]) / Hve)) / (
                Htr_ms + Htr_w + Htr_1)
        # Equ. C.11
        T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[i, t]) / (Htr_is + Hve)
        # Equ. C.12
        T_op = 0.3 * T_air + 0.7 * T_s
        # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
        return m.T_room[i, t] == T_air
m.room_temperature = pyo.Constraint(m.time, m.indices, rule=room_temperature_rc)


instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("gurobi")
results = opt.solve(instance, tee=True)
print(results)


# create plots to visualize results
def show_results():
    Q_a = [instance.Q_heating[t]() for t in m.time]
    T_room = [instance.T_room[t]() for t in m.time]
    T_room_mean = [instance.Tm_t[t]() for t in m.time]

    total_cost = instance.OBJ()

    # total_cost = instance.Objective()
    x_achse = np.arange(24)

    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    # ax1.bar(x_achse, Q_e, label="boiler power")
    ax1.plot(x_achse, Q_a, label="heating power", color="red")
    ax1.plot(x_achse, Q_solar, label="solar power", color="green")
    ax2.plot(x_achse, price, color="orange", label="price")

    ax1.set_ylabel("energy Wh")
    # ax1.yaxis.label.set_color('green')

    ax2.set_ylabel("price per kWh")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="room temperature", color="blue")
    ax3.plot(x_achse, T_room_mean, label="mean room temperature", color="grey")
    ax3.plot(x_achse, temp_outside, label="outside temp", color="skyblue")

    # ax4.plot(x_achse, T_tank, label="tank temperature", color="orange")

    ax3.set_ylabel("room temperature °C")

    # ax4.set_ylabel("tank temperature °C")
    ax3.yaxis.label.set_color('blue')
    ax3.legend()
    ax1.grid()
    ax3.grid()
    ax1.set_title("Total costs: " + str(round(total_cost / 1000, 3)))

    plt.show()


show_results()

