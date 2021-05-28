import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from B_Classes.B2_Building import Building, HeatingCooling_noDR
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A1_Config.A11_Constants import CONS


def showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR):
    red = '#F47070'
    blue = '#8EA9DB'
    green = '#A9D08E'
    orange = '#F4B084'
    yellow = '#FFD966'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#305496'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    plt.style.use('ggplot')
    colors = [red, blue, green]
    x_achse = np.arange(len(Q_Heating_noDR))
    fig1 = plt.figure()
    for i in range(3):
        plt.plot(x_achse, Q_Heating_noDR[:, i], label="household "+str(i), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("Heating loads")
    fig1.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichHeizlast.svg")
    plt.show()

    fig2 = plt.figure()
    for i in [2, 1, 0]:
        plt.plot(x_achse, Q_Cooling_noDR[:, i], label="household "+str(i), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("Cooling loads")
    fig2.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichKuehlast.svg")
    plt.show()

    fig3 = plt.figure()


project_directory_path = Path(__file__).parent.resolve()
base_input_path = project_directory_path / "inputdata"
Temperature_outside = pd.read_csv(base_input_path / "Frankfurt_WeatherData.csv", engine="python", sep=None, header=17)

CONN = DB().create_Connection(CONS().RootDB)
B = HeatingCooling_noDR(DB().read_DataFrame(REG().ID_BuildingOption, CONN))


Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR = B.ref_HeatingCooling(Temperature_outside)
showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR)
