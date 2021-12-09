# -*- coding: utf-8 -*-
__author__ = 'Philipp'

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import tkinter as tk
import sqlite3

from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table, REG_Var
from A_Infrastructure.A1_CONS import CONS
from C_Model_Operation.C4_OperationOptimization_paper import DataSetUp, create_abstract_model, update_instance, \
    create_instances2solve
from C_Model_Operation.C4_NoOptimization import no_DR


class HouseholdToTest:

    def __init__(self):
        self.id_chosen = {}
        self.conn = DB().create_Connection(CONS().RootDB)
        self.test_house = "test_household_IDs"

    def choose_household_tkinter(self, frame_dict: dict):
        """creates popup with tkinter where options are displayed.
        selected options are returned"""

        def turn_value_into_string(value) -> str:
            """turns the provided value into a string without commas for floats"""
            try:
                output_value = float(value)
                output_value = int(round(value))
            except ValueError:
                try:
                    output_value = int(value)
                except ValueError:
                    output_value = value
            return str(output_value)

        for frame_name, frame in frame_dict.items():
            # create tkinter root
            root = tk.Tk()
            # title
            root.title(f"{frame_name}")
            # label in first row
            top_label = tk.Label(root, text="choose one option by \n clicking on the index number")
            top_label.grid(row=0, column=0, columnspan=2)
            # describe columns:
            column_label_1 = tk.Label(root, text="ID")
            column_label_2 = tk.Label(root, text=frame.columns[1])
            column_label_3 = tk.Label(root, text=frame.columns[2])
            column_label_1.grid(row=2, column=0)
            column_label_2.grid(row=2, column=1)
            column_label_3.grid(row=2, column=2)
            # create entry that shows which option was chosen:
            entry = tk.Entry(root, width=35, borderwidth=5)
            entry.grid(row=1, column=0, columnspan=2)

            def button_click(ID_number: int) -> None:
                entry.delete(0, tk.END)
                entry.insert(0, ID_number)

            def hit_enter() -> None:
                number = entry.get()
                # save to dictionary id_chosen
                self.id_chosen[frame_name] = int(number)
                # exit tkinter
                root.destroy()

            # loop over frame:
            button = []
            button_text_1 = []
            button_text_2 = []
            for (row_number, row) in frame.iterrows():
                button.append(tk.Button(root, text=turn_value_into_string(row[0]),
                                        command=lambda id_number=int(row[0]): button_click(id_number),
                                        padx=40, pady=10))
                button_text_1.append(tk.Button(root, text=turn_value_into_string(row[1]),
                                               padx=40, pady=10))
                button_text_2.append(tk.Button(root, text=turn_value_into_string(row[2]),
                                               padx=40, pady=10))

                button[row_number].grid(row=3 + row_number, column=0)
                button_text_1[row_number].grid(row=3 + row_number, column=1)
                button_text_2[row_number].grid(row=3 + row_number, column=2)

            enter_button = tk.Button(root, text="Enter", command=hit_enter, padx=80, pady=10)
            enter_button.grid(row=3 + row_number + 1, column=0, columnspan=2)
            root.mainloop()

    def check_dict_exists(self) -> bool:
        """checks if the dict exists in sqlite db. returns True if table exists and False if not"""
        c = self.conn.cursor()
        # get the count of tables with the name
        exists = pd.read_sql(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND NAME ='{}'".format(self.test_house),
            con=self.conn)
        # if the count is 1, then table exists
        if exists.iloc[0].iloc[0] == 1:
            print('Table exists.')
            # close the connection
            return True
        else:
            print('Table does not exist.')
            # close the connection
            return False

    def define_household_IDs(self, new_id: bool = False) -> pd.DataFrame:
        """defines the household used for the comparison and returns the IDs as Dataframe
        if new_id = False (default) the IDs are loaded from the database if they exist
        if new_id = True the IDs will be newly generated through user input"""
        # check if sqlite file exists:
        if self.check_dict_exists() and not new_id:
            # read the dictionary from database
            test_IDs_df = DB().read_DataFrame(table_name=self.test_house, conn=self.conn)
        else:  # household IDs will be determined by user input
            # list possible IDs:
            electricity_price_table = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, conn=self.conn)
            PV_table = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_PV, self.conn, *["ID", "PVPower", "PVPower_unit"])
            building_table = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Building, self.conn,
                                                 *["ID", "construction_period_start", "construction_period_end",
                                                   "hwb_norm"])
            construction_period = [str(int(row["construction_period_start"])) + "-" +
                                   str(int(row["construction_period_end"])) for (i, row) in building_table.iterrows()]
            building_table = building_table.drop(columns=["construction_period_start", "construction_period_end"])
            building_table["construction_period"] = construction_period
            space_heating_table = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceHeating, self.conn,
                                                      *["ID", "TankSize", "TankSize_unit", "ID_SpaceHeatingPumpType",
                                                        "Name_SpaceHeatingPumpType"])
            space_cooling_table = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceCooling, self.conn,
                                                      *["ID", "SpaceCoolingPower", "SpaceCoolingPower_unit"])
            battery_table = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Battery, self.conn,
                                                *["ID_BatteryType", "Capacity", "Capacity_unit"])

            list_of_frames = {"ID_PV": PV_table,
                              "ID_Building": building_table,
                              "ID_SpaceHeating": space_cooling_table,
                              "ID_SpaceCooling": space_heating_table,
                              "ID_Battery": battery_table,
                              "ID_Environment": electricity_price_table}

            # get the user input:
            self.choose_household_tkinter(list_of_frames)
            # save user input to sqlite
            test_IDs_df = pd.DataFrame.from_dict(self.id_chosen, orient="index", columns=["ID"]).reset_index()  # convert dict to Dataframe
            DB().write_DataFrame(table=test_IDs_df,
                                 table_name=self.test_house,
                                 conn=self.conn,
                                 column_names=None)
            print(f"{self.test_house} saved to DB")

        return test_IDs_df


class TestingModelModes:
    """if new_id = False (default) the IDs are loaded from the database if they exist
        if new_id = True the IDs will be newly generated through user input"""
    def __init__(self, new_id: bool = False):
        self.conn = DB().create_Connection(CONS().RootDB)
        self.testing_ids = HouseholdToTest().define_household_IDs(new_id)


    def extract_Result2Array(self, result_DictValues) -> np.array:
        """extracts the results from the pyomo instance and returns the results as np.array"""
        result_array = np.nan_to_num(np.array(list(result_DictValues.values()), dtype=np.float), nan=0)
        return result_array

    # DB().read_DataFrame(table_name=REG_Table.Gen_OBJ_ID_Household,
    #                     conn=self.conn,
    #                     ID_Building= ,
    #                     ID_SpaceHeating= ,
    #                     ID_SpaceCooling= ,
    #                     ID_PV= ,
    #                     ID_Battery= )

    def run_reference(self) -> np.array:
        """runs the reference calculation and returns the hourly values as numpy array"""

    def run_optimization(self):
        """runs the optimization and returns the hourly results as pyomo instance opject"""
        # model input data
        DC = DataCollector()
        Opt = pyo.SolverFactory("gurobi")
        input_data_initial = DataSetUp().get_input_data(0, 0)
        initial_parameters = input_data_initial["input_parameters"]
        # create the instance once:
        model = create_abstract_model()
        pyomo_instance = model.create_instance(data=initial_parameters)

    def plot_comparison_results(self):
        """visualizes the results from both modes for the whole year and hourly for certain weeks"""


if __name__ == "__main__":
    HouseholdToTest().define_household_IDs()
