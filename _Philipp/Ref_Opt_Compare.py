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
from C_Model_Operation.C4_OptimizationModel import DataSetUp, create_abstract_model, update_instance, \
    create_instances2solve
from C_Model_Operation.C4_ReferenceModel import no_SEMS
from C_Model_Operation.C2_DataCollector import DataCollector


def check_dict_exists(table_name) -> bool:
    """checks if the dict exists in sqlite db. returns True if table exists and False if not"""
    c = DB().create_Connection(CONS().RootDB).cursor()
    # get the count of tables with the name
    exists = pd.read_sql(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND NAME ='{}'".format(table_name),
        con=DB().create_Connection(CONS().RootDB))
    # if the count is 1, then table exists
    if exists.iloc[0].iloc[0] == 1:
        print('Table exists.')
        # close the connection
        return True
    else:
        print('Table does not exist.')
        # close the connection
        return False


class DefineTestHouse:
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
            # check if frame has less than 3 columns:
            if frame.shape[1] < 3:
                frame[""] = ""  # appending an empty column at the end
            else:
                pass
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
            # hit enter instead of button
            root.bind('<Return>', (lambda event: hit_enter()))

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

    def define_test_household_IDs(self, new_id: bool = True) -> pd.DataFrame:
        """defines the household used for the comparison and returns the IDs as Dataframe
        if new_id = True (default) the IDs will be newly generated through user input
        if new_id = False the IDs are loaded from the database if they exist"""
        # check if sqlite file exists and new ID = False:
        if check_dict_exists(self.test_house) and not new_id:
            # read the dictionary from database
            test_IDs_df = DB().read_DataFrame(table_name=self.test_house, conn=self.conn)
        else:  # household IDs will be determined by user input
            # list possible IDs:
            electricity_price_table = DB().read_DataFrame(
                REG_Table().Gen_Sce_ID_Environment,
                conn=self.conn
            )
            PV_table = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_PV,
                                           self.conn,
                                           *[REG_Var().ID_PV, "PVPower", "PVPower_unit"])
            building_table = DB().read_DataFrame(
                REG_Table().Gen_OBJ_ID_Building,
                self.conn,
                *["ID", "construction_period_start", "construction_period_end", "hwb_norm"]
            )
            construction_period = [str(int(row["construction_period_start"])) + "-" +
                                   str(int(row["construction_period_end"])) for (i, row) in building_table.iterrows()]
            building_table = building_table.drop(columns=["construction_period_start", "construction_period_end"])
            building_table["construction_period"] = construction_period

            space_heating_table = DB().read_DataFrame(
                REG_Table().Gen_OBJ_ID_SpaceHeatingSystem,
                self.conn,
                *[REG_Var().ID_SpaceHeatingSystem,  "Name_SpaceHeatingPumpType"]
            )
            space_heating_tank_table = DB().read_DataFrame(
                REG_Table().Gen_OBJ_ID_SpaceHeatingTank,
                self.conn,
                *[REG_Var().ID_SpaceHeatingTank, "TankSize", "TankSize_unit"]
            )
            DHW_tank_table = DB().read_DataFrame(
                REG_Table().Gen_OBJ_ID_DHW_tank,
                self.conn,
                *[REG_Var().ID_DHWTank, "DHWTankSize", "DHWTankSize_unit"]
            )
            space_cooling_table = DB().read_DataFrame(
                REG_Table().Gen_OBJ_ID_SpaceCooling,
                self.conn,
                *[REG_Var().ID_SpaceCooling, "SpaceCoolingPower", "SpaceCoolingPower_unit"]
            )
            battery_table = DB().read_DataFrame(
                REG_Table().Gen_OBJ_ID_Battery,
                self.conn,
                *[REG_Var().ID_Battery, "Capacity", "Capacity_unit"]
            )
            age_group_table = DB().read_DataFrame(
                REG_Table().ID_AgeGroup,
                self.conn
            )

            list_of_frames = {REG_Var().ID_PV: PV_table,
                              REG_Var().ID_Building: building_table,
                              REG_Var().ID_SpaceHeatingSystem: space_heating_table,
                              REG_Var().ID_SpaceHeatingTank: space_heating_tank_table,
                              REG_Var().ID_DHWTank: DHW_tank_table,
                              REG_Var().ID_SpaceCooling: space_cooling_table,
                              REG_Var().ID_Battery: battery_table,
                              REG_Var().ID_AgeGroup: age_group_table,
                              REG_Var().ID_Environment: electricity_price_table}

            # get the user input:
            self.choose_household_tkinter(list_of_frames)
            # save user input to sqlite
            test_IDs_df = pd.DataFrame.from_dict(self.id_chosen, orient="index",
                                                 columns=["ID"]).reset_index()  # convert dict to Dataframe
            DB().write_DataFrame(table=test_IDs_df,
                                 table_name=self.test_house,
                                 conn=self.conn,
                                 column_names=None)
            print(f"{self.test_house} saved to DB")

        return test_IDs_df


class ModelModes:
    def __init__(self):
        self.conn = DB().create_Connection(CONS().RootDB)
        self.table_name_reference_yearly = "test_results_reference_yearly"
        self.table_name_reference_hourly = "test_results_reference_hourly"
        self.table_name_optimization_yearly = "test_results_optimization_yearly"
        self.table_name_optimization_hourly = "test_results_optimization_hourly"


class RunModelModes(ModelModes):
    """if new_id = True (default) the IDs will be newly generated through user input and the
        data will be newly generated and saved to the DB
        if new_id = False the IDs are loaded from the database if they exist as well as the results"""

    def __init__(self, new_id: bool = True):
        super().__init__()
        self.new_id = new_id
        self.testing_ids = DefineTestHouse().define_test_household_IDs(new_id).set_index("index", drop=True)
        self.Gen_OBJ_ID_Household = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Household, self.conn)
        self.Gen_Sce_ID_Environment = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, self.conn)

    def get_row_ids_optimization(self) -> (int, int):
        """gets the row id for the specific testing_ids in the Gen_OBJ_ID_Household frame and
        the row id for the environment ID"""
        single_row_id = self.Gen_OBJ_ID_Household.loc[
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_PV] ==
                             self.testing_ids.loc[REG_Var().ID_PV].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_Building] ==
                             self.testing_ids.loc[REG_Var().ID_Building].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_SpaceHeatingSystem] ==
                             self.testing_ids.loc[REG_Var().ID_SpaceHeatingSystem].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_SpaceHeatingTank] ==
                             self.testing_ids.loc[REG_Var().ID_SpaceHeatingTank].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_DHWTank] ==
                             self.testing_ids.loc[REG_Var().ID_DHWTank].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_SpaceCooling] ==
                             self.testing_ids.loc[REG_Var().ID_SpaceCooling].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_Battery] ==
                             self.testing_ids.loc[REG_Var().ID_Battery].values[0]) &
                            (self.Gen_OBJ_ID_Household.loc[:, REG_Var().ID_AgeGroup] ==
                             self.testing_ids.loc[REG_Var().ID_AgeGroup].values[0])
                            ]["ID"].values[0] - 1  # minus 1 because counting starts at 0

        single_environment_id = self.Gen_Sce_ID_Environment.loc[
                                    self.Gen_Sce_ID_Environment.loc[:, "ID"] ==
                                    self.testing_ids.loc["ID_Environment"].values[0]
                                    ]["ID"].values[0] - 1  # minus 1 because counting starts at 0
        return single_row_id, single_environment_id

    def get_row_ids_reference(self) -> (int, int):
        """gets the row id for the specific testing_ids in the REDUCED Gen_OBJ_ID_Household frame and
        the row id for the environment ID"""
        reduced_gen_obj_id_household = no_SEMS().remove_household_IDs()
        single_row_id = reduced_gen_obj_id_household.loc[
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_PV] ==
                             self.testing_ids.loc[REG_Var().ID_PV].values[0]) &
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_SpaceHeatingSystem] ==
                             self.testing_ids.loc[REG_Var().ID_SpaceHeatingSystem].values[0]) &
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_SpaceHeatingTank] ==
                             self.testing_ids.loc[REG_Var().ID_SpaceHeatingTank].values[0]) &
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_DHWTank] ==
                             self.testing_ids.loc[REG_Var().ID_DHWTank].values[0]) &
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_SpaceCooling] ==
                             self.testing_ids.loc[REG_Var().ID_SpaceCooling].values[0]) &
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_Battery] ==
                             self.testing_ids.loc[REG_Var().ID_Battery].values[0]) &
                            (reduced_gen_obj_id_household.loc[:, REG_Var().ID_AgeGroup] ==
                             self.testing_ids.loc[REG_Var().ID_AgeGroup].values[0])
                            ]["ID"].values[0] - 1  # minus 1 because counting starts at 0

        single_environment_id = self.Gen_Sce_ID_Environment.loc[
                                    self.Gen_Sce_ID_Environment.loc[:, "ID"] ==
                                    self.testing_ids.loc["ID_Environment"].values[0]
                                    ]["ID"].values[0] - 1  # minus 1 because counting starts at 0
        return single_row_id, single_environment_id

    def run_reference(self) -> (np.array, np.array):
        """runs the reference calculation and returns the hourly values as numpy array"""
        household_row_id, environment_row_id = self.get_row_ids_reference()
        yearly_results_all_ids, hourly_results_all_ids = no_SEMS().calculate_noDR(household_row_id, environment_row_id)
        # ref model calculates for all building IDs at the same time -> take out single building ID
        yearly_results = yearly_results_all_ids[yearly_results_all_ids[:, 1] ==
                                                str(self.testing_ids.loc[REG_Var().ID_Building].values[0]),
                         :]  # column 1 is building ID
        hourly_results = hourly_results_all_ids[hourly_results_all_ids[:, 1] ==
                                                str(self.testing_ids.loc[REG_Var().ID_Building].values[0]), :]
        return yearly_results, hourly_results

    def run_optimization(self) -> (np.array, np.array):
        """runs the optimization and returns the yearly and hourly results as numpy array"""
        # model input data
        DC = DataCollector()
        Opt = pyo.SolverFactory("gurobi")
        # get household and environment row ids or set them new
        household_row_id, environment_row_id = self.get_row_ids_optimization()
        input_data = DataSetUp().get_input_data(household_row_id, environment_row_id)
        initial_parameters = input_data["input_parameters"]
        # create the instance once:
        model = create_abstract_model()
        pyomo_instance = model.create_instance(data=initial_parameters)
        instance2solve = update_instance(input_data, pyomo_instance)
        # solve model
        result = Opt.solve(instance2solve, tee=False)
        yearly_results, hourly_results = DC.collect_OptimizationResult(
            input_data["Household"], input_data["Environment"], instance2solve)
        return yearly_results, hourly_results

    def get_results_both_modes(self) -> (np.array, np.array, np.array, np.array):
        """returns the hourly and yearly results of the reference and the optimization mode as numpy arrays
        results are returned as follows:
        yearly-reference, hourly-reference, yearly-optimization, hourly-optimization"""
        if self.new_id:
            # calculate the results, save and return them
            yearly_results_reference, hourly_results_reference = self.run_reference()
            yearly_results_optimization, hourly_results_optimization = self.run_optimization()
            # save
            self.save_results(yearly_results_reference, hourly_results_reference, yearly_results_optimization,
                              hourly_results_optimization)
            return yearly_results_reference, hourly_results_reference, \
                   yearly_results_optimization, hourly_results_optimization
        else:
            # if new_id is False, check if results are in DB:
            # if yes, load them and return them,
            if check_dict_exists(self.table_name_optimization_hourly) \
                    and check_dict_exists(self.table_name_optimization_yearly) \
                    and check_dict_exists(self.table_name_reference_hourly) \
                    and check_dict_exists(self.table_name_reference_yearly):
                # load results
                yearly_results_reference = DB().read_DataFrame(self.table_name_reference_yearly, self.conn)
                hourly_results_reference = DB().read_DataFrame(self.table_name_reference_hourly, self.conn)
                yearly_results_optimization = DB().read_DataFrame(self.table_name_optimization_yearly, self.conn)
                hourly_results_optimization = DB().read_DataFrame(self.table_name_optimization_hourly, self.conn)
                return yearly_results_reference, hourly_results_reference, \
                       yearly_results_optimization, hourly_results_optimization

            # if no, calculate them, save them and return them:
            else:
                # calculate
                yearly_results_reference, hourly_results_reference = self.run_reference()
                yearly_results_optimization, hourly_results_optimization = self.run_optimization()
                # save
                self.save_results(yearly_results_reference, hourly_results_reference, yearly_results_optimization,
                                  hourly_results_optimization)
                # return
                return yearly_results_reference, hourly_results_reference, \
                       yearly_results_optimization, hourly_results_optimization

    def save_results(self, yearly_results_reference, hourly_results_reference, yearly_results_optimization,
                     hourly_results_optimization):
        """save results to DB (mainly for developing the plots so I dont have to wait for every run to be finished)"""
        # reference year
        DB().write_DataFrame(table=yearly_results_reference,
                             table_name=self.table_name_reference_yearly,
                             column_names=DataCollector(self.conn).SystemOperationYear_Column.keys(),
                             conn=self.conn,
                             dtype=DataCollector(self.conn).SystemOperationYear_Column)
        # reference hourly
        DB().write_DataFrame(table=hourly_results_reference,
                             table_name=self.table_name_reference_hourly,
                             column_names=DataCollector(self.conn).SystemOperationHour_Column.keys(),
                             conn=self.conn,
                             dtype=DataCollector(self.conn).SystemOperationHour_Column)
        # optimization year
        DB().write_DataFrame(table=np.array(yearly_results_optimization).reshape(1, -1),
                             table_name=self.table_name_optimization_yearly,
                             column_names=DataCollector(self.conn).SystemOperationYear_Column.keys(),
                             conn=self.conn,
                             dtype=DataCollector(self.conn).SystemOperationYear_Column)
        # optimization hourly
        DB().write_DataFrame(table=hourly_results_optimization,
                             table_name=self.table_name_optimization_hourly,
                             column_names=DataCollector(self.conn).SystemOperationHour_Column.keys(),
                             conn=self.conn,
                             dtype=DataCollector(self.conn).SystemOperationHour_Column)


class CompareModelModes(ModelModes):
    def __init__(self, new_id: bool = False):
        super().__init__()
        self.new_id = new_id
        yearly_results_reference, hourly_results_reference, yearly_results_optimization, hourly_results_optimization = \
            RunModelModes(new_id=self.new_id).get_results_both_modes()
        self.yearly_results_reference = yearly_results_reference
        self.hourly_results_reference = hourly_results_reference
        self.yearly_results_optimization = yearly_results_optimization
        self.hourly_results_optimization = hourly_results_optimization

    def plot_comparison_results(self):
        """visualizes the results from both modes for the whole year and hourly for certain weeks"""

        a = 1


if __name__ == "__main__":
    CompareModelModes(new_id=True).plot_comparison_results()
