import itertools
import os
from typing import TYPE_CHECKING, List, Dict
from pathlib import Path
import numpy as np

import pandas as pd
import sqlalchemy
from utils.tables import InputTables
if TYPE_CHECKING:
    from flex.config import Config


class DB:

    def __init__(self, path):
        self.engine = sqlalchemy.create_engine(f'sqlite:///{path}')
        self.metadata = sqlalchemy.MetaData()
        self.metadata.reflect(bind=self.engine)

    def if_exists(self, table_name: str) -> bool:
        return table_name in self.get_table_names()

    def if_parquet_exists(self, table_name: str, scenario_ID: int) -> bool:
        file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
        path_to_file = self.config.output / file_name
        return path_to_file.exists()

    def get_engine(self):
        return self.engine

    def dispose(self):
        self.engine.dispose()

    def get_table_names(self):
        return sqlalchemy.inspect(self.engine).get_table_names()

    def clear_database(self):
        for table_name in self.get_table_names():
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(f"drop table {table_name}"))

    def drop_table(self, table_name: str):
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(f"drop table if exists {table_name}"))

    def write_dataframe(
            self,
            table_name: str,
            data_frame: pd.DataFrame,
            data_types: dict = None,
            if_exists="append",
    ):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(
            table_name,
            self.engine,
            index=False,
            dtype=data_types,
            if_exists=if_exists,
            chunksize=10_000,
        )
    
    
    def write_to_parquet(self, table_name: str, scenario_ID: int, table: pd.DataFrame, folder: Path) -> None:
        """
        is used to save the results. Writes the dataframe into a parquet file with 'table_name_ID' as file name
        """
        file_name = f"{table_name}_S{scenario_ID}.parquet.gzip"
        saving_path = folder / file_name
        table.to_parquet(path=saving_path, engine="auto", compression='gzip', index=False)

    def read_parquet(self, table_name: str, scenario_ID: int, folder: Path, column_names: List[str] = None) -> pd.DataFrame:
        """
        Returns: dataframe containing the results for the table name and specific scenario ID
        """
        file_name = f"{table_name}_S{scenario_ID}.parquet.gzip"
        path_to_file = folder / file_name
        if column_names:
            df = pd.read_parquet(path=path_to_file, engine="auto", columns=column_names)
        else:
            df = pd.read_parquet(path=path_to_file, engine="auto")
        return df

    def read_dataframe(self, table_name: str, filter: dict = None, column_names: List[str] = None) -> pd.DataFrame:
        """Reads data from a database table with optional filtering and column selection.

                Args:
                    table_name (str): Name of the table to query.
                    filter (dict, optional): Dictionary with {column_name: value} to filter the data.
                    column_names (list of str, optional): List of column names to extract.

                Returns:
                    pd.DataFrame: Resulting dataframe.
                """
        table = self.metadata.tables[table_name]

        if column_names:
            if len(column_names) > 1:
                query = sqlalchemy.select(*[table.columns[name] for name in column_names])
            else:
                query = sqlalchemy.select([table.columns[name] for name in column_names][0])
        else:
            query = sqlalchemy.select(table)

        if filter:
            for key, value in filter.items():
                query = query.where(table.columns[key] == value)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def delete_row_from_table(
            self,
            table_name: str,
            column_name_plus_value: dict
    ) -> None:
        condition_temp = ""
        for key, value in column_name_plus_value.items():
            condition_temp += key + " == '" + str(value) + "' and "
        condition = " where " + condition_temp
        condition = condition[0:-5]  # deleting last "and"

        query = f"DELETE FROM {table_name}" + condition
        with self.engine.connect() as connection:
            connection.execute(sqlalchemy.text(query))

    def query(self, sql) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)



def create_db_conn(config: "Config") -> DB:
    if config.task_id is None:
        conn = DB(config.output / f"{config.project_name}.sqlite")
    else:
        conn = DB(os.path.join(config.task_output, f'{config.project_name}.sqlite'))
    return conn


def init_project_db(config: "Config"):
    db = create_db_conn(config)
    db.clear_database()

    def file_exists(table_name: str):
        df = None
        extensions = {".xlsx": pd.read_excel,
                      ".csv": pd.read_csv}
        for ext, pd_read_func in extensions.items():
            file_path = os.path.join(config.input, table_name + ext)
            if os.path.exists(file_path):
                df = pd_read_func(file_path)
                break
        return df
    
    for input_table in InputTables:
        df = file_exists(input_table.name)
        if df is not None:
            print(f'Loading input table --> {input_table.name}')
            db.write_dataframe(
                table_name=input_table.name,
                data_frame=df.dropna(axis=1, how="all").dropna(axis=0, how="all"),
                if_exists="replace"
            )
    if file_exists(InputTables.OperationScenario.name) is None:
        print(f"Operation Scenario Table was not provided. Creating Operation Scenario Table through permutations.")
        component_table_names = {}
        for input_table in InputTables:
            if "operation" in input_table.name.lower() and "component" in input_table.name.lower():
                component_table_names[f"ID_{input_table.name.split('_')[-1]}"] = input_table.name
        generate_operation_scenario_table(database=db, 
                                          scenario_table_name=InputTables.OperationScenario.name, 
                                          component_table_names=component_table_names,
                                          exclude_from_permutation=None,
                                          start_scenario_table=None)


def fetch_input_tables(config: "Config") -> Dict[str, pd.DataFrame]:
    input_tables = {}
    db = create_db_conn(config)
    for table_name in db.get_table_names():
        input_tables[table_name] = db.read_dataframe(table_name)
    return input_tables


def generate_params_combination_df(params_values: Dict[str, List[int]],
                                    excluded_keys: List[str] = None,
                                    start_scenario_table: pd.DataFrame = None) -> pd.DataFrame:
    """
    The function creates a permutation of all the ID vectors in the params_values dictionary values. If excluded
    keys are provided, these IDs will not be included in the permutation. For example if you have already pre-
    defined the building parameters and you know which building IDs will have a battery storage you should exclude
    building IDs and Battery IDs from the permutation.
    :param start_scenario_table: This table should contain the ID names as column names of the IDs that are
            excluded from the permutation. The table is the starting point of the permutations. All columns
            within the table will be treated as a single value. We need this for example when we try to
            have different PV systems on different building types. Then the table should contain ID_Building and
            the corresponding ID_PV.
    :param params_values: a dictionary containing the ID name as key and the IDs as list as values.
    :param excluded_keys: a list of ID names which should be excluded from the permutation. At least 2!
    :return: pandas DataFrame as Scenario ID table which has the Component IDs as column names
    """
    if excluded_keys:
        # check if at least 2 excluded keys were provided:
        assert len(excluded_keys) > 1, "at least 2 ID names have to be provided in the excluded key list"
        # check if the exclude keys have the same length,
        excluded_values = [params_values[k] for k in excluded_keys]
        if start_scenario_table is not None:
            start_scenario_table = start_scenario_table[excluded_keys]  # sort
            excluded_lists = start_scenario_table[excluded_keys].values.tolist()
        else:
            if not all(len(element) == len(excluded_values[0]) for element in excluded_values):
                assert "values of excluded keys provided dont have the same length"
            # create a list of the excluded lists
            excluded_lists = list(map(list, zip(*excluded_values)))
        # define the first key and its value which is used in the permutation:
        first_excluded_key = excluded_keys[0]
        # create new dictionary where the excluded list is the first element of the first excluded key
        new_dict = {first_excluded_key: excluded_lists}
        for key, values in params_values.items():
            if key not in excluded_keys:
                new_dict[key] = values

        keys, values = zip(*new_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame(permutations_dicts)
        # expand the column with the excluded values with their key names:
        df[excluded_keys] = df[first_excluded_key].apply(pd.Series)
    else:
        keys, values = zip(*params_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        df = pd.DataFrame(permutations_dicts)
    return df


def generate_operation_scenario_table(
        database,
        scenario_table_name: str,
        component_table_names: Dict[str, str],
        exclude_from_permutation: List[str] = None,
        start_scenario_table: pd.DataFrame = None
        ) -> None:
    """
    Creates the Scenario Table and saves it to the sqlite database.
    :param exclude_from_permutation: a list of ID names (eg. ID_Boiler) that should be excluded from permutation
    :param scenario_table_name: Name of the scenario table, default is OperationTable.Scenarios
    :param component_table_names: dictionary containing the ID name (eg. ID_Boiler) and the Table name (eg.
            OperationScenario_Component_Boiler)
    """
    database.drop_table(scenario_table_name)
    scenario_df = generate_params_combination_df(
        params_values=get_component_scenario_ids(database, component_table_names),
        excluded_keys=exclude_from_permutation,
        start_scenario_table=start_scenario_table,
    )
    scenario_ids = np.array(range(1, 1 + len(scenario_df)))
    scenario_df.insert(loc=0, column="ID_Scenario", value=scenario_ids)
    data_types = {name: sqlalchemy.types.Integer for name in scenario_df.columns}
    database.write_dataframe(
        scenario_table_name,
        scenario_df,
        data_types=data_types,
        if_exists="replace",
    )

def get_component_scenario_ids(db, component_table_names: Dict[str, str]) -> Dict[str, List[int]]:
    component_scenario_ids = {}
    engine = db.get_engine().connect()
    for id_name, table_name in component_table_names.items():
        if engine.dialect.has_table(engine, table_name):
            component_scenario_ids[id_name] = db.read_dataframe(table_name)[id_name].unique()
    return component_scenario_ids