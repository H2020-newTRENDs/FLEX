
import pandas as pd
import os
from pathlib import Path
from A_Infrastructure.A2_DB import DB

base_path = Path().absolute().resolve()
dynamic_data_path = Path("inputdata/AUT/001__dynamic_calc_data_bc_2017_AUT.csv")
building_segment_path = Path("inputdata/AUT/040_aut__3__BASE__1_zz_new_bc_seg__b_building_segment_sh.csv")
building_class_path = Path("inputdata/AUT/040_aut__1__BASE__0__b_new_buildings_classes.csv")

dynamic_data = pd.read_csv(Path(base_path / dynamic_data_path), sep=None, engine="python")
building_segment = pd.read_csv(Path(base_path / building_segment_path), sep=None, engine="python")
building_class = pd.read_csv(Path(base_path / building_class_path), sep=None, engine="python")


a=1
