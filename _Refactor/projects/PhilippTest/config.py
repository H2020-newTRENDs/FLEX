
import os

from _Refactor.basic.config import Config

project_config = Config(
    project_name="PhilippTest",
    project_root=os.path.dirname(__file__),
    sqlite_folder='data/sqlite',
    database_name="ProsumagerUpdated_Philipp",
    output_folder='data/output'
)
