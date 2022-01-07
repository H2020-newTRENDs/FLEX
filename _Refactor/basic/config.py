import os

class Config:
    def __init__(self,
                 project_name: str,
                 project_root: str,
                 sqlite_folder: str,
                 database_name: str,
                 output_folder: str
                 ):
        self.root_db_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.root_db = os.path.join(os.path.dirname(os.path.dirname(__file__), sqlite_folder, database_name))
        self.database_name = database_name
        self.project_name = project_name
        self.project_root = project_root
        self.sqlite_folder = sqlite_folder
        self.output_folder = output_folder




