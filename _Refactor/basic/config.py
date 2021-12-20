import os

class Config:
    def __init__(self,
                 project_name: str,
                 project_root: str,
                 sqlite_folder: str,
                 output_folder: str
                 ):
        self.root_db_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.project_name = project_name
        self.project_root = project_root
        self.sqlite_folder = sqlite_folder
        self.output_folder = output_folder




