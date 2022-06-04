import os
from pathlib import Path
from abc import ABC


class Config(ABC):
    def __init__(self, project_name: str):
        self.project_name: str = project_name
        self.project_root: "Path" = self.setup_folder_path(os.path.dirname(__file__))
        self.input_behavior: "Path" = self.setup_folder_path(
            os.path.dirname("data/input_behavior/")
        )
        self.input_operation: "Path" = self.setup_folder_path(
            os.path.dirname("data/input_operation/")
        )
        self.input_investment: "Path" = self.setup_folder_path(
            os.path.dirname("data/input_investment/")
        )
        self.output: "Path" = self.setup_folder_path(os.path.dirname("data/output/"))
        self.fig: "Path" = self.setup_folder_path(os.path.dirname("data/figure/"))

    @staticmethod
    def setup_folder_path(folder_path) -> "Path":
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return Path(folder_path)
