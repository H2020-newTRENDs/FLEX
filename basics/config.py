import os
from pathlib import Path
from abc import ABC


class Config(ABC):
    def __init__(self, project_name: str):
        self.project_name: str = project_name
        self.project_root: "Path" = self.setup_folder_path(Path(__file__).parent.parent)
        self.input_behavior: "Path" = self.setup_folder_path(
            self.project_root / Path("data/input_behavior/")
        )
        self.input_operation: "Path" = self.setup_folder_path(
            self.project_root / Path("data/input_operation/") / f"{self.project_name}"
        )
        self.input_community: "Path" = self.setup_folder_path(
            os.path.dirname("data/input_community/")
        )
        self.input_investment: "Path" = self.setup_folder_path(
            self.project_root / Path("data/input_investment/")
        )
        self.output: "Path" = self.setup_folder_path(self.project_root / Path(f"data/output") / f"{self.project_name}")
        self.fig: "Path" = self.setup_folder_path(self.project_root / Path(f"data/figure/{self.project_name}"))

    @staticmethod
    def setup_folder_path(folder_path) -> "Path":
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return Path(folder_path)
