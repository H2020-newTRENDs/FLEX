import os
from pathlib import Path

class Config:

    def __init__(self, project_name: str, project_path: str):
        self.root = os.path.abspath(os.curdir)
        self.project_name: str = project_name
        self.project_path: str = project_path
        self.input = Path(self.create_folder("input"))
        self.output = Path(self.create_folder("output"))
        self.figure = Path(self.create_folder("output/figure"))
        self.task_id = None
        self.task_output = None

    def create_folder(self, path: str):
        path = os.path.join(self.project_path, path)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def set_task_id(self, task_id: int) -> "Config":
        self.task_id = task_id
        self.task_output = os.path.join(self.output, f"task_{task_id}")
        if not os.path.exists(self.task_output):
            os.makedirs(self.task_output)
        return self

    def make_copy(self) -> "Config":
        rk = self.__class__(self.project_name, self.project_path)
        for k, v in self.__dict__.items():
            rk.__dict__[k] = v
        return rk

