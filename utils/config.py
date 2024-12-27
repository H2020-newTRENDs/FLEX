from pathlib import Path
import copy

class Config:

    def __init__(self, project_name: str, project_path: Path):
        self.root = Path().resolve()
        self.project_name: str = project_name
        self.project_path: Path = project_path
        self.input: Path = self.create_folder("input")
        self.output: Path = self.create_folder("output")
        self.figure: Path = self.create_folder("output/figure")
        self.sqlite_path: Path = self.output / f"{self.project_name}.sqlite"
        self.task_id = None
        self.task_output = None

    def create_folder(self, path: str):
        path = self.project_path / path
        path.mkdir(exist_ok=True, parents=True)
        return path

    def set_task_id(self, task_id: int) -> "Config":
        self.task_id = task_id
        self.task_output = self.output / f"task_{task_id}"
        self.task_output.mkdir(exist_ok=True, parents=True)
        return self

    def make_copy(self) -> "Config":
        return copy.deepcopy(self)

