
import os
from pathlib import Path

class CONS:

    def __init__(self):
        self.ProjectPath = Path(os.path.dirname(__file__)).parent.parent
        self.DatabasePath = str(self.ProjectPath) + "\_Database\\"
        self.FiguresPath = str(self.ProjectPath) + "\_Figures\\"
        self.RootDB = "NewTRENDs"

if __name__ == "__main__":
    print(CONS().ProjectPath)
