
import os
from pathlib import Path

class CONS:

    def __init__(self):
        self.ProjectPath = Path(os.path.dirname(__file__)).parent.parent
        # self.DatabasePath = str(self.ProjectPath) + "\_Database\\"
        # self.DatabasePath = "C:\\Users\yus\Dropbox\Academic\Models\ProsumagerDatabase\\"
        self.DatabasePath = "C:\\Users\\mascherbauer\\Dropbox\\ProsumagerDatabase"
        # @Philipp, @Thomas: you can paste your paths here.
        # We do not to ignore this file then. Everytime we just uncomment our own address and run locally.

        self.FiguresPath = str(self.ProjectPath) + "\_Figures\\"
        self.RootDB = "Prosumager"

if __name__ == "__main__":
    print(CONS().ProjectPath)
