
import os
from pathlib import Path

class Paths:

    def __init__(self):

        self.ProjectPath = Path(os.path.dirname(__file__)).parent
        self.FiguresPath = str(self.ProjectPath) + "\_Figures\\"
        self.DatabasePath = "C:\\Users\yus\Dropbox\Academic\Models\ProsumagerDatabase\\"
        self.Name = "_Songmin"
        # self.DatabasePath = "C:\\Users\\mascherbauer\\Dropbox\\ProsumagerDatabase\\"
        # self.Name = "_Philipp"
        # self.DatabasePath = "C:\\Users\\thoma\Dropbox\ProsumagerDatabase\\"
        # self.Name = "_Thomas"
        self.RootDB = "ProsumagerUpdated" + self.Name