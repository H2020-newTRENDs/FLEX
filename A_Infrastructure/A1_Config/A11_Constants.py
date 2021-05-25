
import os
from pathlib import Path

class CONS:

    def __init__(self):
        self.ProjectPath = Path(os.path.dirname(__file__)).parent.parent

        self.DatabasePath = "C:\\Users\yus\Dropbox\Academic\Models\ProsumagerDatabase\\"
        self.Name = "_Songmin"
        # self.DatabasePath = "C:\\Users\\mascherbauer\\Dropbox\\ProsumagerDatabase\\"
        # self.Name = "_Philipp"
        # self.DatabasePath = "C:\\Users\\thoma\Dropbox\ProsumagerDatabase\\"
        #self.Name = "_Thomas"

        self.FiguresPath = str(self.ProjectPath) + "\_Figures\\"
        self.RootDB = "ProsumagerUpdated" + self.Name
        # self.RootDB = "ProsumagerUpdated"


