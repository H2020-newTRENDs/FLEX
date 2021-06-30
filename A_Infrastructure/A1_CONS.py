
import os
from pathlib import Path

class CONS:

    def __init__(self):

        # -----------
        # Model Paths
        # -----------
        self.ProjectPath = Path(os.path.dirname(__file__)).parent
        self.FiguresPath = str(self.ProjectPath) + "\_Figures\\"
        self.DatabasePath = "C:\\Users\yus\Dropbox\Academic\Models\ProsumagerDatabase\\"
        self.Name = "_Songmin"
        # self.DatabasePath = "C:\\Users\\mascherbauer\\Dropbox\\ProsumagerDatabase\\"
        # self.Name = "_Philipp"
        # self.DatabasePath = "C:\\Users\\thoma\Dropbox\ProsumagerDatabase\\"
        # self.Name = "_Thomas"
        self.RootDB = "ProsumagerUpdated" + self.Name
        # self.RootDB = "ProsumagerUpdated"

        # -----
        # Color
        # -----
        self.red = "#F47070"
        self.blue = "#8EA9DB"
        self.green = '#088A29'
        self.yellow = '#FFBF00'
        self.grey = '#C9C9C9'
        self.pink = '#FA9EFA'
        self.dark_green = '#375623'
        self.dark_blue = '#0404B4'
        self.purple = '#AC0CB0'
        self.turquoise = '#3BE0ED'
        self.dark_red = '#c70d0d'
        self.dark_grey = '#2c2e2e'
        self.light_brown = '#db8b55'
        self.black = "#000000"
        self.red_pink = "#f75d82"
        self.brown = '#A52A2A'
        self.orange = '#FFA500'