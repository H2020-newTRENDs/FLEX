
import os
from pathlib import Path

class CONS:

    def __init__(self):

        # -----------
        # Model Paths
        # -----------
        self.ProjectPath = Path(os.path.dirname(__file__)).parent
        self.FiguresPath = self.ProjectPath / Path("_Figures")

        # Songmin
        # self.DatabasePath = Path("C:/Users/yus/Dropbox/Academic/Models/ProsumagerDatabase")
        # self.Name = "_Songmin.sqlite"

        # Philipp
        # self.DatabasePath = Path("C:/Users/mascherbauer/Dropbox/ProsumagerDatabase")  # Dropbox
        self.DatabasePath = Path("C:/Users/mascherbauer/OneDrive/EEG_Projekte/NewTrends/Backup")  # Onedrive
        self.Name = "_Philipp.sqlite"
        self.RootDB = Path("ProsumagerUpdated" + self.Name)

        # Thomas
        # self.DatabasePath = Path("C:/Users/thoma/Dropbox/ProsumagerDatabase")
        # self.Name = "_Thomas.sqlite"
        # self.RootDB = Path("ProsumagerUpdated" + self.Name)
        # self.RootDB = Path("ProsumagerUpdated")

        # Root for TU Server:
        # self.DatabasePath = Path("/home/users/pmascherbauer/projects2/NewTrends_PM")
        # self.Name = "_Philipp.sqlite"
        # self.RootDB = Path("ProsumagerUpdated" + self.Name)

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