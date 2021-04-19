import h5py
import numpy as np
from pathlib import Path
# from Simple_plots import *


def read_h5(filename):
    print('reading hf file...')
    dict = {}
    hf = h5py.File(filename, "r")
    # save all arrays from hf to dict as np.array:
    for key in hf.keys():
        dict[key] = np.array(hf.get(key))
    hf.close()
    print('done')
    return dict


# load the data from h5 file:
filepath = Path('outputdata')
print(filepath)
filename = 'Building_load_curve_Austria_2050.hdf5'
file = filepath / filename
dict_ = read_h5(file)


dict2 = {}
dict3 = {}
# erste februar woche wird für tests verwendet (stunde 744 bis 912)
# es wird das 1te Profil für die tests verwendet
cooling_load = dict_["Cooling_load"][1][743:911]
DHW_load = dict_["DHW_load"][1][743:911]
heating_load = dict_["Heating_load"][1][743:911]
T_set_heating = dict_["T_set_heating"][1][743:911]
T_set_cooling = dict_["T_set_cooling"][1][743:911]
T_outside = dict_["T_outside"][1][743:911]

dict2["Cooling_load"] = cooling_load
dict2["DHW_load"] = DHW_load
dict2["Heating_load"] = heating_load

dict3["T_set_heating"] = T_set_heating
dict3["T_set_cooling"] = T_set_cooling
dict3["T_outside"] = T_outside

# plot heating cooling and DHW loads as well as temperature settings and outside temp:
one_week(dict2, dict3)

# TODO verschieben einer spitzenlast manuell und neu berechnen der jeweiligen loads mit core_rc_one_step...




a = 1
