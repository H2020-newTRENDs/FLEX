import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.pivot_table(data=sns.load_dataset('PVAndBat'), index = 'AgeClass', values = 'Savings', columns = 'BatterySize')
df.head
