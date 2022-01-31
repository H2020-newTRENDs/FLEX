
import os

import numpy as np

from config import project_config
from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.models.operation.opt import OptOperationModel



household = AbstractScenario(scenario_id=0)
# for loop:
model = OptOperationModel(household)

print(model.scenario.spaceheatingtank_class.loss)
print(model.scenario.hotwatertank_class.loss)

