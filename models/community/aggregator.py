import numpy as np
import pyomo.environ as pyo


class Aggregator:

    def create_opt_instance(self):
        model = self.setup_model()
        instance = model.create_instance()
        return instance

    def setup_model(self):
        m = pyo.AbstractModel()
        self.setup_sets(m)
        self.setup_params(m)
        self.setup_variables(m)
        self.setup_constraint_battery_soc(m)
        self.setup_objective(m)
        return m

    def setup_sets(self, m):
        m.t = pyo.Set(initialize=np.arange(1, 8761))

    def setup_params(self, m):
        m.battery_charge_efficiency = pyo.Param(mutable=True)
        m.battery_discharge_efficiency = pyo.Param(mutable=True)
        m.buy_price = pyo.Param(m.t, mutable=True)
        m.sell_price = pyo.Param(m.t, mutable=True)

    def setup_variables(self, m):
        m.battery_charge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.battery_discharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.battery_soc = pyo.Var(m.t, within=pyo.NonNegativeReals)

    def setup_constraint_battery_soc(self, m):
        def calc_battery_soc(m, t):
            if t == 1:
                return m.battery_soc[t] == 0 + \
                                           m.battery_charge[t] * m.battery_charge_efficiency - \
                                           m.battery_discharge[t] * m.battery_discharge_efficiency
            else:
                return m.battery_soc[t] == m.battery_soc[t - 1] + \
                                           m.battery_charge[t] * m.battery_charge_efficiency - \
                                           m.battery_discharge[t] * m.battery_discharge_efficiency
        m.battery_soc_rule = pyo.Constraint(m.t, rule=calc_battery_soc)

    def setup_objective(self, m):
        def calc_opt_profit(m):
            opt_profit = 0
            for t in m.t:
                opt_profit += m.battery_discharge[t] * m.sell_price[t] - \
                                m.battery_charge[t] * m.buy_price[t]
            return opt_profit
        m.opt_profit_rule = pyo.Objective(rule=calc_opt_profit, sense=pyo.maximize)

