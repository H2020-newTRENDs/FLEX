import numpy as np
import pyomo.environ as pyo

from flex.kit import get_logger
from flex_community.scenario import CommunityScenario

logger = get_logger(__name__)


class CommunityModel:

    def __init__(self, scenario: "CommunityScenario"):
        self.scenario = scenario
        self.hours = 8760

    def solve_aggregator_p2p_trading(self):
        self.p2p_trading = self.scenario.community_p2p_trading
        sell_price = self.scenario.electricity_price * self.scenario.aggregator_sell_price_factor
        buy_price = self.scenario.feed_in_price * self.scenario.aggregator_buy_price_factor
        self.p2p_trading_profit = np.sum(self.p2p_trading * (sell_price - buy_price))
        logger.info(f"p2p_profit: {self.p2p_trading_profit:.2f}")

    def solve_aggregator_optimization(self, instance):
        instance = self.config_instance(instance)
        pyo.SolverFactory("gurobi").solve(instance, tee=False)
        logger.info(f"opt_profit: {instance.opt_profit_rule():.2f}")
        return instance

    def config_instance(self, instance):
        self.config_params(instance)
        self.config_boundaries(instance)
        return instance

    def config_params(self, instance):
        instance.battery_charge_efficiency = self.scenario.aggregator_battery_charge_efficiency
        instance.battery_discharge_efficiency = self.scenario.aggregator_battery_discharge_efficiency
        for t in range(1, self.hours + 1):
            instance.buy_price[t] = self.scenario.feed_in_price[t - 1] * self.scenario.aggregator_buy_price_factor
            instance.sell_price[t] = self.scenario.electricity_price[t - 1] * self.scenario.aggregator_sell_price_factor

    def config_boundaries(self, instance):
        for t in range(1, 8761):
            instance.battery_charge[t].setub(self.scenario.community_grid_feed[t - 1])
            instance.battery_discharge[t].setub(self.scenario.community_grid_consumption[t - 1])
            if self.scenario.aggregator_household_battery_control == 0:
                instance.battery_soc[t].setub(self.scenario.aggregator_battery_size)
            elif self.scenario.aggregator_household_battery_control == 1:
                total_battery_size = self.scenario.aggregator_battery_size + self.scenario.community_battery_size
                instance.battery_soc[t].setub(total_battery_size[t - 1])
            else:
                logger.error(f'ValueError: CommunityScenario.aggregator_household_battery_control = '
                             f'{self.scenario.aggregator_household_battery_control}, which should be 0 or 1.')

