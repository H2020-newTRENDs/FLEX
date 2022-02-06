from _Refactor.models.operation.abstract import AbstractOperationModel
from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.core.elements.rc_model import R5C1Model

import numpy as np


class RefOperationModel(AbstractOperationModel):

    def calculate_DHW_tank_energy(self, electricity_grid_demand, electricity_surplus, hot_water_demand):
        """
        calculates the usage and the energy in the domestic hot water tank. The tank is charged by the heat pump
        with the COP for hot water.
        Whenever there is surplus of PV electricity the DHW tank is charged and it is discharged by the DHW-usage.
        IF a DHW tank is utilized the energy for DHW will always be solemnly provided by the DHW tank. Therefore the
        heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses.

        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW

        """
        TankMinTemperature = self.scenario.hotwatertank_class.temperature_min
        TankMaxTemperature = self.scenario.hotwatertank_class.temperature_max
        TankSize = self.M_WaterTank_DHW
        TankSurfaceArea = self.A_SurfaceTank_DHW
        TankLoss = self.U_ValueTank_DHW
        TankSurroundingTemperature = self.T_TankSurrounding_DHW
        TankStartTemperature = self.T_TankStart_DHW
        COP_DHW = self.HotWaterHourlyCOP

        electricity_surplus_after_DHW = electricity_surplus  # this gets altered through the calculation and returned
        electricity_deficit = np.zeros(electricity_surplus.shape)  # extra electricity needed to keep above min temp
        grid_demand_after_DHW = electricity_grid_demand  # gets altered through  the calculation TODO should be grid demand - surplus + deficit, check!
        Q_DHWTank_in = np.zeros(electricity_surplus.shape)  # hot water charged into the tank
        Q_DHWTank_out = np.zeros(electricity_surplus.shape)  # hot water provided by tank
        Q_HP_DHW = np.copy(hot_water_demand)  # hot water produced by HP
        CurrentTankTemperature = np.zeros(electricity_surplus.shape)
        TankLossHourly = np.zeros(electricity_surplus.shape)

        for i, element in enumerate(electricity_surplus_after_DHW):
            if i == 0:
                # calculate hourly temperature loss:
                TankLossHourly[i] = (TankStartTemperature - TankSurroundingTemperature) * \
                                                TankSurfaceArea * TankLoss  # W
                # surplus of PV electricity is used to charge the tank:
                CurrentTankTemperature[i] = TankStartTemperature + \
                                                        (electricity_surplus[i] *
                                                         COP_DHW[i] - TankLossHourly[i]) / \
                                                        (TankSize * self.cp_water)
                # Q_HP_DHW is increased:
                Q_HP_DHW[i] = electricity_surplus[i] * COP_DHW[i] + \
                                          hot_water_demand[i]
                Q_DHWTank_in[i] = electricity_surplus[i] * COP_DHW[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if CurrentTankTemperature[i] > TankMaxTemperature:
                    electricity_surplus_after_DHW[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * (
                                                                           TankSize * self.cp_water) / COP_DHW[i]  # W
                    CurrentTankTemperature[i] = TankMaxTemperature
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP_DHW[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                              (TankSize * self.cp_water) + hot_water_demand[i]
                    Q_DHWTank_in[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                                  (TankSize * self.cp_water)

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if CurrentTankTemperature[i] <= TankMinTemperature:
                    electricity_deficit[i] = (TankMinTemperature - CurrentTankTemperature[
                        i]) * (TankSize * self.cp_water) / COP_DHW[i]
                    CurrentTankTemperature[i] = TankMinTemperature
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_DHW[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP_DHW[i] = electricity_deficit[i] * COP_DHW[i] + hot_water_demand[i]
                    Q_DHWTank_in[i] = electricity_deficit[i] * COP_DHW[i]

                # if there is energy in the tank it will be used for heating:
                if CurrentTankTemperature[i] > TankMinTemperature:
                    EnergyInTank = (CurrentTankTemperature[i] - TankMinTemperature) * (
                            TankSize * self.cp_water)

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if electricity_grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if electricity_grid_demand[i] > hot_water_demand[i] / COP_DHW[i]:  # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP_DHW[i] = 0  # DHW is provided by tank
                                Q_DHWTank_out[i] = hot_water_demand[i]
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[
                                    i]
                                CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP_DHW[i] = DeficitEnergy  # dhw partly provided by tank
                                Q_DHWTank_out[i] = EnergyInTank
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[
                                    i]
                                CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if electricity_surplus[i] <= hot_water_demand[i] / COP_DHW[i]:
                            # calculate the part that can be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / COP_DHW[i] -
                                                        electricity_surplus[i]) * COP_DHW[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP_DHW[i] = 0  # dhw is provided by tank
                                Q_DHWTank_out[i] = remaining_heating_Energy
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[
                                    i]
                                CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP_DHW[i] = DeficitEnergy
                                Q_DHWTank_out[i] = EnergyInTank
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[i]
                                CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

            if i > 0:
                # calculate hourly temperature loss:
                TankLossHourly[i] = (CurrentTankTemperature[i - 1] -
                                                 TankSurroundingTemperature) * TankSurfaceArea * TankLoss  # W
                # surplus of PV electricity is used to charge the tank:
                CurrentTankTemperature[i] = CurrentTankTemperature[i - 1] + \
                                                        (electricity_surplus[i] *
                                                         COP_DHW[i] - TankLossHourly[i]) / \
                                                        (TankSize * self.cp_water)
                # Q_HP_DHW is increased:
                Q_HP_DHW[i] = electricity_surplus[i] * COP_DHW[i] + hot_water_demand[i]
                Q_DHWTank_in[i] = electricity_surplus[i] * COP_DHW[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if CurrentTankTemperature[i] > TankMaxTemperature:
                    electricity_surplus_after_DHW[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * (
                                                                           TankSize * self.cp_water) / COP_DHW[i]  # W
                    CurrentTankTemperature[i] = TankMaxTemperature
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP_DHW[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                              (TankSize * self.cp_water) + hot_water_demand[i]
                    Q_DHWTank_in[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                                  (TankSize * self.cp_water)

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if CurrentTankTemperature[i] <= TankMinTemperature:
                    electricity_deficit[i] = (TankMinTemperature - CurrentTankTemperature[
                        i]) * (TankSize * self.cp_water) / COP_DHW[i]
                    CurrentTankTemperature[i] = TankMinTemperature
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_DHW[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP_DHW[i] = electricity_deficit[i] * COP_DHW[i] + hot_water_demand[i]
                    Q_DHWTank_in[i] = electricity_deficit[i] * COP_DHW[i]

                # if there is energy in the tank it will be used for heating:
                if CurrentTankTemperature[i] > TankMinTemperature:
                    EnergyInTank = (CurrentTankTemperature[i] - TankMinTemperature) * (
                            TankSize * self.cp_water)

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if electricity_grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if electricity_grid_demand[i] > hot_water_demand[i] / COP_DHW[i]:  # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP_DHW[i] = 0  # DHW is provided by tank
                                Q_DHWTank_out[i] = hot_water_demand[i]
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[
                                    i]
                                CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP_DHW[i] = DeficitEnergy  # dhw partly provided by tank
                                Q_DHWTank_out[i] = EnergyInTank
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[i]
                                CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if electricity_surplus[i] <= hot_water_demand[i] / COP_DHW[i]:
                            # calculate the part that can be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / COP_DHW[i] -
                                                        electricity_surplus[i]) * COP_DHW[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP_DHW[i] = 0  # dhw is provided by tank
                                Q_DHWTank_out[i] = remaining_heating_Energy
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[
                                    i]
                                CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP_DHW[i] = DeficitEnergy
                                Q_DHWTank_out[i] = EnergyInTank
                                grid_demand_after_DHW[i] -= Q_DHWTank_out[i] / COP_DHW[i]
                                CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

        self.Q_DHWTank_out = Q_DHWTank_out
        self.Q_DHWTank_in = Q_DHWTank_in
        self.Q_DHW_HP_out = Q_HP_DHW
        self.E_DHWTank = (CurrentTankTemperature+273.15) * self.cp_water * TankSize
        return grid_demand_after_DHW, electricity_surplus_after_DHW

    def calculate_heating_tank_energy(self, electricity_grid_demand, electricity_surplus, heating_demand):
        """
        Calculates the energy/temperature inside the hot water tank and the heating energy that has to be actually used
        when the tank energy is always used for heating when necessary.

        Returns: grid_demand_after_heating_tank, electricity_surplus_after_tank
        """
        T_outside = self.scenario.region_class.temperature
        TankSize = self.scenario.spaceheatingtank_class.size
        TankMinTemperature = self.scenario.spaceheatingtank_class.temperature_min
        TankMaxTemperature = self.scenario.spaceheatingtank_class.temperature_max
        TankSurfaceArea = self.scenario.spaceheatingtank_class.surface_area
        TankLoss = self.scenario.spaceheatingtank_class.loss
        TankSurroundingTemperature = self.scenario.spaceheatingtank_class.temperature_surrounding
        TankStartTemperature = self.scenario.spaceheatingtank_class.temperature_start
        COP_SpaceHeating = self.COP_HP(T_outside, 35,
                                       self.scenario.boiler_class.carnot_efficiency_factor,
                                       self.scenario.boiler_class.name)  # 35°C supply temperature

        # Assumption: Tank is always kept at minimum temperature except when it is charged with surplus energy:
        TankLoss_hourly = np.zeros(electricity_surplus.shape)
        CurrentTankTemperature = np.zeros(electricity_surplus.shape)
        electricity_surplus_after_tank = np.zeros(electricity_surplus.shape)
        Electricity_deficit = np.zeros(electricity_surplus.shape)
        Q_heating_HP = np.copy(heating_demand)
        Q_heatingTank_in = np.zeros(electricity_surplus.shape)
        Q_heatingTank_out = np.zeros(electricity_surplus.shape)
        grid_demand_after_heating_tank = np.copy(electricity_grid_demand)  # W

        for i, element in enumerate(TankLoss_hourly):
            if i == 0:
                TankLoss_hourly[i] = (TankStartTemperature - TankSurroundingTemperature) * \
                                                 TankSurfaceArea * TankLoss  # W
                # surplus of PV electricity is used to charge the tank:
                CurrentTankTemperature[i] = TankStartTemperature + \
                                                        (electricity_surplus[i] *
                                                         COP_SpaceHeating[i] - TankLoss_hourly[i]) / \
                                                        (TankSize * self.cp_water)
                # Q_heating_HP is increased:
                Q_heating_HP[i] = electricity_surplus[i] * COP_SpaceHeating[i] + heating_demand[i]
                Q_heatingTank_in[i] = electricity_surplus[i] * COP_SpaceHeating[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if CurrentTankTemperature[i] > TankMaxTemperature:
                    electricity_surplus_after_tank[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * (
                                                                 TankSize * self.cp_water) / COP_SpaceHeating[i]  # W
                    CurrentTankTemperature[i] = TankMaxTemperature
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_heating_HP[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                      (TankSize * self.cp_water) + heating_demand[i]
                    Q_heatingTank_in[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                          (TankSize * self.cp_water)

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if CurrentTankTemperature[i] <= TankMinTemperature:
                    Electricity_deficit[i] = (TankMinTemperature - CurrentTankTemperature[
                        i]) * (TankSize * self.cp_water) / COP_SpaceHeating[i]
                    CurrentTankTemperature[i] = TankMinTemperature
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_heating_tank[i] = electricity_grid_demand[i] + Electricity_deficit[i]
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_heating_HP[i] = Electricity_deficit[i] * COP_SpaceHeating[i] + heating_demand[i]
                    Q_heatingTank_in[i] = Electricity_deficit[i] * COP_SpaceHeating[i]

                # if there is energy in the tank it will be used for heating:
                if CurrentTankTemperature[i] > TankMinTemperature:
                    EnergyInTank = (CurrentTankTemperature[i] - TankMinTemperature) * (
                            TankSize * self.cp_water)

                    # if the PV does not cover the whole elctricity in this hour, the tank will cover for heating
                    if electricity_grid_demand[i] > 0:
                        # if the PV does not cover any energy for the heating:
                        if electricity_grid_demand[i] > Q_heating_HP[i] / COP_SpaceHeating[i]:  # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to heat the building, the Q_heating_HP goes to 0
                            if EnergyInTank > Q_heating_HP[i]:
                                SurplusEnergy = EnergyInTank - Q_heating_HP[i]
                                Q_heatingTank_out[i] = heating_demand[i]
                                Q_heating_HP[i] = 0  # Building is heated by tank
                                CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)  # the tank temperature drops to minimal temperature + the energy that is left
                            # if the energy in the tank is not enough to heat the building Q_heating_HP will be just reduced
                            if EnergyInTank <= Q_heating_HP[i]:
                                DeficitEnergy = Q_heating_HP[i] - EnergyInTank
                                Q_heating_HP[i] = DeficitEnergy  # Building is partly heated by tank
                                Q_heatingTank_out[i] = EnergyInTank
                                CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature drops to minimal energy

                        # if the PV does cover part of the heating energy:
                        if electricity_grid_demand[i] <= Q_heating_HP[i] / COP_SpaceHeating[i]:
                            # calculate the part than can be covered by the tank:
                            remaining_heating_Energy = (Q_heating_HP[i] / COP_SpaceHeating[i] -
                                                        electricity_grid_demand[i]) * COP_SpaceHeating[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_heatingTank_out[i] = remaining_heating_Energy
                                Q_heating_HP[i] = 0  # Building is heated by tank and PV
                                CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_heating_HP[i] = DeficitEnergy
                                Q_heatingTank_out[i] = EnergyInTank
                                CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature drops to minimal energy

            if i > 0:

                # if outside temperature is > 20°C the tank will not be charged except to keep min temperature:
                # we assume that heating is not necessary when outside temp is above 20°C
                if T_outside[i] > 20:
                    TankLoss_hourly[i] = (CurrentTankTemperature[
                                                          i - 1] - TankSurroundingTemperature) * \
                                                     TankSurfaceArea * TankLoss  # W
                    CurrentTankTemperature[i] = CurrentTankTemperature[i - 1] - \
                                                            (TankLoss_hourly[i]) / \
                                                            (TankSize * self.cp_water)

                    # if the tank is empty, it will be charged to keep it at minimum temperature
                    if CurrentTankTemperature[i] <= TankMinTemperature:
                        Electricity_deficit[i] = (TankMinTemperature - CurrentTankTemperature[i]) * \
                                                 (TankSize * self.cp_water) / COP_SpaceHeating[i]
                        # if there is PV power left, the tank is kept at minimum temp with PV power:
                        if electricity_surplus[i] >= Electricity_deficit[i]:
                            CurrentTankTemperature[i] = TankMinTemperature
                            # the surplus of PV is sold to the grid minus the part used for heating at minimum:
                            electricity_surplus_after_tank[i] = electricity_surplus[i] - Electricity_deficit[i]
                            Q_heatingTank_in[i] = Electricity_deficit[i] * COP_SpaceHeating[i]
                        # if PV power is not enough to keep the tank at minimum temp:
                        elif electricity_surplus[i] > 0 and electricity_surplus[i] < Electricity_deficit[i]:
                            CurrentTankTemperature[i] = TankMinTemperature
                            # electricity deficit gets reduced by the available PV power
                            Electricity_deficit[i] = Electricity_deficit[i] - electricity_surplus[i]
                            grid_demand_after_heating_tank[i] = electricity_grid_demand[i] + Electricity_deficit[i]
                            Q_heatingTank_in[i] = Electricity_deficit[i] * COP_SpaceHeating[i]

                        # if no PV power available total load is increased
                        else:
                            CurrentTankTemperature[i] = TankMinTemperature
                            # electric energy is raised by the amount the tank has to be heated:
                            grid_demand_after_heating_tank[i] = electricity_grid_demand[i] + Electricity_deficit[i]
                            Q_heatingTank_in[i] = Electricity_deficit[i] * COP_SpaceHeating[i]

                # if outside temperature is < 20°C the tank will always be charged:
                else:
                    TankLoss_hourly[i] = (CurrentTankTemperature[
                                                          i - 1] - TankSurroundingTemperature) * TankSurfaceArea * TankLoss  # W
                    CurrentTankTemperature[i] = CurrentTankTemperature[i - 1] + \
                                                            (electricity_surplus[i] *
                                                             COP_SpaceHeating[i] -
                                                             TankLoss_hourly[i]) / \
                                                            (TankSize * self.cp_water)
                    # Q_heating_HP is increased:
                    Q_heating_HP[i] = electricity_surplus[i] * COP_SpaceHeating[i] + heating_demand[i]
                    Q_heatingTank_in[i] = electricity_surplus[i] * COP_SpaceHeating[i]

                    # if temperature exceed maximum temperature, surplus of electricity is calculated
                    # and temperature is kept at max temperature:
                    if CurrentTankTemperature[i] > TankMaxTemperature:
                        electricity_surplus_after_tank[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * (
                                        TankSize * self.cp_water) / COP_SpaceHeating[i]  # W
                        CurrentTankTemperature[i] = TankMaxTemperature

                        # Q_HP_DHW and Q_DHWTank_in:
                        Q_heating_HP[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                                  (TankSize * self.cp_water) + heating_demand[i]
                        Q_heatingTank_in[i] = (CurrentTankTemperature[i] - TankMaxTemperature) * \
                                                      (TankSize * self.cp_water)

                    # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                    # and necessary electricity is calculated
                    if CurrentTankTemperature[i] <= TankMinTemperature:
                        Electricity_deficit[i] = (TankMinTemperature - CurrentTankTemperature[
                            i]) * (TankSize * self.cp_water) / COP_SpaceHeating[i]
                        CurrentTankTemperature[i] = TankMinTemperature
                        # electric energy is raised by the amount the tank has to be heated:
                        grid_demand_after_heating_tank[i] = electricity_grid_demand[i] + \
                                                                        Electricity_deficit[i]
                        # Q_HP_DHW and Q_DHWTank_in:
                        Q_heating_HP[i] = Electricity_deficit[i] * COP_SpaceHeating[i] + heating_demand[i]
                        Q_heatingTank_in[i] = Electricity_deficit[i] * COP_SpaceHeating[i]

                    # if there is energy in the tank it will be used for heating:
                    if CurrentTankTemperature[i] > TankMinTemperature:
                        EnergyInTank = (CurrentTankTemperature[i] - TankMinTemperature) * (
                                TankSize * self.cp_water)

                        # if the PV does not cover the whole electricity in this hour, the tank will cover for heating
                        if electricity_grid_demand[i] > 0:
                            # if the PV does not cover any energy for the heating:
                            if electricity_grid_demand[i] > Q_heating_HP[i] / COP_SpaceHeating[i]:  # means that the heating is not covered at all by the PV (it covers other things first)
                                # if the Energy in the tank is enough to heat the building, the Q_heating_HP goes to 0
                                if EnergyInTank > Q_heating_HP[i]:
                                    SurplusEnergy = EnergyInTank - Q_heating_HP[i]
                                    # tank exit energy:
                                    Q_heatingTank_out[i] = heating_demand[i]
                                    Q_heating_HP[i] = 0  # Building is heated by tank
                                    CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                            TankSize * self.cp_water)  # the tank temperature drops to minimal temperature + the energy that is left
                                    # the total electric energy will be reduced by the amount of the HP electric energy:
                                    grid_demand_after_heating_tank[i] = electricity_grid_demand[i] - \
                                                                                    heating_demand[i] / \
                                                                                    COP_SpaceHeating[i]

                                # if the energy in the tank is not enough to heat the building Q_heating_HP will be just reduced
                                if EnergyInTank <= Q_heating_HP[i]:
                                    DeficitEnergy = Q_heating_HP[i] - EnergyInTank
                                    Q_heating_HP[i] = DeficitEnergy  # Building is partly heated by tank
                                    CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature drops to minimal energy
                                    # energy leaving the tank:
                                    Q_heatingTank_out[i] = EnergyInTank
                                    # the total electric energy will be reduced by the part of the HP as well:
                                    grid_demand_after_heating_tank[i] = electricity_grid_demand[i] - \
                                                                                    (heating_demand[i] -
                                                                                     Q_heating_HP[i]) / \
                                                                                    COP_SpaceHeating[i]

                            # if the PV does cover part of the heating energy:
                            if electricity_grid_demand[i] <= Q_heating_HP[i] / COP_SpaceHeating[i]:
                                # calculate the part than can be covered by the tank:
                                remaining_heating_Energy = (Q_heating_HP[i] / COP_SpaceHeating[i] -
                                                            electricity_grid_demand[i]) * \
                                                           COP_SpaceHeating[i]
                                # if the energy in the tank is enough to cover the remaining heating energy:
                                if EnergyInTank > remaining_heating_Energy:
                                    SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                    # Q-tank_out:
                                    Q_heatingTank_out[i] = remaining_heating_Energy
                                    Q_heating_HP[i] = 0  # Building is heated by tank and PV
                                    CurrentTankTemperature[i] = TankMinTemperature + SurplusEnergy / (
                                            TankSize * self.cp_water)
                                    # electric total energy is reduced by the part of the HP:
                                    grid_demand_after_heating_tank[i] = electricity_grid_demand[i] - \
                                                                                    heating_demand[i] / \
                                                                                    COP_SpaceHeating[i]
                                # if the energy in the tank is not enough to cover the remaining heating energy:
                                if EnergyInTank <= remaining_heating_Energy:
                                    DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                    Q_heating_HP[i] = DeficitEnergy
                                    Q_heatingTank_out[i] = EnergyInTank
                                    CurrentTankTemperature[i] = TankMinTemperature  # the tank temperature drops to minimal energy
                                    # electric energy is reduced by the part of the HP:
                                    grid_demand_after_heating_tank[i] = electricity_grid_demand[i] - \
                                                                                    (heating_demand[i] -
                                                                                     Q_heating_HP[i]) / \
                                                                                    COP_SpaceHeating[i]

        #  Q_heating_HP is now the heating energy with the usage of the tank
        # Electricity_surplus is the electricity that can not be stored in the tank
        # Electricity_deficit is the electricity needed to keep the tank on temperature
        # TankLoss_hourly are the total losses of the tank
        # Total_Load_WaterTank is the actual electric load with the use of the water storage
        self.Q_HeatingTank_out = Q_heatingTank_out
        self.Q_HeatingTank_in = Q_heatingTank_in
        self.E_HeatingTank = (CurrentTankTemperature+273.15) * self.cp_water * TankSize
        self.Q_Heating_HP_out = Q_heating_HP

        return grid_demand_after_heating_tank, electricity_surplus_after_tank

    def calculate_battery_energy(self, grid_demand, electricity_surplus):
        """
        Assumption: The battery is not discharging itself.
        """
        Capacity = self.scenario.battery_class.capacity  # kWh
        MaxChargePower = self.scenario.battery_class.charge_power_max  # kW
        MaxDischargePower = self.scenario.battery_class.discharge_power_max  # kW
        ChargeEfficiency = self.scenario.battery_class.charge_efficiency
        DischargeEfficiency = self.scenario.battery_class.discharge_efficiency

        # Battery is not charged at the beginning of the simulation
        BatterySOC = np.zeros(electricity_surplus.shape)
        surplus_after_battery = np.zeros(electricity_surplus.shape)
        Total_load_battery = np.copy(grid_demand)
        Battery2Load = np.zeros(electricity_surplus.shape)
        Battery_charge = np.zeros(electricity_surplus.shape)

        for i, element in enumerate(BatterySOC):
            if i == 0:  # there will be no charging at 01:00 clock in the morning of the 1st january:
                BatterySOC[i] = 0

            if i > 0:
                # if there is surplus energy, the battery will be charged
                if electricity_surplus[i] > 0:
                    # check if the battery can store the power or if its already fully charged:
                    if BatterySOC[i - 1] < Capacity:  # there is space to charge

                        if electricity_surplus[i] <= MaxChargePower:  # maximum charging power is not exceeded
                            # determine how much capacity is available:
                            capacity_left = Capacity - BatterySOC[i - 1]
                            BatterySOC[i] = BatterySOC[i - 1] + electricity_surplus[i] * ChargeEfficiency
                            Battery_charge[i] = electricity_surplus[i] * ChargeEfficiency
                            # check if the battery exceeds maximum charge limit, if yes:
                            if BatterySOC[i] > Capacity:
                                charging_energy = capacity_left / ChargeEfficiency
                                surplus_after_battery[i] = electricity_surplus[i] - charging_energy
                                BatterySOC[i] = Capacity
                                Battery_charge[i] = charging_energy

                        if electricity_surplus[i] > MaxChargePower:  # maximum charging power is exceeded
                            # determine how much capacity is available:
                            capacity_left = Capacity - BatterySOC[i - 1]
                            BatterySOC[i] = BatterySOC[i - 1] + MaxChargePower * ChargeEfficiency
                            surplus_after_battery[i] = electricity_surplus[i] - MaxChargePower
                            Battery_charge[i] = MaxChargePower * ChargeEfficiency
                            # check if the battery exceeds maximum capacity limit, if yes:
                            if BatterySOC[i] > Capacity:
                                charging_energy = capacity_left / ChargeEfficiency
                                surplus_after_battery[i] = electricity_surplus[i] - charging_energy
                                BatterySOC[i] = Capacity
                                Battery_charge[i] = charging_energy

                    # if battery can not be charged because its full:
                    if BatterySOC[i - 1] == Capacity:
                        BatterySOC[i] = Capacity
                        surplus_after_battery[i] = electricity_surplus[i]

                # if there is no surplus of energy and the battery has energy stored, it will provide energy:
                if electricity_surplus[i] == 0:
                    # check if battery has power:
                    if BatterySOC[i - 1] > 0:
                        # if the power in battery is enough to cover whole electricity demand:
                        if BatterySOC[i - 1] > grid_demand[i] / DischargeEfficiency:
                            Total_load_battery[i] = 0
                            BatterySOC[i] = BatterySOC[i - 1] - grid_demand[i] / DischargeEfficiency
                            Battery2Load[i] = grid_demand[i] / DischargeEfficiency
                            # check if maximum discharge power is exceeded:
                            if Total_load_battery[i] / DischargeEfficiency > MaxDischargePower:
                                Total_load_battery[i] = grid_demand[i] - MaxDischargePower
                                BatterySOC[i] = BatterySOC[i - 1] - MaxDischargePower / DischargeEfficiency
                                Battery2Load[i] = MaxDischargePower

                        # if the power in the battery is not enough to cover the whole electricity demand:
                        if BatterySOC[i - 1] <= grid_demand[i] / DischargeEfficiency:
                            Total_load_battery[i] = grid_demand[i] - BatterySOC[i - 1] * DischargeEfficiency
                            BatterySOC[i] = 0
                            Battery2Load[i] = BatterySOC[i - 1] * DischargeEfficiency
                            # check if max discharge power is exceeded:
                            if BatterySOC[i - 1] > MaxDischargePower:
                                BatterySOC[i] = BatterySOC[i - 1] - MaxDischargePower / DischargeEfficiency
                                Total_load_battery[i] = grid_demand[i] - MaxDischargePower
                                Battery2Load[i] = MaxDischargePower

        self.BatCharge = Battery_charge
        self.PV2Bat = Battery_charge
        self.BatSoC = BatterySOC
        self.BatDischarge = Battery2Load  # as the battery gets only discharged by providing load coverage
        self.Bat2Load = Battery2Load
        return Total_load_battery, surplus_after_battery, BatterySOC, Battery2Load

    def fill_parameter_values(self) -> None:
        """ fills all self parameter values with the input values"""
        # price
        self.electricity_price = self.scenario.electricityprice_class.electricity_price  # C/Wh
        # Feed in Tariff of Photovoltaic
        self.FiT = self.scenario.feedintariff_class.feed_in_tariff  # C/Wh
        # solar gains:
        self.Q_Solar = self.calculate_solar_gains()  # W
        # outside temperature
        self.T_outside = self.scenario.region_class.temperature  # °C
        # COP of heatpump
        self.SpaceHeatingHourlyCOP = self.COP_HP(self.scenario.region_class.temperature, 35,
                                  self.scenario.boiler_class.carnot_efficiency_factor,
                                  self.scenario.boiler_class.name)  # 35 °C supply temperature
        # COP of cooling
        self.CoolingCOP = self.scenario.airconditioner_class.efficiency  # single value because it is not dependent on time
        # electricity load profile
        self.BaseLoadProfile = self.scenario.electricitydemand_class.electricity_demand
        # PV profile
        self.PhotovoltaicProfile = self.scenario.pv_class.power
        # HotWater
        self.HotWaterProfile = self.scenario.hotwaterdemand_class.hot_water_demand
        self.HotWaterHourlyCOP = self.COP_HP(self.scenario.region_class.temperature, 55,
                                    self.scenario.boiler_class.carnot_efficiency_factor,
                                    self.scenario.boiler_class.name)  # 55 °C supply temperature

        # building data: is not saved in results (to much and not useful)

        # Heating Tank data
        # Mass of water in tank
        self.M_WaterTank_heating = self.scenario.spaceheatingtank_class.size
        # Surface of Tank in m2
        self.A_SurfaceTank_heating = self.scenario.spaceheatingtank_class.surface_area
        # insulation of tank, for calc of losses
        self.U_ValueTank_heating = self.scenario.spaceheatingtank_class.loss
        self.T_TankStart_heating = self.scenario.spaceheatingtank_class.temperature_start
        # surrounding temp of tank
        self.T_TankSurrounding_heating = self.scenario.spaceheatingtank_class.temperature_surrounding

        # DHW Tank data
        # Mass of water in tank
        self.M_WaterTank_DHW = self.scenario.hotwatertank_class.size
        # Surface of Tank in m2
        self.A_SurfaceTank_DHW = self.scenario.hotwatertank_class.surface_area
        # insulation of tank, for calc of losses
        self.U_ValueTank_DHW = self.scenario.hotwatertank_class.loss
        self.T_TankStart_DHW = self.scenario.hotwatertank_class.temperature_start
        # surrounding temp of tank
        self.T_TankSurrounding_DHW = self.scenario.hotwatertank_class.temperature_surrounding

        # heat pump
        self.SpaceHeating_HeatPumpMaximalThermalPower = self.scenario.boiler_class.thermal_power_max

        # Battery data
        self.ChargeEfficiency = self.scenario.battery_class.charge_efficiency
        self.DischargeEfficiency = self.scenario.battery_class.discharge_efficiency

    def run(self):
        """
        Assumption for the Reference scenario: the produced PV power is always used for the immediate electric demand,
        if there is a surplus of PV power, it will be used to charge the Battery,
        if the Battery is full or not available, the PV power will be used to heat the HotWaterTank,
        if the HotWaterTank is not available or full, the PV power will sold to the Grid.
        The surplus of PV energy is never used to Preheat or Precool the building.

        """
        self.fill_parameter_values()  # set input parameters to self values
        # calculate the heating and cooling energy and indoor air + thermal mass temperature:
        heating_demand, cooling_demand, T_Room, Tm_t = \
            R5C1Model(self.scenario).calculate_heating_and_cooling_demand()
        room_heating = heating_demand
        self.Tm_t = Tm_t
        self.T_room = T_Room
        self.Q_RoomCooling = cooling_demand
        # check if heating element has to be used for the HP:
        heating_element = np.zeros(heating_demand.shape)
        for index, element in enumerate(heating_demand):
            if element > self.scenario.boiler_class.thermal_power_max:
                heating_demand[index] = self.scenario.boiler_class.thermal_power_max
                heating_element[index] = element - self.scenario.boiler_class.thermal_power_max

        self.Q_HeatingElement = heating_element
        # electricity for hot water:
        hot_water_electricity_demand = self.HotWaterProfile / self.HotWaterHourlyCOP
        # electricity for heating:
        heating_electricity_demand = heating_demand / self.SpaceHeatingHourlyCOP
        # electricity for cooling:
        cooling_electricity_demand = cooling_demand / self.CoolingCOP

        # add up all electric loads but without Heating:
        total_electricity_load = self.scenario.electricitydemand_class.electricity_demand + \
                                 hot_water_electricity_demand + \
                                 heating_electricity_demand + \
                                 cooling_electricity_demand + \
                                 heating_element

        # The PV profile is subtracted from the total load
        total_load_minus_pv = total_electricity_load - self.PhotovoltaicProfile  # kW


        # determine the surplus PV power:
        PV_profile_surplus = np.copy(total_load_minus_pv)
        PV_profile_surplus[PV_profile_surplus > 0] = 0
        PV_profile_surplus = abs(PV_profile_surplus)
        # determine PV2Load
        self.PV2Load = self.PhotovoltaicProfile - PV_profile_surplus

        # Total load profile can not be negative: make negative numbers to 0
        total_load_minus_pv[total_load_minus_pv < 0] = 0

        # if neither battery nor hot water tank are used:
        grid_demand = np.copy(total_load_minus_pv)
        # electricity surplus is the energy that is sold back to the grid because it can not be used:
        electricity_sold = np.copy(PV_profile_surplus)  # kW

        # Battery storage:
        # if battery is used:
        if self.scenario.battery_class.capacity > 0:
            Total_load_battery, Electricity_surplus_Battery, BatterySOC, Battery2Load = \
                self.calculate_battery_energy(grid_demand, PV_profile_surplus)
            grid_demand = Total_load_battery
            # amount of electricity from PV to Battery:
            electricity_sold = Electricity_surplus_Battery

        # DHW storage:
        if self.scenario.hotwatertank_class.size > 0:
            grid_demand_after_DHW, electricity_surplus_after_DHW = self.calculate_DHW_tank_energy(
                grid_demand, electricity_sold, self.HotWaterProfile)
            electricity_sold = electricity_surplus_after_DHW
            grid_demand = grid_demand_after_DHW

        # When there is PV surplus energy it either goes to a storage or is sold to the grid:
        # Water Tank as storage:
        if self.scenario.spaceheatingtank_class.size > 0:
            grid_demand_after_heating_tank, electricity_surplus_after_tank = self.calculate_heating_tank_energy(
                grid_demand, electricity_sold, heating_demand)
            # remaining surplus of electricity
            electricity_sold = electricity_surplus_after_tank
            grid_demand = grid_demand_after_heating_tank

        # calculate the electricity cost:
        price_hourly = self.scenario.electricityprice_class.electricity_price
        FIT = self.scenario.feedintariff_class.feed_in_tariff
        self.total_operation_cost = price_hourly * grid_demand - electricity_sold * FIT

        # grid variables
        self.Grid = grid_demand
        self.Grid2Load = grid_demand
        self.Grid2Bat = np.full((8760, ), 0)

        # PV variables
        self.PV2Grid = electricity_sold

        # electric load
        self.Load = total_electricity_load

        # electricity fed back to the grid
        self.Feed2Grid = electricity_sold

        # room heating
        self.Q_room_heating = room_heating
        self.Q_HeatingTank_bypass = room_heating - self.Q_HeatingTank_out

        # DHW
        self.Q_DHWTank_bypass = self.HotWaterProfile - self.Q_DHWTank_out

if __name__ == "__main__":
    from _Refactor.models.operation.data_collector import ReferenceDataCollector

    scenario = AbstractScenario(scenario_id=0)
    reference_model = RefOperationModel(scenario)
    reference_model.run()
    hourly_results = ReferenceDataCollector(reference_model).collect_reference_results_hourly()
    yearly_results = ReferenceDataCollector(reference_model).collect_reference_results_yearly()

    # save results to database
    ReferenceDataCollector(reference_model).save_hourly_results()
    ReferenceDataCollector(reference_model).save_yearly_results()


    pass


