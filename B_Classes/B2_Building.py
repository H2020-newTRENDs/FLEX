
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class Building:

    def __init__(self, para_series):
        self.ID_BuildingType = para_series["ID_BuildingType"]
        self.ID_BuildingAgeClass = para_series["ID_BuildingAgeClass"]
        self.FloorArea = para_series["FloorArea"]
        # other attributes to be added
        pass

    def calc_IndoorTemperature(self, Energy_TankToRoom_t, OutdoorTemperature_t):

        """
        :param Energy_TankToRoom_t: float
        :param OutdoorTemperature_t: float
        :return: IndoorTemperature_tp1: float
        """

        # constraints
        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                            ((PHI_ia + m.Q_heating[t]) / Hve) + m.T_outside[t])) / \
                           Htr_2

                # Equ. C.4
                return m.Tm_t[t] == (thermal_mass_starting_temp * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                        (Cm / 3600) + 0.5 * (Htr_3 + Htr_em))
            if t == 168:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t]) / Hve) + m.T_outside[t])) / \
                           Htr_2

                # Equ. C.4
                return m.Tm_t[t] == 20
            else:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t]) / Hve) + T_sup)) / \
                           Htr_2

                # Equ. C.4
                return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                        (Cm / 3600) + 0.5 * (Htr_3 + Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.time, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + thermal_mass_starting_temp) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                            T_sup + (PHI_ia + m.Q_heating[t]) / Hve)) / \
                      (Htr_ms + Htr_w + Htr_1)
                # Equ. C.11
                T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t]) / (Htr_is + Hve)
                # Equ. C.12
                T_op = 0.3 * T_air + 0.7 * T_s
                # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
                return m.T_room[t] == T_air
            else:
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.Tm_t[t - 1]) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                            T_sup + (PHI_ia + m.Q_heating[t]) / Hve)) / \
                      (Htr_ms + Htr_w + Htr_1)
                # Equ. C.11
                T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t]) / (Htr_is + Hve)
                # Equ. C.12
                T_op = 0.3 * T_air + 0.7 * T_s
                # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
                return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.time, rule=room_temperature_rc)



        IndoorTemperature = []
        return IndoorTemperature