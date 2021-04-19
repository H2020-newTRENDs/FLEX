import pandas as pd
import numpy as np


def CREATE_SET_TEMP_PROFILE(RN, YEAR, OUTPUT_PATH):
    datei = RN + '__dynamic_calc_data_up_' + str(YEAR) + '.csv'
    data_user_profiles = pd.read_csv(OUTPUT_PATH / datei)

    # Define vectors from data_user_profiles as numpy arrays:
    Tset_heating_vector_up = data_user_profiles.loc[:, "Tset_heating_use_days"].to_numpy()
    Tset_heating_non_use_vector_up = data_user_profiles.loc[:, "Tset_heating_nonuse_days"].to_numpy()
    Tset_heating_non_office_hours_vector_up = (Tset_heating_vector_up + Tset_heating_non_use_vector_up) / 2

    Tset_cooling_vector_up = data_user_profiles.loc[:, "Tset_cooling_use_days"].to_numpy()
    Tset_cooling_non_use_vector_up = data_user_profiles.loc[:, "Tset_cooling_nonuse_days"].to_numpy()
    Tset_cooling_non_office_hours_vector_up = (Tset_cooling_vector_up + Tset_cooling_non_use_vector_up) / 2

    Tset_heating_8760_up = np.tile(2 + Tset_heating_vector_up, (8760, 1)).T  # TODO warum +2?
    Tset_cooling_8760_up = np.tile(Tset_cooling_vector_up, (8760, 1)).T

    # Days per month
    DpM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum_hours = 0
    # convert data user profiles to numpy to speed up loops:
    data_user_profiles = data_user_profiles.to_numpy()

    for month in range(12):
        days_this_month = DpM[month]
        num_hours = 24 * days_this_month

        use_days_this_month = data_user_profiles[:, 5 + month].round()
        non_use_hours_per_day = data_user_profiles[:, 17].round()

        # Create Tset matrix:
        Daynon_use_vector = np.zeros([len(data_user_profiles), int(num_hours)])
        non_use_days_this_month = days_this_month - use_days_this_month
        non_use_days_per_week_this_month = non_use_days_this_month / 4

        for week in range(1, 5):
            Non_use_days_this_week = (np.maximum(0, non_use_days_per_week_this_month * week * 24 -
                                                 Daynon_use_vector.sum(axis=1)) / 24).round()
            Unique_Non_use_days_this_week = np.unique(Non_use_days_this_week)
            for j in Unique_Non_use_days_this_week:
                idx = np.where(Non_use_days_this_week == j)
                array = np.arange((week * 7 - j) * 24, week * 7 * 24).astype(int)
                Daynon_use_vector[np.ix_(idx[0], array)] = 1
                Tset_heating_8760_up[np.ix_(idx[0],
                                            np.arange(cum_hours + (week * 7 - j) * 24,
                                                      cum_hours + week * 7 * 24).astype(int))] = \
                    np.tile(Tset_heating_non_use_vector_up[idx[0]], (int(j) * 24, 1)).T
                Tset_cooling_8760_up[np.ix_(idx[0],
                                            np.arange(cum_hours + (week * 7 - j) * 24,
                                                      cum_hours + week * 7 * 24).astype(int))] = \
                    np.tile(Tset_cooling_non_use_vector_up[idx[0]], (int(j) * 24, 1)).T

        unique_non_use_hours_per_day = np.unique(non_use_hours_per_day)

        for j in unique_non_use_hours_per_day:
            idx = np.where(non_use_hours_per_day == j)
            if j > 6:  # TODO warum 6?
                for day in np.arange(1, days_this_month + 1):
                    Tset_heating_8760_up[np.ix_(idx[0], np.arange(cum_hours + (day - 1) * 24,
                                                                  cum_hours + (day - 1) * 24 + 6))] = \
                        np.tile(Tset_heating_non_office_hours_vector_up[idx[0]], (6, 1)).T
                    Tset_heating_8760_up[np.ix_(idx[0], np.arange(cum_hours + day * 24 - (int(j) - 6),
                                                                  cum_hours + day * 24))] = \
                        np.tile(Tset_heating_non_office_hours_vector_up[idx[0]], ((int(j) - 6), 1)).T

                    Tset_cooling_8760_up[np.ix_(idx[0], np.arange(cum_hours + (day - 1) * 24,
                                                                  cum_hours + (day - 1) * 24 + 6))] = \
                        np.tile(Tset_cooling_non_office_hours_vector_up[idx[0]], (6, 1)).T
                    Tset_cooling_8760_up[np.ix_(idx[0], np.arange(cum_hours + day * 24 - (int(j) - 6),
                                                                  cum_hours + day * 24))] = \
                        np.tile(Tset_cooling_non_office_hours_vector_up[idx[0]], ((int(j) - 6), 1)).T
            else:
                for day in np.arange(1, days_this_month + 1):
                    Tset_heating_8760_up[np.ix_(idx[0], np.arange(cum_hours + (day - 1) * 24 + 7 - int(j) - 1,
                                                                  cum_hours + (day - 1) * 24 + 6))] = \
                        np.tile(Tset_heating_non_office_hours_vector_up[idx[0]], (int(j), 1)).T
                    Tset_cooling_8760_up[np.ix_(idx[0], np.arange(cum_hours + (day - 1) * 24 + 7 - int(j) - 1,
                                                                  cum_hours + (day - 1) * 24 + 6))] = \
                        np.tile(Tset_cooling_non_office_hours_vector_up[idx[0]], (int(j), 1)).T
        cum_hours = cum_hours + num_hours

    # Nachtabsenkung
    T_24 = np.zeros((1, 24))
    T_24[0, :6] = 1
    T_24[0, 21:] = 1
    # so oft verdoppeln dass er 8760 einträge hat (*365):
    HOURLY_PROFILE = np.tile(T_24, (1, 365))
    # Auswählen des Minimums (Tset_heating_vektor_up - 2°C für nachtabsenkung oder der schon bestehende Wert:
    Tset_heating_8760_up[:, np.where(HOURLY_PROFILE == 1)[1]] = \
        np.minimum(Tset_heating_8760_up[:, np.where(HOURLY_PROFILE == 1)[1]],
                   np.tile(Tset_heating_vector_up - 2, (int(T_24.sum()) * 365, 1)).T)

    return Tset_heating_8760_up, Tset_cooling_8760_up