import pandas as pd
import numpy as np


def CREATE_DHW_ENERGYDEMAND_PROFILE(RN, YEAR, OUTPUT_PATH):
    # import user profiles for DHW:
    datei = RN + '__dynamic_calc_data_up_' + str(YEAR) + '.csv'
    data_user_profiles = pd.read_csv(OUTPUT_PATH / datei)
    # Number if user profiles:
    num_up = len(data_user_profiles)

    # to numpy:
    DHW_need_day_m2 = data_user_profiles["DHW_per_day"].to_numpy()
    DHW_need_day_m2_8760_up = np.ones((num_up, 8760))

    # Days per month
    DpM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum_hours = 0
    use_DaysPerYear = np.zeros(shape=DHW_need_day_m2.shape)
    non_use_hours_per_day = 24 - data_user_profiles["hours_use_per_day"].to_numpy().round()

    data_user_profiles = data_user_profiles.to_numpy()

    # iterate through month:
    for month in range(12):
        days_this_month = DpM[month]
        num_hours = 24 * days_this_month
        # number of days DHW is used this month: is rounded because the user profiles are not full numbers
        use_days_this_month = data_user_profiles[:, 5 + month].round()
        # add days up to get used days per year to check later how the rounding affected the outcome
        use_DaysPerYear = use_DaysPerYear + use_days_this_month

        # Create SET TEMPERATURE Matrix
        Daynon_use_vector = np.zeros((num_up, num_hours))
        # Anzahl der Tage an denen kein DHW verwendet wird im monat:
        non_use_days_this_month = days_this_month - use_days_this_month
        # Anzahl der Tage an denen kein DHW verwendet wir in der Woche:
        non_use_days_per_week_this_month = non_use_days_this_month / 4
        # Unique Anzahl der Stunden an denen am Tag kein DHW verwendet wird:
        unique_non_use_hours_per_day = np.unique(non_use_hours_per_day)

        # Erstelle Matrix aus 1 und 0.3 die dann mit dem Userprofil multipliziert wird:
        # iteriere durch die Stunden am Tag ohne DHW:
        for j in unique_non_use_hours_per_day:
            # index der Haushalte die diese Anzahl an Stunden kein DHW verwenden:
            idx = np.where(non_use_hours_per_day == j)

            # Asumption: Used days after buisness hours demand is 30% of usual demand:
            # wenn es mehr als 6 Stunden sind werden diese aufgeteilt auf die ersten paar stunden (ab 00:00) vom Tag,
            # und auf die letzten Stunden vom vorherigen Tag. Dabei werden fix 6 Stunden von (00:00 bis 06:00) auf 0.3
            # gestellt und die restlichen Stunden bis 24:00 rückwärts auf 0.3 gestellt.
            # Bsp: Bei 7 Studen nicht verwendung wird der DHW verbrauch von 00:00 bis 06:00 auf 0.3 gestellt und
            # von 23:00 bis 24:00. TODO warum der vorherige Tag?
            if j > 6:
                for day in np.arange(1, days_this_month + 1):
                    DHW_need_day_m2_8760_up[np.ix_(idx[0], np.arange(cum_hours + (day - 1) * 24,
                                                                     cum_hours + (day - 1) * 24 + 6))] = 0.3
                    DHW_need_day_m2_8760_up[np.ix_(idx[0], np.arange(cum_hours + day * 24 - (int(j) - 6),
                                                                     cum_hours + day * 24))] = 0.3
            # wenn es weniger als 7 Stunden sind wird der Verbrauch in den frühen Morgenstunden bis maximal 06:00
            # gesenkt. Bsp. Bei 2 Stunden nicht gebrauch wird der DHW verbrauch von 04:00 bis 06:00 auf 0.3 gesenkt.
            else:
                for day in np.arange(1, days_this_month + 1):
                    DHW_need_day_m2_8760_up[np.ix_(idx[0], np.arange(cum_hours + (day - 1) * 24 + 7 - int(j) - 1,
                                                                     cum_hours + (day - 1) * 24 + 6))] = 0.3
        # iteriere über Wochen:
        # An tagen wo den ganzen Tag niemand da ist, ist der DHW Verbrauch 15%. -> Matrix mit 0.15 füllen:
        for week in range(1, 5):
            Non_use_days_this_week = (np.maximum(0, non_use_days_per_week_this_month * week * 24 -
                                                 Daynon_use_vector.sum(axis=1)) / 24).round()
            Unique_Non_use_days_this_week = np.unique(Non_use_days_this_week)
            # Asumption: On NON-used days the consumption is 15%:
            for j in Unique_Non_use_days_this_week:
                idx = np.where(Non_use_days_this_week == j)
                Daynon_use_vector[np.ix_(idx[0], np.arange((week * 7 - int(j)) * 24, week * 7 * 24))] = 1
                # It is asumed that the last days of the week are not used (weekend if its 2 days):
                DHW_need_day_m2_8760_up[np.ix_(idx[0], np.arange(cum_hours + (week * 7 - int(j)) * 24,
                                                                 cum_hours + week * 7 * 24))] = 0.15

        cum_hours = cum_hours + num_hours

    # prozentuale Aufteilung des DHW Gebrauchs:
    # TODO unterschiedliche DHW Verbrauchsprofile einbauen!
    DHW_h0_6 = np.linspace(0, 0.1, 7)
    DHW_h6_12 = np.linspace(DHW_h0_6[-1], 0.55, 7)
    DHW_h12_16 = np.linspace(DHW_h6_12[-1], 0.60, 5)
    DHW_h16_21 = np.linspace(DHW_h12_16[-1], 0.95, 6)
    DHW_h21_24 = np.linspace(DHW_h16_21[-1], 1, 4)

    cum_DHW_hourly_profile = np.concatenate([DHW_h0_6[1:], DHW_h6_12[1:], DHW_h12_16[1:], DHW_h16_21[1:],
                                             DHW_h21_24[1:]])
    DHW_hourly_profile = cum_DHW_hourly_profile - np.insert(cum_DHW_hourly_profile, 0, 0)[:-1]
    # get DHW_hourly_profile into 8760 * 34 matrix:
    DHW_hourly_profile = np.tile(DHW_hourly_profile, (34, 365))
    # multiply this matrix with the DHW demand per floor area
    DHW_need_day_m2_8760_up = DHW_need_day_m2_8760_up * DHW_hourly_profile
    # sum up usage of every User over the year and compare it to the counted days from above
    # the sum if DHW_need_day_m2_8760 is smaller because of the rounding error
    deviation_SOLL_vs_IST_usage = DHW_need_day_m2_8760_up.sum(axis=1) / use_DaysPerYear
    # Correcting the rounding error by dividing through the deviation_SOLL_IST_usage:
    DHW_need_day_m2_8760_up = DHW_need_day_m2_8760_up / np.tile(deviation_SOLL_vs_IST_usage, (8760, 1)).T

    # DHW Circulation loss 40% Loss:
    DHW_loss_Circulation_040_day_m2_8760_up = (0.4 / 24 * np.tile(DHW_need_day_m2, (8760, 1)).T)  # pro stunde
    # DHW need every hour in Wh:
    DHW_need_day_m2_8760_up = DHW_need_day_m2_8760_up * np.tile(DHW_need_day_m2, (8760, 1)).T

    return DHW_need_day_m2_8760_up, DHW_loss_Circulation_040_day_m2_8760_up
