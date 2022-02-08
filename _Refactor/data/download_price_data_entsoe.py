import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient, Area

"""
Query data from ENTSO-E transparency portal
------------------------------------------------------------------------------
DISCLAIMER: 
You may use the Code for any private or commercial purpose. However, you may not sell, 
sub-license, rent, lease, lend, assign or otherwise transfer, duplicate or otherwise 
reproduce, directly or indirectly, the Code in whole or in part. 

You acknowledge that the Code is provided “AS IS” and thesmartinsights.com expressly 
disclaims all warranties and conditions including, but not limited to, any implied 
warranties for suitability of the Code for a particular purpose, or any form of warranty 
that operation of the Code will be error-free.

You acknowledge that in no event shall thesmartinsights.com or any of its affiliates be 
liable for any damages arising out of the use of the Code or otherwise in connection with 
this agreement, including, without limitation, any direct, indirect special, incidental 
or consequential damages, whether any claim for such recovery is based on theories of 
contract, negligence, and even if thesmartinsights.com has knowledge of the possibility 
of potential loss or damage.
------------------------------------------------------------------------------- 
"""

pd.options.display.max_columns = None

def get_entsoe_prices(api_key: str,
                      start_time: str,
                      end_time: str,
                      country_code: str) -> np.array:
    # %% parameter definitions
    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp(start_time, tz='CET')
    end = pd.Timestamp(end_time, tz='CET')


    # Get day-ahead prices from ENTSO-E Transparency
    print('Prices in zone ' + country_code)
    DA_prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    prices = pd.DataFrame(DA_prices).reset_index(drop=True).to_numpy()
    return prices



        # correct daylight saving shifts: in fall, replace new hour by mean of two hours; in spring, use value of hour before
        # TODO i think this hour change is useless for us (need still saving to root db and aproprate function design)
        # DL_saving_28_Oct_18_1 = df_prices.iloc[650]['price_' + BZ]
        # DL_saving_28_Oct_18_2 = df_prices.iloc[651]['price_' + BZ]
        # DL_saving_28_Oct_18_new = (DL_saving_28_Oct_18_1 + DL_saving_28_Oct_18_2) / 2
        # # replace price at index 650 and column 1 by mean of the two hours
        # df_prices.iat[650, 1] = DL_saving_28_Oct_18_new
        #
        # DL_saving_27_Oct_19_1 = df_prices.iloc[9386]['price_' + BZ]
        # DL_saving_27_Oct_19_2 = df_prices.iloc[9387]['price_' + BZ]
        # DL_saving_27_Oct_19_new = (DL_saving_27_Oct_19_1 + DL_saving_27_Oct_19_2) / 2
        # # replace price at index 9386 and column 1 by mean of the two hours
        # df_prices.iat[9386, 1] = DL_saving_27_Oct_19_new
        #
        # # insert value of hour 01:00-02:00 for virtual 02:00-03:00 in spring
        # DL_saving_31_Mar_19_1 = df_prices.iloc[4346]['price_' + BZ]
        # # Create new dataframe without time shift
        # df_prices_new = df_prices.iloc[0:4347]
        # df_prices_rest = df_prices.iloc[4347:10969]
        # new_row = {'Date/time': '2019-03-31 02:00:00+01:00', 'price_' + BZ: DL_saving_31_Mar_19_1}
        # df_prices_new = df_prices_new.append(new_row, ignore_index=True)
        # df_prices_new = df_prices_new.append(df_prices_rest, ignore_index=True)
        # print('End prices')
        #
        # # Create dataframe containing all data
        # print('Start summary')
        # df_all = pd.DataFrame({'Date/time': df_prices_new['Date/time'],
        #                        'price' + BZ: df_prices_new['price_' + BZ],
        #                        # 'load_'+BZ: df_load_new['load_'+BZ],
        #                        # 'solar_'+BZ: df_solar_new['solar_'+BZ],
        #                        # 'wind_offs_'+BZ: df_wind_offs_new['wind_offs_'+BZ],
        #                        # wind_ons_'+BZ: df_wind_ons_new['wind_ons_'+BZ]
        #                        })
        # # remove double hours
        # df_all = df_all.drop([651])
        # df_all = df_all.drop([9388])
        # print('End summary')
        #
        # file_name = 'output/data_2019_' + BZ + '.xlsx'
        # print(df_all['Date/time'].dtypes)
        # df_all['Date/time'] = df_all['Date/time'].astype(str)
        # df_all.to_excel(file_name)
        # print(df_all.head())







