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
                      country_code: str,
                      grid_fee: float) -> np.array:
    # %% parameter definitions

    #ToDo missing for all countries
    if country_code == "DE":
        country = "DE_LU"
    else:
        country = country_code

    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp(start_time, tz='CET')
    end = pd.Timestamp(end_time, tz='CET')

    # Get day-ahead prices from ENTSO-E Transparency
    print('Prices in zone ' + country)
    DA_prices = client.query_day_ahead_prices(country, start=start, end=end, resolution="60T")
    # drop last hour
    da_prices = DA_prices.iloc[:-1]
    prices = pd.DataFrame(da_prices).reset_index(drop=True).to_numpy() / 10 / 1_000  # €/MWh in ct/kWh & ct/kWh in ct/Wh
    # add grid fees:
    prices_total = prices + grid_fee / 1_000  # also in ct/Wh
    return prices_total


year = 2019
country = "AT"
electricity_price_config = {
    # variable price
    "api_key": 'b16a316d-4e35-4223-a3ac-5e02d5fa89ef',
    "start": f"{year}0101",
    "end": f"{year+1}0101",
    "country_code": country,
    "grid_fee": 20,  # ct/kWh
    "feed_in": 0.07,
    "gas_price": 0.009313796
}

variable_electricity_price = get_entsoe_prices(
    api_key=electricity_price_config["api_key"],
    start_time=electricity_price_config["start"],
    end_time=electricity_price_config["end"],
    country_code=electricity_price_config["country_code"],
    grid_fee=electricity_price_config["grid_fee"],
)

variable_price_to_db = np.column_stack(
    [np.full((8760,), electricity_price_config["country_code"]),
     np.full((8760,), "2019"),
     np.arange(1, 8761),  # id hour
     np.full((8760,), 1),  # ID
     variable_electricity_price,  # electricity_1
     np.full((8760,), electricity_price_config["feed_in"]),  # electricity_feed_in_1
     np.full((8760,), electricity_price_config["gas_price"])]
)

price_table = pd.DataFrame(variable_price_to_db, columns=["region", "year",	"id_hour", "ID_Electricity", "electricity_1", "electricity_feed_in_1", "gases_1"])

price_table.to_excel(r"C:\Users\mascherbauer\PycharmProjects\FLEX\data\input_operation\5R1C_validation\OperationScenario_EnergyPrice_var_real.xlsx", engine="openpyxl")
