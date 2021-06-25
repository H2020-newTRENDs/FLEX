# =============================================================================
# needed modules
# =============================================================================
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import CRS,Transformer

# =============================================================================
# global paths and variables
# =============================================================================
base_dir = Path(__file__).parent.resolve()

# =============================================================================
# functions and classes
# =============================================================================
def getcenter(  nuts_id,
                url_nuts0_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_0.geojson",
                url_nuts1_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_1.geojson",
                url_nuts2_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_2.geojson",
                url_nuts3_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_3.geojson"):
    nuts_id = nuts_id.strip()
    if len(nuts_id) == 2:
            url= url_nuts0_poly
    elif len(nuts_id) == 3:
        url = url_nuts1_poly
    elif len(nuts_id) == 4:
        url = url_nuts2_poly
    elif len(nuts_id) > 4:
        url = url_nuts3_poly
    else:
        assert False, f"Error could not identify nuts level for {nuts_id}"
    nuts = gpd.read_file(url)
    transformer = Transformer.from_crs(CRS("EPSG:3035"),CRS("EPSG:4326"))
    point = nuts[nuts.NUTS_ID == nuts_id].centroid.values[0]
    return transformer.transform(point.y,point.x)

def main(**kwargs):

    nuts_id = kwargs.get("nutscode",None)
    if nuts_id is None:
        lat = kwargs.get("lat",None)
        lon = kwargs.get("lon",None)
    else:
        lat,lon = getcenter(nuts_id)

    # % JRC data
    req = f"https://re.jrc.ec.europa.eu/api/tmy?lat={lat}&lon={lon}"
    df = pd.read_csv(req,sep=",",skiprows=list(range(16))+list(range(8777,8790)))[["time(UTC)","T2m"]]
    df["Time"] = pd.to_datetime(df["time(UTC)"],format="%Y%m%d:%H%M")
    # Create a pseudo year
    pseudo_year = 2030
    df["index"] = pd.to_datetime(dict(year=pseudo_year, month=df.Time.dt.month, day=df.Time.dt.day,hour=df.Time.dt.hour))

    return df

# =============================================================================
# main function
# =============================================================================
if __name__ == "__main__":
    print("Main entered")
    # Karlskirche
    lat,lon = 48.198276026158595, 16.371923278715357
    df1 = main(lat=lat,lon=lon)
    # Mittelpunkt von NUTS0 
    nuts0 = "AT"
    df2 = main(nutscode=nuts0)
    # Mittelpunkt von NUTS1
    nuts1 = "AT1"
    df3 = main(nutscode=nuts1)
    # Mittelpunkt von NUTS2
    nuts2 = "AT11"
    df4 = main(nutscode=nuts2)
    # Mittelpunkt von NUTS3
    nuts3 = "AT111"
    df4 = main(nutscode=nuts3)
    print("Main end")
    
    print(df4)
# =============================================================================