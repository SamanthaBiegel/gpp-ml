import os, glob
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date

file_paths = glob.glob(os.path.join("../fluxneteo", "*average_cutout*.nc"))
dfs = []

for fp in file_paths:
    ds = nc.Dataset(fp)
    site = ds.site_ID

    tvar = ds.variables["time"]
    cftime_dates = num2date(
        tvar[:],
        units=tvar.units,
        calendar=getattr(tvar, "calendar", "standard")
    )
    dates = pd.to_datetime([d.isoformat() for d in cftime_dates])

    data = {"site": site, "date": dates}
    df = pd.DataFrame(data)

    for var in [
        "RED", "NIR", "BLUE", "GREEN",
        "SWIR1", "SWIR2", "SWIR3",
        "LST_TERRA_Day_VZA0", "LST_TERRA_Night_VZA0"
    ]:
        df[var] = ds.variables[var][:]

    dfs.append(df)
    ds.close()

# 4) concatenate them all
df_all = pd.concat(dfs, ignore_index=True)

sites_in_df = set(df['sitename'].unique())
sites_in_df_all = set(df_all['site'].unique())
missing_sites = sites_in_df - sites_in_df_all
if missing_sites:
    print(f"Sites in df but not in df_all: {missing_sites}")

data = pd.read_csv('data/fdk_v342_ml_old.csv', parse_dates=['TIMESTAMP'])
df_all = df_all.rename(columns={'site': 'sitename', 'date': 'TIMESTAMP'})
data_all = data.merge(df_all, on=['sitename', 'TIMESTAMP'], how='left')
data_all = data_all.dropna()
data_all.to_csv('data/fdk_v342_ml.csv')