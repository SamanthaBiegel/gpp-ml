import argparse
import os
import glob
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.cwd import apply_cwd_per_site, pet

# Parse arguments 
parser = argparse.ArgumentParser(description='Data pre-processing')

parser.add_argument('-d', '--data_path', type=str,
                    help='Path to the folder containing the FluxDataKit data (in csv format)')

args = parser.parse_args()

fdk_version = 'v342'

# Load data
csv_files = glob.glob(os.path.join(args.data_path, "*_DD_*.csv"))
dataframes = []
for file in csv_files:
    sitename = os.path.basename(file)[4:10]
    df_tmp = pd.read_csv(file, parse_dates=['TIMESTAMP'])
    df_tmp["sitename"] = sitename
    dataframes.append(df_tmp)
df = pd.concat(dataframes, ignore_index=True)

if (fdk_version == 'v342') | (fdk_version == 'v34'):
    csv_files = glob.glob(os.path.join(args.data_path.replace(fdk_version, 'v3'), "*_DD_*.csv"))
    dataframes = []
    for file in csv_files:
        sitename = os.path.basename(file)[4:10]
        df_tmp = pd.read_csv(file, parse_dates=['TIMESTAMP'])
        df_tmp["sitename"] = sitename
        dataframes.append(df_tmp)
    df_v3 = pd.concat(dataframes, ignore_index=True)
    df = df.drop(columns=['LW_IN_F_MDS'])
    df = pd.merge(df, df_v3[['TIMESTAMP', 'LW_IN_F_MDS', 'sitename']], on=['TIMESTAMP', 'sitename'], how='left')
    print(f"Replaced LW_IN_F_MDS with v3 data, number of missing values: {df['LW_IN_F_MDS'].isna().sum()}")

df['year'] = df['TIMESTAMP'].dt.year
df['month'] = df['TIMESTAMP'].dt.month

# Filter out sites with cropland and wetland land use type
sites_meta = pd.read_csv(f"../data/fdk_site_info_{fdk_version}.csv")
sel_sites_vegtype = sites_meta[~sites_meta["igbp_land_use"].isin(["CRO", "WET"])]["sitename"].tolist()
df = df[df.sitename.isin(sel_sites_vegtype)]
print("Nr sites filtered out due to land use: ", len(sites_meta) - len(sel_sites_vegtype))

# Filter out sites with less than 5 years of GPP data
sites_ys = pd.read_csv(f"../data/fdk_site_fullyearsequence_{fdk_version}.csv", parse_dates=["start_gpp", "end_gpp"])
sites_ys['year_end_gpp'] = sites_ys['end_gpp'].apply(lambda x: x.year if x.month == 12 and x.day >= 30 else x.year - 1)
sites_ys['nyears_gpp'] = sites_ys['year_end_gpp'] - sites_ys['year_start_gpp'] + 1
sites_ys["date_start_gpp"] = pd.to_datetime(sites_ys["year_start_gpp"], format='%Y')
sites_ys["date_end_gpp"] = pd.to_datetime(sites_ys["year_end_gpp"] + 1, format='%Y')
minimum_nr_years = 5
merged_df = df.merge(sites_ys[['sitename', 'date_start_gpp', 'date_end_gpp', 'nyears_gpp']], on='sitename', how='left')
filtered_df = merged_df[(merged_df['TIMESTAMP'] >= merged_df['date_start_gpp']) & (merged_df['TIMESTAMP'] < merged_df['date_end_gpp']) & (merged_df['nyears_gpp'] >= minimum_nr_years)]
filtered_df = filtered_df.drop(columns=['date_start_gpp', 'date_end_gpp', 'nyears_gpp'])
print("Nr sites filtered out due to insufficient years of GPP data: ", len(merged_df.sitename.unique()) - len(filtered_df.sitename.unique()))
df = filtered_df.copy()

# Filter out invalid years
valid_years = pd.read_csv("../data/valid_years.csv")
sites_df = df.sitename.unique()
sites_vy = valid_years.Site.unique()
diff = np.setdiff1d(sites_df, sites_vy)
valid_years['years'] = valid_years['end_year'] - valid_years['start_year'] + 1
valid_years['start_date'] = pd.to_datetime(valid_years['start_year'], format='%Y')
valid_years['end_date'] = pd.to_datetime(valid_years['end_year'] + 1, format='%Y')
merged_df = df.merge(valid_years[['Site', 'start_date', 'end_date', 'years']], left_on='sitename', right_on='Site', how='left')
merged_df = merged_df.fillna({'start_date': pd.to_datetime('1900-01-01'), 'end_date': pd.to_datetime('2100-01-01'), 'years': 5})
filtered_df = merged_df[(merged_df['TIMESTAMP'] >= merged_df['start_date']) & (merged_df['TIMESTAMP'] < merged_df['end_date']) & (merged_df['years'] >= 5)]
filtered_df = filtered_df.drop(columns=['start_date', 'end_date', 'Site'])
filtered_df['year'] = filtered_df['TIMESTAMP'].dt.year
print("Nr of observations filtered out due to invalid years: ", len(merged_df) - len(filtered_df))
df = filtered_df.copy()

# Check start and end dates per site
def test_start_end_date(df):
    invalid_dates = []
    for site, group in df.groupby('sitename'):
        start_date = group['TIMESTAMP'].min()
        end_date = group['TIMESTAMP'].max()
        if start_date.day != 1 or start_date.month != 1 or (end_date.day != 31 and end_date.day != 30) or end_date.month != 12:
            invalid_dates.append(site)
    print("Sites with invalid start/end dates: ", invalid_dates)
test_start_end_date(df)

# Remove years with gaps
def remove_years_with_gaps(df, site_column='sitename', year_column='year', target_column='GPP_NT_VUT_REF', max_gap_length=0):
    to_drop = []
    for (site, year), group in df.groupby([site_column, year_column]):
        nan_sequences = group[target_column].isna().astype(int).groupby(group[target_column].notna().astype(int).cumsum()).sum()
        if nan_sequences.max() > max_gap_length:
            to_drop.append((site, year))
    if to_drop:
        for site, year in to_drop:
            df = df[~((df[site_column] == site) & (df[year_column] == year))]
    print("Nr years filtered out due to long gaps: ", len(to_drop))
    return df
df = remove_years_with_gaps(df)

# Filter out sites with less than 5 years of valid GPP data
def filters_sites_with_nans(df):
    """ Filter out sites with less than 5 years of valid GPP data """
    sites_to_remove = []
    for site, group in df.groupby('sitename'):
        df_gpp = group.dropna(subset=['GPP_NT_VUT_REF'])
        max_timestamp = df_gpp['TIMESTAMP'].max()
        min_timestamp = df_gpp['TIMESTAMP'].min()
        if (max_timestamp.year - min_timestamp.year) < 4:
            sites_to_remove.append(site)
    print("Nr sites filtered out due to less than 5 years of valid GPP data: ", len(sites_to_remove))
    return df[~df.sitename.isin(sites_to_remove)]
df = filters_sites_with_nans(df)

def fdk_impute_knn(input_df, target, predictors, k=5):
    features = predictors + [target]
    impute_df = input_df.copy()

    if impute_df[target].isna().all():
        print(f"Target variable {target} is missing for all rows, skipping imputation.")
        return input_df
    
    # Scale all features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(impute_df[features])
    
    # Apply KNN Imputation
    imputer = KNNImputer(n_neighbors=k)
    features_imputed = imputer.fit_transform(features_scaled)
    
    # Inverse transform to get back to original scale
    features_unscaled = scaler.inverse_transform(features_imputed)
    imputed_df = pd.DataFrame(features_unscaled, columns=features, index=impute_df.index)
    
    # Update only the target variable
    impute_df[target] = imputed_df[target]
    
    # Mark imputed values
    impute_df[f"{target}_is_imputed"] = input_df[target].isna()
    
    return impute_df

df = fdk_impute_knn(df, 'NETRAD', ['SW_IN_F_MDS', 'LW_IN_F_MDS'], k=5)

df['PET'] = 60 * 60 * 24 * pet(df['NETRAD'], df['TA_F_MDS'], df['PA_F']*1000)
site_totals = df.groupby('sitename').agg({'PET': 'sum', 'P_F': 'sum'})
site_totals['ai'] = site_totals['P_F'] / site_totals['PET']
df = pd.merge(df, site_totals[['ai']], on='sitename', how='left')

# Get soil moisture
df_soil_moisture = pd.read_csv(f"../data/soil_moisture_{fdk_version}.csv", parse_dates=['date'])
df = df.merge(df_soil_moisture, left_on=['sitename', 'TIMESTAMP'], right_on=['sitename', 'date'], how='left')
print("Nr sites without soil moisture data: ", df.groupby('sitename')['wscal'].apply(lambda x: x.isna().all()).sum())
print("Fraction of missing soil moisture data: ", df['wscal'].isna().sum() / len(df), ", imputing...")
dfs = []
for sitename, group in tqdm(df.groupby('sitename')):
    imputed_group = fdk_impute_knn(group, 'wscal', ['P_F', 'PET'], k=5)
    dfs.append(imputed_group)
df = pd.concat(dfs)
df['wscal'] = df['wscal'].fillna(df['wscal'].mean())

# Compute potential cumulative water deficit
# Following approach from: https://github.com/geco-bern/cwd
monthly_precip = df.groupby(['sitename', 'year', 'month'])['P_F'].sum().reset_index()
wettest_month = monthly_precip.groupby(['sitename', 'month'])['P_F'].mean().reset_index()
wettest_month = wettest_month.loc[wettest_month.groupby('sitename')['P_F'].idxmax()]
wettest_month = wettest_month[['sitename', 'month']].rename(columns={'month': 'wettest_month'})
df = df.merge(wettest_month, on='sitename', how='left')
df['month_reset'] = (df['wettest_month'] % 12) + 1
df['year_reset'] = df['year'] + (df['month_reset'] == 1).astype(int)
df['doy_reset'] = pd.to_datetime(df['year_reset'].astype(str) + '-' + df['month_reset'].astype(str) + '-01').dt.dayofyear
df['pwbal'] = df['P_F'] - df['PET']
df = df.groupby('sitename', group_keys=False).apply(apply_cwd_per_site)
df = df.rename(columns={'cwd': 'pcwd'})

# Impute missing air pressure values
sites_with_low_pa = df[df['PA_F']<60]['sitename'].unique()
print("Sites with low air pressure: ", sites_with_low_pa)
dfs = []
for site in sites_with_low_pa:
    group = df[df['sitename']==site]
    group.loc[group['PA_F']<60, 'PA_F'] = np.nan
    group = fdk_impute_knn(group, 'PA_F', ['TA_F_MDS', 'VPD_DAY_F_MDS', 'P_F', 'WS_F'], k=5)
    dfs.append(group)
if len(dfs) > 0:
    df = pd.concat([df[~df['sitename'].isin(sites_with_low_pa)]] + dfs)

df_ml = df[["TIMESTAMP", "TA_F_MDS", "TA_DAY_F_MDS", "SW_IN_F_MDS", "LW_IN_F_MDS", "VPD_DAY_F_MDS", "PA_F", "P_F", "WS_F", "FPAR", "ai", "pcwd", "wscal", "GPP_NT_VUT_REF", "sitename"]].copy()

print("Total number of sites: ", len(df_ml.sitename.unique()))

# Add metadata
sites_meta = pd.read_csv(f"../data/fdk_site_info_{fdk_version}.csv")
df_ml = df_ml.merge(sites_meta[['sitename', 'koeppen_code', 'igbp_land_use', 'whc']], left_on="sitename", right_on="sitename", how="left")

df_ml = df_ml.sort_values(by=["sitename", "TIMESTAMP"])

df_ml.to_csv(f"../data/fdk_{fdk_version}_ml.csv", index=False)