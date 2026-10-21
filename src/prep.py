"""Load the raw HF CSV, engineer features, cache as parquet."""
import pandas as pd, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw", "SriLanka-Bird-Diversity-Dataset.csv")
CACHE = os.path.join(ROOT, "cache", "features.parquet")

ENV = ['avg_rad','NDVI','LandCover_Class','elevation_meters','Carbon_Mass','Dust_Mass',
       'SO2_Mass','Sulfate_Mass','Sea_Salt_Mass','Total_Aerosol_Extinction','temp_mean',
       'rainfall','wind_mean','humid_mean','shortwave_radiation','lka_general_2020']
GEO  = ['decimalLatitude','decimalLongitude']
TIME = ['year','month','doy','sin_doy','cos_doy']

def build():
    df = pd.read_csv(RAW)
    d = pd.to_datetime(df.eventDate)
    df['year'] = d.dt.year.astype('int16')
    df['month'] = d.dt.month.astype('int8')
    df['doy'] = d.dt.dayofyear.astype('int16')
    df['sin_doy'] = np.sin(2*np.pi*df.doy/365.25).astype('float32')
    df['cos_doy'] = np.cos(2*np.pi*df.doy/365.25).astype('float32')
    for c in ENV + GEO:
        df[c] = df[c].astype('float32')
    df['individualCount'] = df['individualCount'].astype('float32')
    keep = ['verbatimScientificName','stateProvince','individualCount'] + GEO + ENV + TIME
    df[keep].to_parquet(CACHE, index=False)
    return df[keep]

def load():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    return build()

if __name__ == "__main__":
    df = build()
    print("cached", CACHE, df.shape)
