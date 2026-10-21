"""Dump small assets the Streamlit app needs, so the app never loads the 403MB CSV."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load, ENV, GEO, TIME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts"); os.makedirs(ART, exist_ok=True)
FEATS = GEO + ENV + TIME + ["individualCount"]

LABELS = {
    "decimalLatitude": "Latitude (°N)", "decimalLongitude": "Longitude (°E)",
    "avg_rad": "Night-time radiance", "NDVI": "NDVI (vegetation index)",
    "LandCover_Class": "Land-cover class", "elevation_meters": "Elevation (m)",
    "Carbon_Mass": "Carbon aerosol mass", "Dust_Mass": "Dust aerosol mass",
    "SO2_Mass": "SO₂ mass", "Sulfate_Mass": "Sulfate mass",
    "Sea_Salt_Mass": "Sea-salt mass", "Total_Aerosol_Extinction": "Aerosol extinction",
    "temp_mean": "Mean temperature (°C)", "rainfall": "Rainfall (mm)",
    "wind_mean": "Wind speed (m/s)", "humid_mean": "Relative humidity (%)",
    "shortwave_radiation": "Shortwave radiation", "lka_general_2020": "Population density baseline",
    "year": "Year", "month": "Month", "doy": "Day of year",
    "sin_doy": "Seasonality (sin)", "cos_doy": "Seasonality (cos)",
    "individualCount": "Individuals counted",
}
GROUPS = {
    "Location": GEO + ["elevation_meters"],
    "Climate": ["temp_mean", "rainfall", "wind_mean", "humid_mean", "shortwave_radiation"],
    "Habitat": ["NDVI", "LandCover_Class", "avg_rad", "lka_general_2020"],
    "Air quality": ["Carbon_Mass", "Dust_Mass", "SO2_Mass", "Sulfate_Mass",
                    "Sea_Salt_Mass", "Total_Aerosol_Extinction"],
    "Observation": ["year", "month", "doy", "individualCount"],
}

if __name__ == "__main__":
    df = load()
    stats = {}
    for c in FEATS:
        s = df[c]
        stats[c] = {"min": float(s.min()), "max": float(s.max()),
                    "median": float(s.median()), "mean": float(s.mean()),
                    "q01": float(s.quantile(0.01)), "q99": float(s.quantile(0.99)),
                    "label": LABELS.get(c, c), "n_unique": int(s.nunique())}
    json.dump({"stats": stats, "groups": GROUPS, "features": FEATS,
               "derived": ["sin_doy", "cos_doy"]},
              open(os.path.join(ART, "feature_stats.json"), "w"), indent=1)

    # a stratified handful of real records, for the "load a real observation" button
    samp = (df.groupby("stateProvince", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 24), random_state=0)))
    cols = FEATS + ["stateProvince", "verbatimScientificName"]
    samp[cols].to_csv(os.path.join(ART, "samples.csv"), index=False)

    # dataset-level facts used by the "About the data" tab
    K = ["decimalLatitude", "decimalLongitude"]
    facts = {
        "n_records": int(len(df)),
        "n_species": int(df.verbatimScientificName.nunique()),
        "n_provinces": int(df.stateProvince.nunique()),
        "year_min": int(df.year.min()), "year_max": int(df.year.max()),
        "n_sites": int(df.groupby(K, observed=True).ngroups),
        "province_share": df.stateProvince.value_counts(normalize=True).round(5).to_dict(),
    }
    json.dump(facts, open(os.path.join(ART, "dataset_facts.json"), "w"), indent=1)
    print("wrote feature_stats.json, samples.csv, dataset_facts.json")
    print("samples:", samp.shape)
