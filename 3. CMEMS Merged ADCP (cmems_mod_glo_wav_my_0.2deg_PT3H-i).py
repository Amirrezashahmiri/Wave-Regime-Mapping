#!/usr/bin/env python
# merge_nc_excel.py
# --------------------------------------------------------------------
# Merge CMEMS NetCDF (3-hour) point files with 3-hour ADCP Excel sheets.
# ▸ Inputs are assumed to be in UTC.
# ▸ Both sources are 3-hour, so NO resampling is performed.
# ▸ Merge is EXACT match (Inner Join).
# ▸ Includes a verification report to ensure row counts match exactly.
# ▸ Calculates and reports Bias & Correlation for physical parameter pairs.
# ▸ Generates and saves a professional Correlation Matrix plot.
#
# NEW FEATURES:
# ▸ Adds a 'Season' column (1=Winter, 2=Spring, 3=Summer, 4=Autumn).
# ▸ Extracts and adds 'Depth_CMEMS' from a static bathymetry NetCDF file.
# ▸ Aggregates "Best Match" locations (High Corr, Low Bias) into a single 
#   global Excel file at the end, SORTED BY LOCATION THEN TIME.
# --------------------------------------------------------------------
from __future__ import annotations
import re, pathlib, warnings
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# 1. Folders (UPDATED PATHS)
# --------------------------------------------------------------------
# Input NetCDF files
PATH_NC = pathlib.Path(r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 1\cmems_downloads")

# Input Excel files
PATH_XLSX = pathlib.Path(r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 2\All Locations (Without Code)")

# Static Bathymetry File
PATH_BATHY = pathlib.Path(r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 1\cmems_downloads\Bathymetry Statics\cmems_mod_glo_wav_my_0.2deg_static_1765282455944.nc")

# Output folder for merged files
OUTPUT_DIR = pathlib.Path(r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 3")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# 2. Helpers
# --------------------------------------------------------------------
TAG_RE = re.compile(r"(?P<tag>[A-Za-z0-9_]+_segment_\d+)")

def extract_tag(name: str) -> str | None:
    m = TAG_RE.search(name.replace(" ", "_"))
    return m.group("tag") if m else None

def get_season(month: int) -> int | None:
    """Returns season index: 1=Winter, 2=Spring, 3=Summer, 4=Autumn"""
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    elif month in [9, 10, 11]:
        return 4  # Autumn
    else:
        return None

# --------------------------------------------------------------------
# 3. Build lookup tables & Load Bathymetry
# --------------------------------------------------------------------
nc_files   = {extract_tag(f.name): f for f in PATH_NC.glob("*.nc")     if extract_tag(f.name)}
xlsx_files = {extract_tag(f.name): f for f in PATH_XLSX.glob("*.xls*") if extract_tag(f.name)}
tags = sorted(set(nc_files) & set(xlsx_files))

if not tags:
    print("DEBUG: Searched for NC in:", PATH_NC)
    print("DEBUG: Searched for XLSX in:", PATH_XLSX)
    raise SystemExit("No matching station-segment tags found.")
print(f"Found {len(tags)} matching file pairs.\n")

# Load Global Bathymetry once
print("⚓ Loading Static Bathymetry file...")
try:
    ds_bathy = xr.open_dataset(PATH_BATHY)
    print("   ✅ Bathymetry loaded successfully.\n")
except Exception as e:
    print(f"   ❌ Error loading bathymetry: {e}")
    raise SystemExit("Bathymetry file could not be loaded.")

# --------------------------------------------------------------------
# PRE-LOOP SETUP FOR BEST MATCH AGGREGATION
# --------------------------------------------------------------------
best_match_stats = []     # To store Hm0 stats for every tag
all_merged_frames = {}    # To store the actual DataFrames for later concatenation

# --------------------------------------------------------------------
# 4. Main loop
# --------------------------------------------------------------------
for tag in tags:
    nc_path, xlsx_path = nc_files[tag], xlsx_files[tag]
    print(f"🔄  Processing {tag}...")

    # 5.1 Excel (already 3-hour UTC)
    df_xls = pd.read_excel(xlsx_path, engine="openpyxl")
    if "Date and Time" not in df_xls.columns:
        warnings.warn(f"{xlsx_path.name}: missing 'Date and Time' – skipped.")
        continue
    df_xls["Date and Time"] = pd.to_datetime(df_xls["Date and Time"], errors="coerce")
    df_xls = df_xls.dropna(subset=["Date and Time"]).sort_values("Date and Time")

    # 5.2 NetCDF → tidy DataFrame
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        
        # --- EXTRACT BATHYMETRY FOR THIS STATION ---
        # 1. Get Lat/Lon from the wave file before dropping vars
        st_lat = ds['latitude'].values.item() if 'latitude' in ds else ds['lat'].values.item()
        st_lon = ds['longitude'].values.item() if 'longitude' in ds else ds['lon'].values.item()
        
        # 2. Query the static bathymetry file (Nearest Neighbor)
        try:
            station_depth = ds_bathy['deptho'].sel(latitude=st_lat, longitude=st_lon, method='nearest').values.item()
        except KeyError:
             # Fallback if bathy file uses 'lat'/'lon'
            station_depth = ds_bathy['deptho'].sel(lat=st_lat, lon=st_lon, method='nearest').values.item()
        
        # -------------------------------------------

        # Convert to DataFrame
        df_nc = (
            ds.drop_vars([v for v in ("lat", "lon", "latitude", "longitude") if v in ds])
              .to_dataframe()
              .reset_index()
              .drop(columns=[c for c in ("lat", "lon", "latitude", "longitude") if c in ds.coords])
        )

    # 5.3 Clean datetime (No Resampling)
    df_nc["time"] = pd.to_datetime(df_nc["time"], errors="coerce")
    
    # Remove timezone info if present (to match Excel)
    if df_nc["time"].dt.tz is not None:
        df_nc["time"] = df_nc["time"].dt.tz_localize(None)

    # Just sort and drop duplicates/NaNs
    df_nc = (df_nc.dropna(subset=["time"])
                  .sort_values("time")
                  .drop_duplicates("time"))

    # ----------------------------------------------------------------
    # 5.4 Verification Report (Pre-Merge Check)
    # ----------------------------------------------------------------
    timestamps_xls = set(df_xls["Date and Time"])
    timestamps_nc = set(df_nc["time"])
    
    # Find mismatches
    missing_in_nc = timestamps_xls - timestamps_nc
    missing_in_xls = timestamps_nc - timestamps_xls
    
    n_xls = len(df_xls)
    n_nc = len(df_nc)
    
    print(f"\n   📊 Verification Report for: {tag}")
    print(f"      • Excel Rows (Input):  {n_xls}")
    print(f"      • NetCDF Rows (Input): {n_nc}")
    
    if len(missing_in_nc) == 0 and len(missing_in_xls) == 0 and n_xls == n_nc:
        print("      ✅ PERFECT MATCH: Both files have identical timestamps.")
    else:
        print("      ⚠️  MISMATCH DETECTED:")
        if len(missing_in_nc) > 0:
            print(f"         - {len(missing_in_nc)} rows in Excel have NO MATCH in NetCDF.")
            print(f"           (Example missing in NC: {sorted(list(missing_in_nc))[0]})")
        if len(missing_in_xls) > 0:
            print(f"         - {len(missing_in_xls)} rows in NetCDF have NO MATCH in Excel (Extra data).")

    # ----------------------------------------------------------------
    # 5.5 Align to Excel timeline (EXACT MATCH) & ADD EXTRAS
    # ----------------------------------------------------------------
    # Using 'inner' join ensures only rows with identical timestamps in both files are kept.
    df_merged = pd.merge(
        df_xls,
        df_nc,
        left_on="Date and Time",
        right_on="time",
        how="inner"
    ).drop(columns=["time"])

    # --- ADD SEASON COLUMN ---
    df_merged['Season'] = df_merged['Date and Time'].dt.month.apply(get_season)

    # --- ADD BATHYMETRY COLUMN ---
    df_merged['Depth_CMEMS'] = station_depth

    n_merged = len(df_merged)
    print(f"      • Merged Rows (Output): {n_merged}")
    print(f"      • Added Depth (CMEMS):  {station_depth:.2f} m")

    # Final Safety Check
    if n_merged == n_xls:
        print("      ✅ COMPLETE: No Excel rows were lost in the merge.")
    else:
        diff = n_xls - n_merged
        print(f"      ❌ WARNING: {diff} Excel rows were DROPPED because exact matches weren't found in NetCDF.")

    # ----------------------------------------------------------------
    # 5.6 Physical Validation Report (Bias & Specific Correlations)
    # ----------------------------------------------------------------
    # Map ADCP columns to Copernicus columns for physical validation
    # Format: (ADCP_Column, CMEMS_Column)
    physics_pairs = [
        ("Hm0", "VHM0"),               # Significant Wave Height
        ("Tp", "VTPK"),                # Peak Period
        ("Tm02", "VTM02"),             # Mean Period (Zero-crossing)
        ("Mean Wave Direction", "VMDR") # Mean Direction
    ]

    print(f"\n      🔎 Physical Validation (ADCP vs Model):")
    print(f"         {'Feature Pair':<35} | {'Corr':<8} | {'Bias (Mod-Obs)':<15}")
    print(f"         {'-'*35}-|-{'-'*8}-|-{'-'*15}")

    # Store stats for this specific tag
    current_hm0_corr = 0
    current_hm0_bias = 100 # Default high bias

    for adcp_col, cmems_col in physics_pairs:
        if adcp_col in df_merged.columns and cmems_col in df_merged.columns:
            # Drop NaNs for valid stats
            valid_data = df_merged[[adcp_col, cmems_col]].dropna()
            
            if not valid_data.empty:
                # Calculate Pearson Correlation
                corr = valid_data[adcp_col].corr(valid_data[cmems_col])
                
                # Calculate Bias (Mean Error = Mean(Model) - Mean(Observation))
                bias = (valid_data[cmems_col] - valid_data[adcp_col]).mean()
                
                pair_name = f"{adcp_col} vs {cmems_col}"
                print(f"         {pair_name:<35} | {corr:>.4f}   | {bias:>.4f}")

                # CAPTURE STATS FOR Hm0 (Used for Best Match Logic)
                if adcp_col == "Hm0":
                    current_hm0_corr = corr
                    current_hm0_bias = bias

            else:
                print(f"         {adcp_col} vs {cmems_col:<25} |   N/A    |   N/A (No Data)")
        else:
             # Debug info if columns are missing
             missing = []
             if adcp_col not in df_merged.columns: missing.append(adcp_col)
             if cmems_col not in df_merged.columns: missing.append(cmems_col)
             print(f"         Skipping {adcp_col}/{cmems_col} (Missing cols: {', '.join(missing)})")

    # Save stats to list for final processing
    best_match_stats.append({
        'tag': tag,
        'hm0_corr': current_hm0_corr,
        'hm0_bias': current_hm0_bias
    })
    
    # Store DataFrame in memory for final merge (Add a column for Source Tag)
    df_merged_tagged = df_merged.copy()
    df_merged_tagged.insert(0, "Location_Tag", tag)
    all_merged_frames[tag] = df_merged_tagged

    # 5.7 Diagnostics
    empty_cols = [c for c in df_merged.columns if df_merged[c].isna().all()]
    if empty_cols:
        warnings.warn(f"{tag}: columns with no NetCDF data → {', '.join(empty_cols)}")

    # ----------------------------------------------------------------
    # 5.8 Correlation Matrix Plot (Improved for Many Columns)
    # ----------------------------------------------------------------
    # Select only numeric columns for correlation
    numeric_df = df_merged.select_dtypes(include=[np.number])
    
    # Remove irrelevant index/constant columns to reduce clutter
    cols_to_drop = ["Measurement No"]
    numeric_df = numeric_df.drop(columns=[c for c in cols_to_drop if c in numeric_df.columns], errors='ignore')
    
    # Drop constant columns (std=0) as they don't have correlation
    numeric_df = numeric_df.loc[:, numeric_df.apply(pd.Series.nunique) > 1]

    if not numeric_df.empty and numeric_df.shape[1] > 1:
        # Greatly increased figure size for readability with 20+ columns
        plt.figure(figsize=(24, 20))
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create a mask to hide the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Draw the heatmap
        sns.set_theme(style="white")
        heatmap = sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .7},
            annot_kws={"size": 9}  # Smaller font for numbers to fit in boxes
        )
        
        plt.title(f'Correlation Matrix - {tag}', fontsize=20, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_path = OUTPUT_DIR / f"{tag}_correlation.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"    🎨 Correlation plot saved → {plot_path}")
    else:
        print("    ⚠️  Not enough numeric data for correlation plot.")

    # 5.9 Save result to OUTPUT_DIR
    out_path = OUTPUT_DIR / f"{tag}_merged.xlsx"
    df_merged.to_excel(out_path, index=False, engine="openpyxl")
    print(f"    ✔ saved → {out_path}\n")


# ==============================================================================
# 6. GLOBAL AGGREGATION OF BEST MATCHES (High Correlation & Low Bias)
# ==============================================================================
print("\n" + "="*60)
print("🚀  INITIATING GLOBAL BEST MATCH AGGREGATION")
print("="*60)

# --- CONFIGURATION FOR BEST MATCH ---
MIN_CORR_THRESHOLD = 0.80  # Minimum Correlation required (e.g., 0.8)
MAX_BIAS_THRESHOLD = 0.50  # Maximum Absolute Bias allowed (e.g., +/- 0.5m)

best_match_dfs = []
report_data = []

print(f"🔎  Filtering Criteria: Hm0 Correlation >= {MIN_CORR_THRESHOLD} AND Abs(Hm0 Bias) <= {MAX_BIAS_THRESHOLD}")

# Filter loop
for record in best_match_stats:
    t = record['tag']
    c = record['hm0_corr']
    b = record['hm0_bias']
    
    is_best_match = (c >= MIN_CORR_THRESHOLD) and (abs(b) <= MAX_BIAS_THRESHOLD)
    
    status = "✅ PASS" if is_best_match else "❌ REJECT"
    print(f"    • {t:<25} | Corr: {c:.3f} | Bias: {b:.3f} | {status}")
    
    if is_best_match:
        best_match_dfs.append(all_merged_frames[t])
        report_data.append(record)

# Concatenate and Save
if best_match_dfs:
    print(f"\n📦  Aggregating {len(best_match_dfs)} matching locations...")
    
    global_df = pd.concat(best_match_dfs, ignore_index=True)
    
    # --- FIXED: SORT BY LOCATION THEN TIME ---
    # This ensures that data for each segment/location stays together in a block,
    # rather than being mixed up chronologically across different stations.
    global_df = global_df.sort_values(by=['Location_Tag', 'Date and Time'])
    
    output_global_path = OUTPUT_DIR / "Global_Best_Matches_Merged.xlsx"
    global_df.to_excel(output_global_path, index=False, engine="openpyxl")
    
    print(f"🎉  SUCCESS: Global file saved at:\n    📄 {output_global_path}")
    
    # Save a small text report of which locations were included
    report_path = OUTPUT_DIR / "Global_Best_Matches_Report.txt"
    with open(report_path, "w") as f:
        f.write("Global Best Matches Report\n")
        f.write("==========================\n")
        f.write(f"Criteria: Corr >= {MIN_CORR_THRESHOLD}, Abs(Bias) <= {MAX_BIAS_THRESHOLD}\n\n")
        f.write(f"{'Location Tag':<30} | {'Hm0 Corr':<10} | {'Hm0 Bias':<10}\n")
        f.write("-" * 55 + "\n")
        for r in report_data:
            f.write(f"{r['tag']:<30} | {r['hm0_corr']:<10.4f} | {r['hm0_bias']:<10.4f}\n")
            
else:
    print("\n⚠️  WARNING: No locations met the strict 'Best Match' criteria.")
    print("    Try lowering the thresholds in the script (MIN_CORR_THRESHOLD / MAX_BIAS_THRESHOLD).")

print("🎉 Done.")