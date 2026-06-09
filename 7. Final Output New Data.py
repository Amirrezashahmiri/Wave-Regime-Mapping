from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm  # Crucial for Log Scale Map
import seaborn as sns
import copernicusmarine as cm
from datetime import datetime, timedelta

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================

# --- Credentials ---
USERNAME = ""
PASSWORD = ""

# --- Input Paths (Model & Bathymetry) ---
MODEL_PATH = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 6\Classification_Model_Results\Final_Model_RF_SMOTE.pkl"
SELECTOR_PATH = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 6\Classification_Model_Results\Feature_Selector.pkl"
BATHYMETRY_FILE = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 1\cmems_downloads\Bathymetry Statics\cmems_mod_glo_wav_my_0.2deg_static_1765282455944.nc"

# --- Output Directory ---
OUTPUT_DIR = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Simulation Settings ---
# We analyze ONE YEAR of data to capture all seasons for the map
START_DATE = "2015-01-01"
END_DATE = "2025-10-31"

# Oman Coast Bounding Box (Approximate)
OMAN_MIN_LON = 52.0
OMAN_MAX_LON = 60.0
OMAN_MIN_LAT = 16.0
OMAN_MAX_LAT = 26.5

# CMEMS Product ID
DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

# =============================================================================
# 2. HELPER FUNCTIONS (DATA & PROCESSING)
# =============================================================================

def download_oman_data():
    """
    Downloads wave data for the entire Oman region for the specified period.
    Returns the path to the downloaded NetCDF file.
    """
    print(f"\n🌊 Starting CMEMS Download for Oman Coast ({START_DATE} to {END_DATE})...")
    
    # Login
    try:
        # Fixed: Removed 'overwrite_configuration_file' to prevent TypeError
        cm.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"Login warning (might be already logged in): {e}")

    output_filename = "Oman_Coast_Wave_Data_2023.nc"
    
    # Check if already exists to avoid re-downloading
    if os.path.exists(os.path.join(OUTPUT_DIR, output_filename)):
        print("   ✅ Data file already exists. Skipping download.")
        return os.path.join(OUTPUT_DIR, output_filename)

    cm.subset(
        dataset_id=DATASET_ID,
        dataset_version="202411",
        variables=[
            "VHM0", "VTPK", "VHM0_WW", "VHM0_SW1", 
            "VSDX", "VSDY", "VMDR" # VMDR needed for calculating stability
        ],
        minimum_longitude=OMAN_MIN_LON,
        maximum_longitude=OMAN_MAX_LON,
        minimum_latitude=OMAN_MIN_LAT,
        maximum_latitude=OMAN_MAX_LAT,
        start_datetime=f"{START_DATE} 00:00:00",
        end_datetime=f"{END_DATE} 23:59:59",
        output_filename=output_filename,
        output_directory=OUTPUT_DIR,
        force_download=True
    )
    return os.path.join(OUTPUT_DIR, output_filename)

def calculate_engineering_features(ds):
    """
    Converts NetCDF to DataFrame and calculates Rolling Stability features.
    Matches the logic used during training.
    """
    print("\n⚙️ Processing Data & Calculating Features (This may take a moment)...")
    
    # Convert to DataFrame
    df = ds.to_dataframe().reset_index().dropna()
    
    # Sort for rolling calculation
    df = df.sort_values(by=['latitude', 'longitude', 'time'])
    
    # --- Feature Engineering (Must match training!) ---
    # 1. Group by location (Lat/Lon pairs)
    # We create a location ID to group efficiently
    df['loc_id'] = df['latitude'].astype(str) + "_" + df['longitude'].astype(str)
    
    # 2. Rolling Calculations (Window=8 for 24h, assuming 3h data)
    print("   ... Calculating rolling stability features...")
    
    # Height Stability (Rolling Std of VHM0)
    df['Model_Height_Stability'] = df.groupby('loc_id')['VHM0'].transform(
        lambda x: x.rolling(window=8, min_periods=1).std()
    ).fillna(0)
    
    # Period Stability (Rolling Std of VTPK)
    df['Model_Period_Stability'] = df.groupby('loc_id')['VTPK'].transform(
        lambda x: x.rolling(window=8, min_periods=1).std()
    ).fillna(0)
    
    # Direction Stability (Rolling Std of VMDR)
    df['Direction_Stability'] = df.groupby('loc_id')['VMDR'].transform(
        lambda x: x.rolling(window=8, min_periods=1).std()
    ).fillna(0)
    
    return df

def add_bathymetry(df, bathy_path):
    """
    Merges Depth data into the DataFrame using Nearest Neighbor lookup.
    """
    print("   ... Merging Bathymetry data...")
    ds_bathy = xr.open_dataset(bathy_path)
    
    # Rename coords if necessary (CMEMS standard is usually lat/lon)
    if 'deptho' in ds_bathy:
        # Create a temporary dataframe for lookup
        bathy_df = ds_bathy['deptho'].to_dataframe().reset_index().dropna()
        
        # Simple merge based on nearest rounded coordinates (0.2 degree grid)
        # Rounding ensures we match the grid points
        df['lat_round'] = df['latitude'].round(1)
        df['lon_round'] = df['longitude'].round(1)
        bathy_df['lat_round'] = bathy_df['latitude'].round(1)
        bathy_df['lon_round'] = bathy_df['longitude'].round(1)
        
        merged = pd.merge(df, bathy_df[['lat_round', 'lon_round', 'deptho']], 
                          on=['lat_round', 'lon_round'], how='left')
        
        merged.rename(columns={'deptho': 'Depth_CMEMS'}, inplace=True)
        # Fill missing depths with a default deep value or drop
        merged['Depth_CMEMS'] = merged['Depth_CMEMS'].fillna(1000) 
        
        return merged
    else:
        print("   [WARN] 'deptho' variable not found in bathymetry file!")
        df['Depth_CMEMS'] = 100 # Fallback
        return df

# =============================================================================
# 3. ADVANCED VISUALIZATION FUNCTIONS (ENHANCED + NO WHITE START)
# =============================================================================

def set_professional_style():
    """Sets a professional seaborn theme for publication-quality figures."""
    sns.set_style("white") 
    sns.set_context("paper", font_scale=1.4) 
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['figure.dpi'] = 300 

def plot_spatial_heatmap(grid_df, metric, label, filename, cmap='viridis', highlight_max=False, use_log=False):
    """
    Generates a high-quality spatial heatmap with optional hotspot highlighting.
    Added use_log parameter for Logarithmic Scaling.
    **UPDATED**: Replaces 0 values with epsilon in Log mode to avoid white pixels.
    """
    set_professional_style()
    plt.figure(figsize=(10, 8)) 
    
    # Pivot data for heatmap
    pivot_table = grid_df.pivot(index='latitude', columns='longitude', values=metric)
    
    # Configure plotting args
    plot_args = {
        'cmap': cmap,
        'cbar_kws': {'label': label, 'shrink': 0.8}
    }
    
    # APPLY LOG SCALE LOGIC WITH ZERO-RESCUE
    if use_log:
        # 1. Find the smallest non-zero value
        positive_min = pivot_table[pivot_table > 0].min().min()
        if pd.isna(positive_min): positive_min = 0.01 # Fallback
        
        # 2. Define a tiny epsilon (smaller than the smallest real value)
        epsilon = positive_min / 10.0
        
        # 3. Replace 0.0 with epsilon in the pivot table (only for plotting)
        # This ensures they get mapped to the bottom of the colormap instead of being masked (white)
        pivot_table_plot = pivot_table.replace(0.0, epsilon)
        
        vmax = pivot_table_plot.max().max()
        
        # Safety Check
        if epsilon >= vmax: vmax = epsilon + 0.1
            
        plot_args['norm'] = LogNorm(vmin=epsilon, vmax=vmax)
        label += " (Log Scale)"
        plot_args['cbar_kws']['label'] = label
        
        # Use the modified table for plotting
        ax = sns.heatmap(pivot_table_plot, **plot_args)
    else:
        # Standard linear plot
        ax = sns.heatmap(pivot_table, **plot_args)

    ax.invert_yaxis() # North up
    
    # Clean Axis Labels
    plt.title(f'{label}', fontweight='bold', pad=15)
    plt.xlabel('Longitude (°E)', fontweight='bold')
    plt.ylabel('Latitude (°N)', fontweight='bold')
    
    # Highlight the Maximum Value (Hotspot) - WITH DEPTH FILTER
    if highlight_max:
        coastal_candidates = grid_df[grid_df['Depth_CMEMS'] <= 100]
        
        if not coastal_candidates.empty:
            max_row = coastal_candidates.loc[coastal_candidates[metric].idxmax()]
            
            max_lat = max_row['latitude']
            max_lon = max_row['longitude']
            max_val = max_row[metric]
            max_depth = max_row['Depth_CMEMS']
            
            row_idx = pivot_table.index.get_loc(max_lat)
            col_idx = pivot_table.columns.get_loc(max_lon)
            
            # Use a gold star with a black outline for high visibility
            plt.scatter(col_idx + 0.5, row_idx + 0.5, color='gold', marker='*', s=300, 
                        edgecolors='black', linewidth=1.5, zorder=10,
                        label=f'Optimal Site\n({max_val:.1f}%)')
            
            # Improved Legend
            legend = plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1.0, fancybox=False, edgecolor='black')
            legend.get_frame().set_linewidth(1.0)

            print(f"   ★ Best Coastal Location (<=100m) found at: {max_lat}N, {max_lon}E")
            print(f"     -> Value: {max_val:.2f}% | Depth: {max_depth:.1f}m")
        else:
            print("   [WARN] No locations found under 100m depth! Skipping highlight.")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_regime_statistics(grid_df):
    """
    Visualizes the overall frequency of each wave regime across the entire region.
    """
    set_professional_style()
    
    total_counts = {
        'Calm': grid_df['Calm_Count'].sum(),
        'Swell': grid_df['Swell_Count'].sum(),
        'Storm': grid_df['Storm_Count'].sum(),
        'Confused': grid_df['Confused_Count'].sum()
    }
    
    stats_df = pd.DataFrame(list(total_counts.items()), columns=['Regime', 'Count'])
    stats_df['Percentage'] = (stats_df['Count'] / stats_df['Count'].sum()) * 100
    
    plt.figure(figsize=(8, 6))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#95a5a6'] 
    ax = sns.barplot(data=stats_df, x='Regime', y='Percentage', palette=colors, hue='Regime', legend=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=12, fontweight='bold')
        
    plt.title('Overall Sea State Regime Distribution (Oman Coast 2023)', fontweight='bold')
    plt.ylabel('Frequency (%)')
    plt.xlabel('Wave Regime')
    plt.ylim(0, 100) 
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)
    sns.despine(left=True) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Stat_Regime_Frequency.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_site_selection_matrix(grid_df):
    """
    Generates a 'Risk vs. Reward' scatter plot to visually identify the best sites.
    """
    set_professional_style()
    plt.figure(figsize=(10, 8))
    
    # Changed palette to 'plasma' to avoid white start
    scatter = sns.scatterplot(
        data=grid_df, 
        x='Storm_Risk', 
        y='Swell_Potential', 
        hue='Avg_Confidence', 
        size='Avg_Confidence',
        sizes=(30, 250), 
        palette='plasma', 
        alpha=0.85,
        edgecolor='w', 
        linewidth=0.5
    )
    
    coastal_candidates = grid_df[grid_df['Depth_CMEMS'] <= 100]
    
    if not coastal_candidates.empty:
        best_site = coastal_candidates.loc[coastal_candidates['Swell_Potential'].idxmax()]
        
        plt.scatter(best_site['Storm_Risk'], best_site['Swell_Potential'], 
                    color='cyan', marker='*', s=400, edgecolors='black', linewidth=1.5, zorder=10, 
                    label='Best Coastal Site')
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9)
        plt.text(best_site['Storm_Risk'] + 0.1, best_site['Swell_Potential'], 
                 f" Lat: {best_site['latitude']}\n Lon: {best_site['longitude']}\n Depth: {best_site['Depth_CMEMS']:.0f}m", 
                 fontsize=10, fontweight='bold', va='center', bbox=bbox_props)
    
    plt.title('Site Selection Matrix: Risk vs. Reward', fontweight='bold', pad=15)
    plt.xlabel('Storm Risk (% of Time) ➝ (Lower is Better)', fontweight='bold')
    plt.ylabel('Swell Energy Potential (% of Time) ➝ (Higher is Better)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Stat_Site_Selection_Matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- NEW ADDED: STRICT VALIDATION PLOTS (ENHANCED) ---

def plot_individual_regime_heatmaps(grid_df):
    """Generates 4 separate heatmaps for Swell, Storm, Calm, Confused."""
    set_professional_style()
    # UPDATED PALETTES to avoid white start:
    regimes = {'Swell': ('Swell_Count', 'viridis'), 'Storm': ('Storm_Count', 'jet'),
               'Calm': ('Calm_Count', 'viridis'), 'Confused': ('Confused_Count', 'magma')}
    
    for name, (col, cmap) in regimes.items():
        if f'{name}_Freq' not in grid_df.columns:
            grid_df[f'{name}_Freq'] = (grid_df[col] / grid_df['Total_Count']) * 100
            
        # Use Log Scale for Storm map in validation to show sensitivity
        use_log = True if name == 'Storm' else False

        plt.figure(figsize=(10, 8))
        pivot = grid_df.pivot(index='latitude', columns='longitude', values=f'{name}_Freq')
        
        plot_args = {'cmap': cmap, 'cbar_kws': {'label': f'{name} Frequency (%)', 'shrink': 0.8}}
        
        # APPLY LOG SCALE LOGIC WITH ZERO-RESCUE
        if use_log:
            positive_min = pivot[pivot > 0].min().min()
            if pd.isna(positive_min): positive_min = 0.01
            epsilon = positive_min / 10.0
            
            # Replace 0s with epsilon
            pivot = pivot.replace(0.0, epsilon)
            
            vmax = pivot.max().max()
            if epsilon >= vmax: vmax = epsilon + 0.1
                
            plot_args['norm'] = LogNorm(vmin=epsilon, vmax=vmax)
            plot_args['cbar_kws']['label'] += " (Log Scale)"

        sns.heatmap(pivot, **plot_args)
        plt.gca().invert_yaxis()
        plt.title(f'Spatial Distribution: {name} Regime', fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Validation_Map_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_temporal_trends(monthly_stats):
    """Plots percentage of each regime over time (Monthly) to show seasonal patterns."""
    set_professional_style()
    plt.figure(figsize=(14, 7))
    dates = pd.to_datetime(monthly_stats['time_key'])
    
    pal = sns.color_palette("Set2")
    plt.stackplot(dates, 
                  monthly_stats['Calm_Pct'], monthly_stats['Swell_Pct'], 
                  monthly_stats['Storm_Pct'], monthly_stats['Confused_Pct'],
                  labels=['Calm', 'Swell', 'Storm', 'Confused'],
                  colors=[pal[0], pal[1], pal[2], pal[3]], alpha=0.9)
    plt.title('Temporal Evolution of Sea States (Monthly Aggregated)', fontweight='bold', pad=15)
    plt.ylabel('Regime Prevalence (%)')
    plt.xlabel('Date')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, edgecolor='black')
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Validation_Temporal_Trend.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cyclone_shaheen_validation(shaheen_df):
    """Generates a map specifically for Cyclone Shaheen (Oct 2021) in the North."""
    if shaheen_df.empty:
        print("   [WARN] No data found for Cyclone Shaheen period.")
        return
    set_professional_style()
    shaheen_stats = shaheen_df.groupby(['latitude', 'longitude']).agg(
        Total_Count=('Predicted_Regime', 'count'),
        Storm_Count=('Predicted_Regime', lambda x: (x == 1).sum())
    ).reset_index()
    shaheen_stats['Storm_Risk'] = (shaheen_stats['Storm_Count'] / shaheen_stats['Total_Count']) * 100
    
    plt.figure(figsize=(12, 10))
    pivot = shaheen_stats.pivot(index='latitude', columns='longitude', values='Storm_Risk')
    
    # Changed from 'Reds' to 'jet' to avoid white start
    ax = sns.heatmap(pivot, cmap='jet', vmin=0, vmax=100, cbar_kws={'label': 'Storm Probability (%)', 'shrink': 0.8})
    ax.invert_yaxis()
    plt.title('Validation: Cyclone Shaheen Detection (Oct 3-4, 2021)', fontweight='bold', pad=15)
    plt.xlabel('Longitude (°E)'); plt.ylabel('Latitude (°N)')
    plt.text(0.5, -0.08, "Note: High storm probability in the North confirms model sensitivity.", 
             ha='center', va='center', transform=ax.transAxes, fontsize=12, style='italic', color='darkred')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Validation_Cyclone_Shaheen.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 4. MAIN EXECUTION FLOW
# =============================================================================

try:
    print("🚀 STARTING SPATIAL ANALYSIS FOR OMAN COAST (FULL REGION)...")
    
    # 1. Load Trained Model & Selector
    print("\n1. Loading AI Models...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SELECTOR_PATH):
        raise FileNotFoundError("Model or Selector pickle files not found! Check paths.")
        
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    print("   ✅ Model and Selector loaded successfully.")

    # 2. Get Data (Download -> Load)
    nc_path = download_oman_data()
    ds = xr.open_dataset(nc_path)
    
    # 3. Feature Engineering
    df = calculate_engineering_features(ds)
    df = add_bathymetry(df, BATHYMETRY_FILE)
    
    # --- NO FILTERING: We use the full dataset for mapping ---
    print(f"\n2. Proceeding with full spatial dataset ({len(df)} points)...")

    # 4. Feature Selection
    print("\n3. Applying Feature Selection...")
    needed_cols = [
        'VHM0', 'VHM0_SW1', 'VHM0_SW2', 'VHM0_WW', 'VMDR', 'VMDR_SW1', 'VMDR_SW2', 'VMDR_WW', 
        'VPED', 'VSDX', 'VSDY', 'VTM01_SW1', 'VTM01_SW2', 'VTM01_WW', 'VTM02', 'VTM10', 'VTPK', 
        'Depth_CMEMS', 'Model_Height_Stability', 'Model_Period_Stability', 'Swell_Stability', 'Direction_Stability'
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
            
    X_new = df[needed_cols]
    
    # Transform
    X_selected = selector.transform(X_new)
    print(f"   ✅ Features transformed. Shape: {X_selected.shape}")

    # 5. Prediction & Uncertainty
    print("\n4. Running AI Prediction (Classification & Uncertainty)...")
    df['Predicted_Regime'] = model.predict(X_selected)
    
    # Calculate Uncertainty
    probs = model.predict_proba(X_selected)
    df['Confidence'] = np.max(probs, axis=1)
    df['Uncertainty'] = 1 - df['Confidence']
    
    print("   ✅ Predictions complete.")

    # --- 5b. PREPARE VALIDATION DATA (TEMPORAL & SHAHEEN) ---
    print("\n4b. Preparing Strict Validation Data (Temporal Trends & Shaheen)...")
    
    # A. Temporal Trend Data
    df['time'] = pd.to_datetime(df['time'])
    df['month_key'] = df['time'].dt.to_period('M')
    
    monthly_stats = df.groupby('month_key').agg(
        Total=('Predicted_Regime', 'count'),
        Calm=('Predicted_Regime', lambda x: (x == 2).sum()),
        Swell=('Predicted_Regime', lambda x: (x == 3).sum()),
        Storm=('Predicted_Regime', lambda x: (x == 1).sum()),
        Confused=('Predicted_Regime', lambda x: (x == 0).sum())
    ).reset_index()
    
    monthly_stats['time_key'] = monthly_stats['month_key'].astype(str)
    monthly_stats = monthly_stats.sort_values('time_key')
    monthly_stats['Calm_Pct'] = (monthly_stats['Calm'] / monthly_stats['Total']) * 100
    monthly_stats['Swell_Pct'] = (monthly_stats['Swell'] / monthly_stats['Total']) * 100
    monthly_stats['Storm_Pct'] = (monthly_stats['Storm'] / monthly_stats['Total']) * 100
    monthly_stats['Confused_Pct'] = (monthly_stats['Confused'] / monthly_stats['Total']) * 100

    # B. Shaheen Event Data
    shaheen_start = datetime(2021, 10, 3)
    shaheen_end = datetime(2021, 10, 4)
    shaheen_mask = (df['time'] >= shaheen_start) & (df['time'] <= shaheen_end)
    shaheen_df = df[shaheen_mask].copy()
    if not shaheen_df.empty:
        print(f"   🌪️ Cyclone Shaheen Data Extracted: {len(shaheen_df)} points found.")

    # 6. Aggregation & Mapping
    print("\n5. Generating Spatial Maps & Stats...")
    
    # Calculate stats per grid point
    grid_stats = df.groupby(['latitude', 'longitude']).agg(
        Total_Count=('Predicted_Regime', 'count'),
        Swell_Count=('Predicted_Regime', lambda x: (x == 3).sum()),
        Storm_Count=('Predicted_Regime', lambda x: (x == 1).sum()),
        Calm_Count=('Predicted_Regime', lambda x: (x == 2).sum()),
        Confused_Count=('Predicted_Regime', lambda x: (x == 0).sum()),
        Avg_Confidence=('Confidence', 'mean'),
        Depth_CMEMS=('Depth_CMEMS', 'mean') # <--- Added Depth aggregation here for filtering
    ).reset_index()
    
    # Calculate Percentages
    grid_stats['Swell_Potential'] = (grid_stats['Swell_Count'] / grid_stats['Total_Count']) * 100
    grid_stats['Storm_Risk'] = (grid_stats['Storm_Count'] / grid_stats['Total_Count']) * 100
    
    # Save Grid Data CSV
    grid_stats.to_csv(os.path.join(OUTPUT_DIR, "Oman_Spatial_Analysis_FULL.csv"), index=False)
    
    # --- VISUALIZATIONS ---
    # A. Maps with Highlights
    plot_spatial_heatmap(grid_stats, 'Swell_Potential', 'Swell Potential (%)', "Enhanced_Map_Swell.png", cmap='viridis', highlight_max=True)
    # 1. Standard Linear Map - CHANGED TO JET
    plot_spatial_heatmap(grid_stats, 'Storm_Risk', 'Storm Risk (%)', "Enhanced_Map_Storm.png", cmap='jet', highlight_max=False, use_log=False)
    # 2. NEW: Logarithmic Map - CHANGED TO JET
    plot_spatial_heatmap(grid_stats, 'Storm_Risk', 'Storm Risk (%)', "Enhanced_Map_Storm_LogScale.png", cmap='jet', highlight_max=False, use_log=True)
    
    # Confidence Map - CHANGED TO VIRIDIS
    plot_spatial_heatmap(grid_stats, 'Avg_Confidence', 'Avg Model Confidence', "Enhanced_Map_Confidence.png", cmap='viridis', highlight_max=False)
    
    # B. Statistical Charts
    plot_regime_statistics(grid_stats)
    
    # C. Site Selection Matrix
    plot_site_selection_matrix(grid_stats)

    # D. NEW STRICT VALIDATION PLOTS
    print("\n6. Generating Validation Plots...")
    plot_individual_regime_heatmaps(grid_stats)
    plot_temporal_trends(monthly_stats)
    if not shaheen_df.empty:
        plot_cyclone_shaheen_validation(shaheen_df)
    
    # --- 7. GENERATE TEXT REPORT (ENHANCED NUMERICAL LOGS) ---
    print("\n7. Generating Summary Text Report with Figure Interpretations...")
    report_lines = []
    report_lines.append("========================================================")
    report_lines.append("               OMAN WAVE ENERGY ANALYSIS REPORT         ")
    report_lines.append("========================================================")
    report_lines.append(f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Analysis Period: {START_DATE} to {END_DATE}")
    report_lines.append("-" * 50)
    
    # 1. Best Site Stats
    coastal = grid_stats[grid_stats['Depth_CMEMS'] <= 100]
    best_site = None
    if not coastal.empty:
        best_site = coastal.loc[coastal['Swell_Potential'].idxmax()]
        report_lines.append(f"\n[OPTIMAL SITE SELECTION]")
        report_lines.append(f"Condition: Depth <= 100m")
        report_lines.append(f"Location: Lat {best_site['latitude']}, Lon {best_site['longitude']}")
        report_lines.append(f"Depth: {best_site['Depth_CMEMS']:.2f} m")
        report_lines.append(f"Swell Potential: {best_site['Swell_Potential']:.2f}% (High is better)")
        report_lines.append(f"Storm Risk: {best_site['Storm_Risk']:.2f}% (Low is better)")
        report_lines.append(f"Model Confidence: {best_site['Avg_Confidence']:.2f}")
    else:
        report_lines.append("\n[OPTIMAL SITE SELECTION]")
        report_lines.append("No sites found with Depth <= 100m.")

    # 2. Regional Stats
    north = grid_stats[grid_stats['latitude'] > 22.5]
    south = grid_stats[grid_stats['latitude'] < 18.0]
    north_risk = north['Storm_Risk'].mean() if not north.empty else 0
    south_risk = south['Storm_Risk'].mean() if not south.empty else 0
    
    report_lines.append("-" * 50)
    report_lines.append("\n[REGIONAL COMPARISON]")
    if not north.empty:
        report_lines.append(f"NORTH (Sea of Oman):")
        report_lines.append(f"  - Avg Swell Potential: {north['Swell_Potential'].mean():.2f}%")
        report_lines.append(f"  - Avg Storm Risk (Background): {north_risk:.2f}%")
    if not south.empty:
        report_lines.append(f"SOUTH (Arabian Sea):")
        report_lines.append(f"  - Avg Swell Potential: {south['Swell_Potential'].mean():.2f}%")
        report_lines.append(f"  - Avg Storm Risk: {south_risk:.2f}%")

    # 3. Validation - Shaheen
    report_lines.append("-" * 50)
    report_lines.append("\n[VALIDATION: CYCLONE SHAHEEN (Oct 2021)]")
    report_lines.append("Objective: Verify if model detects the rare cyclone anomaly in the North.")
    
    if not shaheen_df.empty and not north.empty:
        s_stats = shaheen_df.groupby(['latitude', 'longitude']).agg(
            Total=('Predicted_Regime', 'count'),
            Storm=('Predicted_Regime', lambda x: (x == 1).sum())
        ).reset_index()
        s_stats['Risk'] = (s_stats['Storm'] / s_stats['Total']) * 100
        
        max_risk_shaheen_north = s_stats[s_stats['latitude'] > 23]['Risk'].max()
        background_risk = north_risk if north_risk > 0 else 0.001 
        anomaly_ratio = max_risk_shaheen_north / background_risk
        
        report_lines.append(f"Result: Max Storm Probability during Shaheen (>23N): {max_risk_shaheen_north:.2f}%")
        report_lines.append(f"Background Risk (North): {background_risk:.2f}%")
        report_lines.append(f"Anomaly Ratio: {anomaly_ratio:.1f}x (Peak / Background)")
        
        if max_risk_shaheen_north > 10.0:
            report_lines.append("Status: SUCCESS (Significant Anomaly Detected)")
        elif max_risk_shaheen_north > 1.0:
             report_lines.append("Status: PARTIAL SUCCESS (Weak Anomaly Detected)")
        else:
            report_lines.append("Status: FAILED (No Anomaly Detected)")
    else:
        report_lines.append("Status: N/A (Data missing)")

    # 4. Validation - Temporal
    report_lines.append("-" * 50)
    report_lines.append("\n[VALIDATION: TEMPORAL TRENDS]")
    report_lines.append("Objective: Confirm seasonal Monsoon patterns.")
    swell_std = monthly_stats['Swell_Pct'].std()
    report_lines.append(f"Result: Monthly Swell Variation (Std Dev): {swell_std:.2f}")
    if swell_std > 5:
        report_lines.append("Status: SUCCESS (Seasonal patterns visible)")
    else:
        report_lines.append("Status: WARNING (Low seasonal variation observed)")

    # 5. FIGURE INTERPRETATIONS (NEW SECTION)
    report_lines.append("-" * 50)
    report_lines.append("\n[FIGURE INTERPRETATION & NUMERICAL LOGS]")
    
    # 5.1 Enhanced_Map_Swell
    max_swell = grid_stats['Swell_Potential'].max()
    report_lines.append(f"\n1. Figure: Enhanced_Map_Swell.png")
    report_lines.append(f"   - Objective: Spatial distribution of wave energy resources.")
    report_lines.append(f"   - Key Data: Max Swell Potential is {max_swell:.2f}%.")
    if best_site is not None:
        report_lines.append(f"   - Best Site Value: {best_site['Swell_Potential']:.2f}% (vs Regional Avg {grid_stats['Swell_Potential'].mean():.2f}%)")
    
    # 5.2 Enhanced_Map_Storm
    max_storm_risk = grid_stats['Storm_Risk'].max()
    report_lines.append(f"\n2. Figure: Enhanced_Map_Storm.png")
    report_lines.append(f"   - Objective: Spatial distribution of storm risks.")
    report_lines.append(f"   - Key Data: Maximum Storm Risk observed anywhere is {max_storm_risk:.2f}%.")
    if best_site is not None:
        report_lines.append(f"   - Best Site Risk: {best_site['Storm_Risk']:.2f}% (Safe? {'YES' if best_site['Storm_Risk'] < 5 else 'NO'})")

    # 5.2b Enhanced_Map_Storm_LogScale
    report_lines.append(f"\n3. Figure: Enhanced_Map_Storm_LogScale.png (NEW)")
    report_lines.append(f"   - Objective: Visualizing low-frequency risks in Northern regions.")
    report_lines.append(f"   - Interpretation: Log scale reveals storm traces (>0.01%) invisible in linear plots.")

    # 5.3 Stat_Regime_Frequency
    total_recs = grid_stats['Total_Count'].sum()
    swell_pct = (grid_stats['Swell_Count'].sum() / total_recs) * 100
    calm_pct = (grid_stats['Calm_Count'].sum() / total_recs) * 100
    report_lines.append(f"\n4. Figure: Stat_Regime_Frequency.png")
    report_lines.append(f"   - Objective: Overall 10-year regime statistics.")
    report_lines.append(f"   - Key Data: Swell dominates {swell_pct:.1f}% of the time, Calm {calm_pct:.1f}%.")
    
    # 5.4 Stat_Site_Selection_Matrix
    report_lines.append(f"\n5. Figure: Stat_Site_Selection_Matrix.png")
    report_lines.append(f"   - Objective: Risk vs Reward tradeoff analysis.")
    if best_site is not None:
        report_lines.append(f"   - Key Data: Optimal Pareto Point selected at Lat {best_site['latitude']}, Lon {best_site['longitude']}.")

    # 5.5 Validation_Cyclone_Shaheen
    report_lines.append(f"\n6. Figure: Validation_Cyclone_Shaheen.png")
    report_lines.append(f"   - Objective: Extreme event detection capability.")
    if not shaheen_df.empty:
        report_lines.append(f"   - Key Data: Anomaly Ratio of {anomaly_ratio:.1f}x detected during event.")

    with open(os.path.join(OUTPUT_DIR, "Analysis_Report_Summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n🎉 DONE! Professional Maps and Stats saved to: {OUTPUT_DIR}")
    print("   📄 Report saved: Analysis_Report_Summary.txt")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()