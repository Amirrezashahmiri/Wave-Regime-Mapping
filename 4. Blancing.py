import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define specific Input and Output paths
INPUT_FILE_PATH = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 3\Global_Best_Matches_Merged.xlsx"
OUTPUT_DIR = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 4"
OUTPUT_FILE_NAME = 'Final_Balanced_Data_with_Features_Rolling.csv'
METADATA_FILE_NAME = 'Feature_Metadata_Report.txt'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'Visualizations')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Column mapping
STATION_COLUMN = 'Location_Tag'
DATETIME_COLUMN = 'Date and Time'
HM0_COLUMN = 'Hm0'
TP_COLUMN = 'Tp'
HMAX_COLUMN = 'Hmax'

# Threshold for including a group (30 days * 8 records/day = 240)
MINIMUM_GROUP_SIZE_THRESHOLD = 240

# --- FEATURE DICTIONARY (Glossary) ---
FEATURE_METADATA = {
    # -- General --
    'Location_Tag': {'Source': 'Meta', 'Desc': 'Unique identifier for the station segment'},
    'Base_Station': {'Source': 'Meta', 'Desc': 'Standardized station name'},
    'Season': {'Source': 'Calculated', 'Desc': 'Season index (1-4) derived from month'},
    'Season_Name': {'Source': 'Calculated', 'Desc': 'Season name (Winter, Spring...)'},
    'Date and Time': {'Source': 'Meta', 'Desc': 'Timestamp (UTC)'},
    'Depth_CMEMS': {'Source': 'CMEMS Static', 'Desc': 'Water depth at location', 'Unit': 'm'},

    # -- ADCP (Observed) --
    'Hm0': {'Source': 'ADCP', 'Desc': 'Significant Wave Height', 'Unit': 'm'},
    'Tp': {'Source': 'ADCP', 'Desc': 'Peak Wave Period', 'Unit': 's'},
    'Tm02': {'Source': 'ADCP', 'Desc': 'Mean Zero-Crossing Period', 'Unit': 's'},
    'Hmax': {'Source': 'ADCP', 'Desc': 'Maximum Wave Height', 'Unit': 'm'},
    'Hmean': {'Source': 'ADCP', 'Desc': 'Mean Wave Height', 'Unit': 'm'},
    
    # Updated Descriptions based on circular statistics
    'Mean Wave Direction': {'Source': 'ADCP', 'Desc': 'Mean Wave Direction (Calculated via Circular Vector Mean)', 'Unit': 'deg'},
    'Angle Fluctuation Std': {'Source': 'ADCP', 'Desc': 'Standard Deviation of Wave Direction (Calculated via Yamartino Method)', 'Unit': 'deg'},

    # -- CMEMS (Model) --
    'VHM0': {'Source': 'CMEMS', 'Desc': 'Sea surface wave significant height', 'Unit': 'm'},
    'VHM0_SW1': {'Source': 'CMEMS', 'Desc': 'Primary swell wave significant height', 'Unit': 'm'},
    'VHM0_SW2': {'Source': 'CMEMS', 'Desc': 'Secondary swell wave significant height', 'Unit': 'm'},
    'VHM0_WW': {'Source': 'CMEMS', 'Desc': 'Wind wave significant height', 'Unit': 'm'},
    'VMDR': {'Source': 'CMEMS', 'Desc': 'Mean wave direction', 'Unit': 'deg'},
    'VMDR_SW1': {'Source': 'CMEMS', 'Desc': 'Primary swell wave direction', 'Unit': 'deg'},
    'VMDR_SW2': {'Source': 'CMEMS', 'Desc': 'Secondary swell wave direction', 'Unit': 'deg'},
    'VMDR_WW': {'Source': 'CMEMS', 'Desc': 'Wind wave direction', 'Unit': 'deg'},
    'VPED': {'Source': 'CMEMS', 'Desc': 'Wave direction at spectral peak', 'Unit': 'deg'},
    'VSDX': {'Source': 'CMEMS', 'Desc': 'Stokes drift X velocity', 'Unit': 'm/s'},
    'VSDY': {'Source': 'CMEMS', 'Desc': 'Stokes drift Y velocity', 'Unit': 'm/s'},
    'VTM01_SW1': {'Source': 'CMEMS', 'Desc': 'Primary swell mean period', 'Unit': 's'},
    'VTM01_SW2': {'Source': 'CMEMS', 'Desc': 'Secondary swell mean period', 'Unit': 's'},
    'VTM01_WW': {'Source': 'CMEMS', 'Desc': 'Wind wave mean period', 'Unit': 's'},
    'VTM02': {'Source': 'CMEMS', 'Desc': 'Mean wave period (spectral moment 0,2)', 'Unit': 's'},
    'VTM10': {'Source': 'CMEMS', 'Desc': 'Mean wave period (spectral moment -1,0)', 'Unit': 's'},
    'VTPK': {'Source': 'CMEMS', 'Desc': 'Wave period at spectral peak', 'Unit': 's'},

    # -- Calculated Engineering Features --
    'Wave_Power_Kw': {'Source': 'Calculated', 'Desc': 'Wave Power Potential (0.5 * Hm0^2 * 0.9*Tp)', 'Unit': 'kW/m'},
    'Power_Fluctuation': {'Source': 'Calculated (Rolling)', 'Desc': 'Std Dev of Wave Power (24h window)', 'Unit': 'kW/m'},
    'Daily_Max_Hmax': {'Source': 'Calculated (Rolling)', 'Desc': 'Max Hmax observed in last 24h (Survivability)', 'Unit': 'm'},
    'Period_Fluctuation': {'Source': 'Calculated (Rolling)', 'Desc': 'Std Dev of Tp (24h window)', 'Unit': 's'},
    
    # -- Calculated Model Rolling Features --
    'Model_Height_Stability': {'Source': 'Calculated (Rolling)', 'Desc': 'Std Dev of VHM0 (24h window)', 'Unit': 'm'},
    'Model_Period_Stability': {'Source': 'Calculated (Rolling)', 'Desc': 'Std Dev of VTM02 (24h window)', 'Unit': 's'},
    'Swell_Stability': {'Source': 'Calculated (Rolling)', 'Desc': 'Std Dev of VHM0_SW1 (24h window)', 'Unit': 'm'},
    'Direction_Stability': {'Source': 'Calculated (Rolling)', 'Desc': 'Std Dev of VMDR (24h window)', 'Unit': 'deg'},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_base_station_name(station_name):
    if not isinstance(station_name, str):
        return 'Unknown'
    base_name = re.sub(r'(_segment)?_\d+$', '', station_name, flags=re.IGNORECASE).strip()
    return base_name

def get_season_label(date):
    month = date.month
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("🚀 Starting Data Preparation for Clustering...")
print(f"📂 Input:  {INPUT_FILE_PATH}")
print(f"📂 Output: {os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)}")
print("-" * 80)

try:
    # --- Step 1: Load Data ---
    print("1️⃣  Loading Data...")
    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE_PATH}")
        
    df = pd.read_excel(INPUT_FILE_PATH, engine='openpyxl')

    # --- INFO: PRINT FEATURES ---
    print("\n📋 [INFO] DETECTED INPUT FEATURES:")
    print("=" * 40)
    for col in df.columns:
        print(f"   🔹 {col}")
    print("=" * 40 + "\n")
    
    # Clean column names
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Ensure Datetime format
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce')
    
    # Drop rows with critical missing values
    initial_len = len(df)
    df.dropna(subset=[DATETIME_COLUMN, STATION_COLUMN, HM0_COLUMN, TP_COLUMN, HMAX_COLUMN], inplace=True)
    cleaned_len = len(df)
    
    print(f"   - Loaded {initial_len} rows.")
    print(f"   - After cleaning NaNs: {cleaned_len} rows.")

    # --- Step 2: Prepare for Calculation (Sorting is Vital) ---
    print("2️⃣  Preparing for Rolling Calculations...")
    
    # Create Base Station Name first (needed for grouping)
    df['Base_Station'] = df[STATION_COLUMN].apply(get_base_station_name)
    
    # CRITICAL: Sort by Station AND Time to ensure continuity
    df.sort_values(by=['Base_Station', DATETIME_COLUMN], inplace=True)
    print("   - Data sorted by Location and Time to ensure correct physics.")

    # --- Step 3: Calculate Engineering Features (ON FULL DATASET) ---
    print("3️⃣  Calculating ADVANCED Physics Features (Full Time-Series)...")
    
    # A. Wave Power Potential (kW/m) - Observation
    df['Wave_Power_Kw'] = 0.5 * (df[HM0_COLUMN]**2) * (0.9 * df[TP_COLUMN])
    
    # --- Rolling Calculation Logic (24-Hour Window) ---
    
    # 1. OBSERVED DATA FEATURES
    print("   - Calculating: Observed Power Stability (Rolling Std)")
    df['Power_Fluctuation'] = (
        df.groupby('Base_Station')['Wave_Power_Kw']
        .rolling(window=8, min_periods=1).std()
        .reset_index(level=0, drop=True)
    ).fillna(0)

    print("   - Calculating: Survivability Index (Rolling Max Hmax)")
    df['Daily_Max_Hmax'] = (
        df.groupby('Base_Station')[HMAX_COLUMN]
        .rolling(window=8, min_periods=1).max()
        .reset_index(level=0, drop=True)
    ).fillna(df[HMAX_COLUMN])

    print("   - Calculating: Observed Period Stability (Rolling Std Tp)")
    df['Period_Fluctuation'] = (
        df.groupby('Base_Station')[TP_COLUMN]
        .rolling(window=8, min_periods=1).std()
        .reset_index(level=0, drop=True)
    ).fillna(0)

    # 2. MODEL DATA FEATURES (NEW)
    if 'VHM0' in df.columns:
        print("   - Calculating: Model Wave Stability (Rolling Std VHM0)")
        df['Model_Height_Stability'] = (
            df.groupby('Base_Station')['VHM0']
            .rolling(window=8, min_periods=1).std()
            .reset_index(level=0, drop=True)
        ).fillna(0)

    if 'VTM02' in df.columns:
        print("   - Calculating: Model Period Stability (Rolling Std VTM02)")
        df['Model_Period_Stability'] = (
            df.groupby('Base_Station')['VTM02']
            .rolling(window=8, min_periods=1).std()
            .reset_index(level=0, drop=True)
        ).fillna(0)

    if 'VHM0_SW1' in df.columns:
        print("   - Calculating: Swell Stability (Rolling Std VHM0_SW1)")
        df['Swell_Stability'] = (
            df.groupby('Base_Station')['VHM0_SW1']
            .rolling(window=8, min_periods=1).std()
            .reset_index(level=0, drop=True)
        ).fillna(0)

    if 'VMDR' in df.columns:
        print("   - Calculating: Directional Stability (Rolling Std VMDR)")
        df['Direction_Stability'] = (
            df.groupby('Base_Station')['VMDR']
            .rolling(window=8, min_periods=1).std()
            .reset_index(level=0, drop=True)
        ).fillna(0)

    print("   - ✅ Features calculated on continuous time-series.")

    # --- Step 4: Stratification Setup ---
    print("4️⃣  Setting up Stratification (Seasons)...")
    df['Season_Name'] = df[DATETIME_COLUMN].apply(get_season_label)
    strat_key = 'Stratify_Key'
    df[strat_key] = df['Base_Station'] + '_' + df['Season_Name']

    # --- Step 5: Filter Small Groups ---
    print(f"5️⃣  Filtering groups < {MINIMUM_GROUP_SIZE_THRESHOLD} rows...")
    counts = df[strat_key].value_counts()
    valid_groups = counts[counts >= MINIMUM_GROUP_SIZE_THRESHOLD].index.tolist()
    
    if not valid_groups:
        raise ValueError("No groups met the minimum size threshold!")
        
    df_filtered = df[df[strat_key].isin(valid_groups)].copy()
    print(f"   - Retained {len(valid_groups)} valid groups.")

    # --- Step 6: Stratified Balancing (Sampling) ---
    print("6️⃣  Balancing Dataset (Stratified Sampling)...")
    
    min_size = df_filtered[strat_key].value_counts().min()
    print(f"   - Target Sample Size per Group: {min_size}")
    
    balanced_dfs = []
    for g_name, g_data in df_filtered.groupby(strat_key):
        balanced_dfs.append(g_data.sample(n=min_size, random_state=42))
        
    df_balanced = pd.concat(balanced_dfs)
    df_balanced.sort_values(by=['Base_Station', DATETIME_COLUMN], inplace=True)
    print(f"   - Balancing Complete. Final rows: {len(df_balanced)}")

    # --- Step 7: Save Output ---
    print("7️⃣  Saving Final Dataset...")
    final_output = df_balanced.drop(columns=[strat_key])
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    final_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"   ✅ File saved: {output_path}")

    # --- Step 8: Generate Metadata Report ---
    print("8️⃣  Generating Feature Metadata Report...")
    report_path = os.path.join(OUTPUT_DIR, METADATA_FILE_NAME)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=================================================================\n")
        f.write("       WAVE ENERGY DATASET - FEATURE METADATA REPORT             \n")
        f.write("=================================================================\n\n")
        f.write(f"Total Columns: {len(final_output.columns)}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'COLUMN NAME':<30} | {'SOURCE':<20} | {'UNIT':<10}\n")
        f.write("-" * 65 + "\n")
        
        for col in final_output.columns:
            meta = FEATURE_METADATA.get(col, {'Source': 'Unknown', 'Desc': 'No description', 'Unit': '-'})
            f.write(f"{col:<30} | {meta.get('Source', 'Unknown'):<20} | {meta.get('Unit', '-'):<10}\n")
            f.write(f"   ↳ {meta.get('Desc', 'No description')}\n")
            f.write("-" * 65 + "\n")
            
    print(f"   ✅ Metadata saved: {report_path}")

    # --- Step 9: Generate Visualizations (NEW) ---
    print("9️⃣  Generating Visualizations...")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # A. Temporal Distribution (Timeline)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=final_output, 
        x=DATETIME_COLUMN, 
        y='Base_Station', 
        hue='Season_Name', 
        palette='viridis', 
        s=10, 
        edgecolor=None
    )
    plt.title('Temporal Distribution of Final Balanced Data', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Location', fontsize=12)
    plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Temporal_Distribution.png'), dpi=300)
    print("   📸 Plot saved: Temporal_Distribution.png")
    
    # B. Seasonal Count Bar Chart
    plt.figure(figsize=(10, 6))
    count_data = final_output.groupby(['Base_Station', 'Season_Name']).size().reset_index(name='Count')
    sns.barplot(data=count_data, x='Base_Station', y='Count', hue='Season_Name', palette='coolwarm')
    plt.title('Balanced Data Count per Location & Season', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Records', fontsize=12)
    plt.xlabel('Location', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Seasonal_Data_Balance.png'), dpi=300)
    print("   📸 Plot saved: Seasonal_Data_Balance.png")

    print("\n🎉 Process Completed Successfully.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")