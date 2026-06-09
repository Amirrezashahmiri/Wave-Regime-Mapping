import pandas as pd
import os
import numpy as np
from scipy.stats import hmean
from sklearn.experimental import enable_iterative_imputer  # Enable experimental IterativeImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Optional
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
input_file = r'C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\ADCP\Wudam South.xlsx'
# Updated output directory to H3
output_dir = r'C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 2\Wudam South'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

base_name = os.path.splitext(os.path.basename(input_file))[0]

# -----------------------------------------------------------------------------
# 2. Circular Statistics Functions
# -----------------------------------------------------------------------------
def circular_mean(angles):
    """Calculates the vector mean of angles (returns 0-360)."""
    if angles.isna().all():
        return np.nan
    rads = np.deg2rad(angles.dropna())
    sin_sum = np.sum(np.sin(rads))
    cos_sum = np.sum(np.cos(rads))
    result = np.arctan2(sin_sum, cos_sum)
    return np.degrees(result) % 360

def circular_std(angles):
    """Calculates the circular standard deviation (Yamartino method)."""
    if angles.isna().all():
        return np.nan
    rad_angles = np.deg2rad(angles.dropna())
    if len(rad_angles) == 0:
        return np.nan
    sin_sum = np.sum(np.sin(rad_angles))
    cos_sum = np.sum(np.cos(rad_angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(rad_angles)
    if R == 0:
        return np.inf 
    if R > 1: R = 1 # Avoid precision errors
    return np.rad2deg(np.sqrt(-2 * np.log(R)))

# -----------------------------------------------------------------------------
# 3. Load & Preprocess Data (UTC & Resampling)
# -----------------------------------------------------------------------------
# Load ONLY WAVE sheet
df_wave = pd.read_excel(input_file, sheet_name='WAVE')

# Timestamp conversion
df_wave['Date and Time'] = pd.to_datetime(df_wave['Date and Time'])

# Convert Local Time (Oman UTC+4) to UTC by subtracting 4 hours
df_wave['Date and Time'] = df_wave['Date and Time'] - pd.Timedelta(hours=4)
logger.info("Data converted to UTC (subtracted 4 hours from local time).")

print(f"\nWAVE time range (UTC): {df_wave['Date and Time'].min()} to {df_wave['Date and Time'].max()}")

# Aggregation rules
wave_agg = {
    'Station ID': 'first',
    'Measurement No': 'first',
    'Hm0': 'mean',
    'Tp': 'mean',
    'MeanDir': circular_std,  # Calculates Deviation (Std)
    'Hmax': 'mean',
    'Hmean': 'mean',
    'Tm02': 'mean'
}

# Resample to 3-Hour intervals
df_resampled = df_wave.set_index('Date and Time').resample('3h').agg(wave_agg)

# Calculate Circular Mean separately
df_resampled['Mean Wave Direction'] = df_wave.set_index('Date and Time')['MeanDir'].resample('3h').apply(circular_mean)

# Rename Std column
df_resampled.rename(columns={'MeanDir': 'Angle Fluctuation Std'}, inplace=True)

# Drop empty bins (where no wave data exists)
df_resampled = df_resampled.dropna(subset=['Hm0'])

# Columns to process
process_cols = ['Hm0', 'Tp', 'Mean Wave Direction', 'Angle Fluctuation Std', 'Hmax', 'Hmean', 'Tm02']

# -----------------------------------------------------------------------------
# 4. Outlier Detection & Imputation Logic
# -----------------------------------------------------------------------------
def detect_and_handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'ensemble', 
                               threshold: float = 3, window_sizes: List[int] = [6, 12, 24]) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columns:
        if df[col].notna().sum() < 10:
            continue
            
        # 1. Hampel (Rolling Median)
        best_window = window_sizes[0]
        min_outliers = float('inf')
        for w in window_sizes:
             rmed = df[col].rolling(w, center=True, min_periods=1).median()
             rmad = (df[col] - rmed).abs().rolling(w, center=True, min_periods=1).median()
             cnt = ((df[col] - rmed).abs() > threshold * rmad).sum()
             if 0 < cnt < min_outliers:
                 min_outliers = cnt
                 best_window = w
        
        roll_med = df[col].rolling(best_window, center=True, min_periods=1).median()
        roll_mad = (df[col] - roll_med).abs().rolling(best_window, center=True, min_periods=1).median()
        mask_hampel = (df[col] - roll_med).abs() > threshold * roll_mad
        
        # 2. Z-Score
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        mask_zscore = z_scores.abs() > threshold
        
        # 3. IQR
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask_iqr = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
        
        # Ensemble Voting (2 out of 3)
        if method == 'ensemble':
            vote_count = mask_hampel.astype(int) + mask_zscore.astype(int) + mask_iqr.astype(int)
            final_mask = vote_count >= 2
            replacement = roll_med
        else:
            final_mask = mask_hampel
            replacement = roll_med

        if final_mask.sum() > 0:
            df_clean.loc[final_mask, col] = replacement[final_mask]
            logger.info(f"Outliers removed in {col}: {final_mask.sum()}")
            
    return df_clean

def dineof_imputation(df: pd.DataFrame, numeric_cols: List[str], max_iter=50) -> pd.DataFrame:
    # Basic DINEOF implementation wrapper
    valid_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().sum() > 5]
    if not valid_cols: return df
    
    data = df[valid_cols].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.fillna(data.mean()))
    missing_mask = np.isnan(df[valid_cols].values)
    
    # SVD Iteration
    prev_rmse = float('inf')
    for i in range(max_iter):
        U, s, Vt = np.linalg.svd(data_scaled, full_matrices=False)
        # Keep top 3 components or dynamic
        k = min(3, data_scaled.shape[1])
        s[k:] = 0
        recon = U @ np.diag(s) @ Vt
        
        rmse = np.sqrt(np.mean((data_scaled[~missing_mask] - recon[~missing_mask])**2))
        if abs(prev_rmse - rmse) < 1e-5: break
        prev_rmse = rmse
        data_scaled[missing_mask] = recon[missing_mask]
        
    imputed = scaler.inverse_transform(data_scaled)
    df_out = df.copy()
    for idx, col in enumerate(valid_cols):
        miss = df[col].isna()
        df_out.loc[miss, col] = imputed[miss.values, idx]
    return df_out

def sophisticated_imputation(df: pd.DataFrame, process_cols: List[str], max_gap: int = 6) -> pd.DataFrame:
    initial_nans = df[process_cols].isna().sum()
    logger.info(f"Initial NaN counts: {initial_nans[initial_nans > 0]}")
    
    # 1. Separate Directional vs Linear columns
    directional_cols = ['Mean Wave Direction']
    linear_cols = [c for c in process_cols if c not in directional_cols]
    
    # 2. Outlier Detection (ONLY on Linear columns)
    df = detect_and_handle_outliers(df, linear_cols, method='ensemble')
    
    # 3. Impute Direction (Sin/Cos ONLY - No DINEOF for degrees)
    for col in directional_cols:
        if col in df.columns and df[col].isna().any():
            logger.info(f"Imputing directional variable: {col} using Sin/Cos")
            rad = np.radians(df[col])
            df[f"{col}_sin"] = np.sin(rad)
            df[f"{col}_cos"] = np.cos(rad)
            # Interpolate components
            for comp in [f"{col}_sin", f"{col}_cos"]:
                df[comp] = df[comp].interpolate(method='time', limit=max_gap, limit_area='inside')
                # If still NaN (gap > max_gap), we leave it as NaN (Safer than linear guess)
            
            # Reconstruct angle only where interpolation succeeded
            mask_valid = df[f"{col}_sin"].notna() & df[f"{col}_cos"].notna()
            df.loc[mask_valid & df[col].isna(), col] = np.degrees(np.arctan2(
                df.loc[mask_valid, f"{col}_sin"],
                df.loc[mask_valid, f"{col}_cos"]
            )) % 360
            
            df = df.drop(columns=[f"{col}_sin", f"{col}_cos"])
    
    # 4. Impute Linear Variables
    imputation_strategies = {
        'Hm0': 'dineof', 'Tp': 'dineof',
        'Hmax': 'dineof', 'Hmean': 'dineof', 'Tm02': 'dineof'
        # REMOVED: 'Mean Wave Direction': 'dineof' (To prevent linear math on degrees)
    }
    
    # Pre-fill small linear gaps
    for col in linear_cols:
        if df[col].isna().any():
             df[col] = df[col].interpolate(method='time', limit=max_gap, limit_area='inside')

    # Advanced Linear Imputation
    for col in linear_cols:
        if not df[col].isna().any(): continue
        
        logger.info(f"Advanced imputation for {col}")
        strategy = imputation_strategies.get(col, 'hybrid')
        
        if strategy == 'dineof':
            try:
                # Use correlated vars if available
                vars_to_use = [col]
                if 'Hm0' in df.columns and col != 'Hm0': vars_to_use.append('Hm0')
                if 'Tp' in df.columns and col != 'Tp': vars_to_use.append('Tp')
                
                df = dineof_imputation(df, vars_to_use)
            except Exception as e:
                logger.warning(f"DINEOF failed for {col}: {e}. Falling back to KNN.")
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]])
                except:
                    pass
                    
        # Final fallback for linear columns
        if df[col].isna().any():
            df[col] = df[col].ffill(limit=max_gap).bfill(limit=max_gap)
                
    return df

# -----------------------------------------------------------------------------
# 5. Segmentation & Execution
# -----------------------------------------------------------------------------
# Detect gaps > 24 hours
segment_id = (df_resampled.index.to_series().diff() > pd.Timedelta(hours=24)).cumsum()
df_resampled['segment_id'] = segment_id

for i, (_, segment) in enumerate(df_resampled.groupby('segment_id')):
    segment = segment.drop(columns='segment_id')
    if segment.empty: continue
    
    logger.info(f"Processing Segment {i+1} ({len(segment)} rows)...")
    
    # Backup for plotting
    seg_raw = segment.copy()
    
    # Process
    seg_clean = sophisticated_imputation(segment, process_cols, max_gap=6) # 6*3h = 18h gap fill
    
    # Save
    out_name = os.path.join(output_dir, f'{base_name}_segment_{i+1}_resampled.xlsx')
    seg_clean.reset_index().to_excel(out_name, index=False)
    print(f"Saved: {out_name}")
    
    # Plotting
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for col in process_cols:
        if col not in seg_clean.columns: continue
        plt.figure(figsize=(10, 5))
        plt.plot(seg_raw.index, seg_raw[col], label='Raw', color='blue', alpha=0.5)
        plt.plot(seg_clean.index, seg_clean[col], label='Imputed', color='red', alpha=0.5, linestyle='--')
        plt.title(f"{col} - Segment {i+1}")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{col}_seg{i+1}.png'))
        plt.close()

print("\nProcessing Complete.")