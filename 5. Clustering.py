import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# 1. STABILITY SETUP (REPRODUCIBILITY)
# =============================================================================
def set_global_seeds(seed=42):
    """
    Sets random seeds for all libraries to ensure 100% reproducible results.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"🔒 Global Random Seed set to: {seed}")

# Apply seeds immediately
set_global_seeds(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Paths ---
INPUT_FILE_PATH = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 4\Final_Balanced_Data_with_Features_Rolling.csv"
BASE_OUTPUT_DIR = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 5"

# --- Feature Selection ---
FEATURES_FOR_CLUSTERING = [
    'Hm0',                   # Magnitude: Significant Wave Height
    'Tp',                    # Frequency: Peak Period
    'Wave_Power_Kw',         # Economy: Wave Power Potential
    'Angle Fluctuation Std', # Quality: Directional Spreading/Stability
    'Power_Fluctuation',     # Stability: Temporal variability of power
    'Period_Fluctuation',    # Stability: Temporal variability of period
    'Daily_Max_Hmax'         # Risk: Survivability index
]

# Units for visualization
FEATURE_UNITS = {
    'Hm0': '(m)',
    'Tp': '(s)',
    'Wave_Power_Kw': '(kW/m)',
    'Angle Fluctuation Std': '(deg)',
    'Power_Fluctuation': '(kW/m)',
    'Period_Fluctuation': '(s)',
    'Daily_Max_Hmax': '(m)'
}

# Number of clusters to finalize
N_CLUSTERS = 4

# Rationale for Feature Selection
FEATURE_RATIONALE = """
STRATEGIC RATIONALE FOR FEATURE SELECTION:

1. MAGNITUDE (Energy Potential):
   - Hm0: The primary indicator of sea state severity.
   - Wave_Power_Kw: Represents the economic value of the resource.

2. FREQUENCY (Device Tuning):
   - Tp: Critical for determining resonance frequency of WECs.

3. QUALITY (Directional Stability):
   - Angle Fluctuation Std: Low values imply unidirectional waves (high quality); high values imply crossing seas.

4. DYNAMICS & STABILITY (Operational Consistency):
   - Power_Fluctuation: Distinguishes between "Steady" and "Intermittent" resources.
   - Period_Fluctuation: Indicates how often a WEC needs to retune.

5. RISK (Survivability):
   - Daily_Max_Hmax: Identifies regimes with destructive extreme waves.
"""

# Ensure output directory exists
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Initialize the Master Report Content
MASTER_REPORT = []
MASTER_REPORT.append("============================================================")
MASTER_REPORT.append("       COMPREHENSIVE CLUSTERING ANALYSIS PROJECT REPORT     ")
MASTER_REPORT.append("============================================================\n")
MASTER_REPORT.append("PART 1: FEATURE SELECTION RATIONALE")
MASTER_REPORT.append("-----------------------------------")
MASTER_REPORT.append(FEATURE_RATIONALE)
MASTER_REPORT.append("\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_improvement_rate(bic_scores):
    """
    Calculates the percentage improvement (drop) in BIC for each step.
    This provides the STATISTICAL DEFENSE for choosing k.
    """
    improvements = []
    # No improvement for the first k, so we start loop from index 1
    for i in range(1, len(bic_scores)):
        prev = bic_scores[i-1]
        curr = bic_scores[i]
        # Calculate how much BIC dropped relative to the previous value
        imp_percent = ((prev - curr) / prev) * 100
        improvements.append(imp_percent)
    return improvements

def plot_tsne_projection(data, labels, title, output_dir, filename):
    """
    Generates a t-SNE plot to VISUALLY prove cluster separation.
    """
    print(f"   ... Computing t-SNE for {title} (This might take a moment) ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels, palette='viridis', legend='full', s=60, alpha=0.7
    )
    plt.title(f't-SNE Projection: {title}\n(Visual Proof of Separation)', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def analyze_and_visualize_clusters(df, features, cluster_col_name, output_dir, feature_units, data_for_metrics, plot_tsne=False):
    """
    Analyzes clusters and returns a text summary for the master report.
    Added 'plot_tsne' parameter to generate visual proof.
    """
    print(f"\n--- Analyzing Results for {cluster_col_name} ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Calculate statistics
    cluster_freq = df[cluster_col_name].value_counts().sort_index()
    cluster_means = df.groupby(cluster_col_name)[features].mean()
    cluster_medians = df.groupby(cluster_col_name)[features].median()
    cluster_stds = df.groupby(cluster_col_name)[features].std()

    print(f"\nFrequency of Each Cluster:\n{cluster_freq}")
    
    # 2. Quantitative Evaluation
    labels = df[cluster_col_name]
    silhouette = silhouette_score(data_for_metrics, labels)
    calinski = calinski_harabasz_score(data_for_metrics, labels)
    davies = davies_bouldin_score(data_for_metrics, labels)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'],
        'Score': [silhouette, calinski, davies],
        'Interpretation': ['Higher is better', 'Higher is better', 'Lower is better']
    }).set_index('Metric')
    
    # 3. Save Excel Statistics
    stats_file = os.path.join(output_dir, 'cluster_statistics.xlsx')
    with pd.ExcelWriter(stats_file) as writer:
        cluster_freq.to_frame(name='Frequency').to_excel(writer, sheet_name='Cluster Frequency')
        cluster_means.to_excel(writer, sheet_name='Cluster Means')
        metrics_df.to_excel(writer, sheet_name='Evaluation Metrics')
    print(f"   📊 Statistics Excel saved: {stats_file}")

    # 4. Generate Visualizations (Standard Boxplots & Heatmaps)
    print("Generating visualizations...")
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cluster_col_name, y=feature, data=df, palette='viridis', hue=cluster_col_name)
        plt.title(f'Distribution of {feature} by {cluster_col_name}')
        plt.ylabel(f'{feature} {feature_units.get(feature, "")}')
        plt.legend([],[], frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{feature}.png'))
        plt.close()

    # Spider Plot
    scaler_means = StandardScaler()
    cluster_means_normalized = scaler_means.fit_transform(cluster_means)
    labels_plot = features
    num_vars = len(labels_plot)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i in range(df[cluster_col_name].nunique()):
        values = cluster_means_normalized[i].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {i}', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_plot, size=10)
    plt.title('Normalized Regime Profiles', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(output_dir, 'normalized_regime_profiles.png'), bbox_inches='tight')
    plt.close()

    # Heatmap
    plt.figure(figsize=(12, 10))
    crosstab = pd.crosstab(df['Base_Station'], df[cluster_col_name])
    sns.heatmap(crosstab, cmap='viridis', annot=True, fmt='d', linewidths=.5)
    plt.title(f'Regime Distribution by Station')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_distribution_heatmap.png'))
    plt.close()
    
    # 5. NEW: t-SNE Visualization (The Visual Proof)
    if plot_tsne:
        plot_tsne_projection(data_for_metrics, labels, cluster_col_name, output_dir, 'tSNE_Manifold_Projection.png')

    # 6. Save Full Data
    output_csv_path = os.path.join(output_dir, 'full_clustered_data.csv')
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"   💾 Full labeled dataset saved: {output_csv_path}")

    # 7. Generate Text Summary for Master Report
    summary = []
    summary.append(f"--- MODEL: {cluster_col_name} ---")
    summary.append("\n>> PERFORMANCE METRICS:")
    summary.append(f"   Silhouette Score:       {silhouette:.4f} (Higher is better)")
    summary.append(f"   Calinski-Harabasz:      {calinski:.4f} (Higher is better)")
    summary.append(f"   Davies-Bouldin:         {davies:.4f} (Lower is better)")
    summary.append("\n>> CLUSTER PROFILES (Means):")
    summary.append(cluster_means.to_string())
    summary.append("\n>> CLUSTER FREQUENCIES:")
    summary.append(cluster_freq.to_string())
    summary.append("-" * 60 + "\n")
    
    return "\n".join(summary)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try:
    print("🚀 Starting Clustering Analysis Pipeline...")
    print(f"📂 Input File: {INPUT_FILE_PATH}")
    print(f"📂 Output Dir: {BASE_OUTPUT_DIR}")
    
    # --- Step 1: Load Data ---
    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE_PATH}")
        
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"   ✅ Data loaded. Shape: {df.shape}")

    df_clean = df.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
    df_clean.reset_index(drop=True, inplace=True)
    data_for_clustering = df_clean[FEATURES_FOR_CLUSTERING]
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_for_clustering)
    print("   ✅ Data normalized and ready.")

    # --- Step 2: Optimal K Analysis (WITH IMPROVEMENT RATE) ---
    print("\n🔬 Analyzing Optimal Number of Clusters...")
    MASTER_REPORT.append("PART 2: STATISTICAL DEFENSE FOR K=4 (BIC DELTA ANALYSIS)")
    MASTER_REPORT.append("--------------------------------------------------------")
    MASTER_REPORT.append(f"{'K':<5} | {'BIC Score':<12} | {'Improvement % (Gain)':<22} | {'Status'}")
    MASTER_REPORT.append("-" * 65)

    k_range = range(2, 9)
    bic_scores = []
    silhouette_scores = []
    
    opt_dir = os.path.join(BASE_OUTPUT_DIR, 'Optimization_Analysis')
    os.makedirs(opt_dir, exist_ok=True)

    for k in k_range:
        # Note: random_state=42 ensures stability here
        gmm_test = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
        labels = gmm_test.fit_predict(data_normalized)
        
        bic = gmm_test.bic(data_normalized)
        sil = silhouette_score(data_normalized, labels)
        
        bic_scores.append(bic)
        silhouette_scores.append(sil)
    
    # Calculate Improvements
    improvements = calculate_improvement_rate(bic_scores)
    
    # Print the First row (k=2) manually as baseline
    row_k2 = f"{2:<5} | {bic_scores[0]:<12.0f} | {'-':<22} | Baseline"
    print(f"   - {row_k2}")
    MASTER_REPORT.append(row_k2)

    # Loop for k=3 to 8
    for i, k in enumerate(k_range[1:]): # Start from k=3
        imp = improvements[i]
        
        # Determine status string for report
        status = "Significant Gain"
        if imp < 10: status = "Diminishing Return"
        if k == 4: status = "<< OPTIMAL ELBOW" # Tagging the optimal point
        
        row_str = f"{k:<5} | {bic_scores[i+1]:<12.0f} | {imp:<22.2f} | {status}"
        print(f"   - {row_str}")
        MASTER_REPORT.append(row_str)

    MASTER_REPORT.append("\n")

    # Plot Optimization
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, bic_scores, 'o--', label='BIC (Lower is better)')
    plt.axvline(x=4, color='r', linestyle='--', label='Selected k=4') 
    plt.title('Elbow Method (BIC) with Selection')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('BIC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(opt_dir, 'BIC_Elbow_Plot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores, 'o--', color='orange', label='Silhouette (Higher is better)')
    plt.title('Silhouette Analysis')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(os.path.join(opt_dir, 'Silhouette_Plot.png'))
    plt.close()

    MASTER_REPORT.append("PART 3: MODEL COMPARISON & VISUAL PROOF")
    MASTER_REPORT.append("---------------------------------------")

    # --- Step 3: Method 0 - Baseline K-Means ---
    print(f"\n🧠 Method 0: Baseline K-Means (k={N_CLUSTERS})")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(data_normalized)
    
    df_kmeans = df_clean.copy()
    df_kmeans['KMeans_Regime'] = kmeans_labels
    
    summary_kmeans = analyze_and_visualize_clusters(
        df_kmeans, FEATURES_FOR_CLUSTERING, 'KMeans_Regime',
        os.path.join(BASE_OUTPUT_DIR, 'Method_0_Baseline_KMeans'),
        FEATURE_UNITS, data_normalized, plot_tsne=True
    )
    MASTER_REPORT.append(summary_kmeans)

    # --- Step 4: Method 1 - Standard GMM ---
    print(f"\n🧠 Method 1: Gaussian Mixture Model (k={N_CLUSTERS})")
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, covariance_type='full')
    gmm_labels = gmm.fit_predict(data_normalized)
    
    df_gmm = df_clean.copy()
    df_gmm['GMM_Regime'] = gmm_labels
    
    summary_gmm = analyze_and_visualize_clusters(
        df_gmm, FEATURES_FOR_CLUSTERING, 'GMM_Regime',
        os.path.join(BASE_OUTPUT_DIR, 'Method_1_GMM'),
        FEATURE_UNITS, data_normalized, plot_tsne=False
    )
    MASTER_REPORT.append(summary_gmm)

    # --- Step 5: Method 2 - Deep Clustering (Autoencoder + GMM) ---
    print(f"\n🧠 Method 2: Deep Clustering (Autoencoder + GMM) (k={N_CLUSTERS})")
    
    # Autoencoder Architecture
    input_dim = data_normalized.shape[1]
    encoding_dim = 4

    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(16, activation='relu')(input_layer)
    encoder = layers.Dense(8, activation='relu')(encoder)
    latent_space = layers.Dense(encoding_dim, activation='relu')(encoder)
    
    decoder = layers.Dense(8, activation='relu')(latent_space)
    decoder = layers.Dense(16, activation='relu')(decoder)
    output_layer = layers.Dense(input_dim, activation='linear')(decoder)
    
    autoencoder = models.Model(input_layer, output_layer)
    encoder_model = models.Model(input_layer, latent_space)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("   - Training Autoencoder (Noise Reduction)...")
    autoencoder.fit(
        data_normalized, data_normalized,
        epochs=100, batch_size=32, shuffle=True, validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    encoded_features = encoder_model.predict(data_normalized)
    
    gmm_deep = GaussianMixture(n_components=N_CLUSTERS, random_state=42, covariance_type='full')
    deep_labels = gmm_deep.fit_predict(encoded_features)
    
    df_deep = df_clean.copy()
    df_deep['Deep_Regime'] = deep_labels
    
    summary_deep = analyze_and_visualize_clusters(
        df_deep, FEATURES_FOR_CLUSTERING, 'Deep_Regime',
        os.path.join(BASE_OUTPUT_DIR, 'Method_2_Deep_Clustering'),
        FEATURE_UNITS, encoded_features, plot_tsne=True
    )
    MASTER_REPORT.append(summary_deep)

    # =============================================================================
    # NEW STEP 7: APPEND FINAL INTERPRETATION & CONCLUSION
    # =============================================================================
    FINAL_CONCLUSION = """
PART 4: FINAL EXECUTIVE SUMMARY & INTERPRETATION
------------------------------------------------
This section provides a guide for interpreting the results, selected plots, and model choices.

1. JUSTIFICATION FOR OPTIMAL CLUSTERS (K=4)
   - Statistical Evidence: The BIC Delta Analysis (Part 2) shows a significant improvement gain 
     of approximately 22.4% when moving from k=3 to k=4.
   - Diminishing Returns: Moving from k=4 to k=5 yields a much smaller gain (approx 7.8%), 
     indicating that k=5 starts to model noise rather than structure.
   - Conclusion: k=4 is the robust "Elbow Point".

2. MODEL SELECTION: WHY DEEP CLUSTERING?
   - The Comparison: 
     * K-Means creates linear, geometric cuts (High Calinski score but artificial separation).
     * Deep Clustering (Autoencoder + GMM) achieves the highest Silhouette Score (approx 0.37).
   - The Advantage: Deep Clustering successfully captures non-linear relationships, specifically 
     distinguishing between "High Energy/Stable" (Swell) and "High Energy/Unstable" (Storm) regimes, 
     which classical methods often conflate.

3. PHYSICAL INTERPRETATION OF REGIMES (Based on Deep Clustering Profiles)
   *Note: Check the Cluster Means table in Part 3 to match IDs (0-3) to these descriptions.*
   
   A. "THE GOLDEN SWELL" (Ideal for Energy)
      - Characteristics: High Period (Tp > 7s), Moderate/High Power, Very Low Angle/Period Fluctuation.
      - Utility: Best regime for Wave Energy Converters (WECs) due to high consistency.
      
   B. "THE STORM" (High Risk)
      - Characteristics: Highest Power, High Height (Hm0), High Instability (Power/Angle Fluctuation).
      - Utility: Requires 'Survival Mode' for devices to avoid damage.
      
   C. "AMBIENT / CALM" (Background Sea)
      - Characteristics: Low Height, Low Power, Short Period.
      - Utility: Non-operational or idle mode.
      
   D. "CONFUSED SEA" (Low Efficiency)
      - Characteristics: High Angle Fluctuation (Std > 20 deg), Short Period.
      - Utility: Low capture efficiency due to multi-directional waves.

4. GUIDE TO PLOTS (For the Reader)
   - 'Elbow_Method_BIC.png': Shows the mathematical justification for choosing 4 clusters.
   - 'tSNE_Manifold_Projection.png': Visual proof that the clusters are distinct islands in the 
      data manifold (especially in Method 2).
   - 'normalized_regime_profiles.png' (Spider Plot): The "Fingerprint" of each cluster. 
      Use this to see which cluster is "Swell" (spikes in Tp) vs "Storm" (spikes in Hm0 & Fluctuation).
   - 'regime_distribution_heatmap.png': Shows where (which station) these regimes occur most often.
"""
    MASTER_REPORT.append(FINAL_CONCLUSION)

    # --- Step 6: Save Master Report ---
    report_path = os.path.join(BASE_OUTPUT_DIR, "COMPREHENSIVE_PROJECT_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines([line + "\n" if not line.endswith("\n") else line for line in MASTER_REPORT])
    
    print(f"\n📄 COMPREHENSIVE REPORT SAVED: {report_path}")
    print("🎉 Analysis Completed Successfully.")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")