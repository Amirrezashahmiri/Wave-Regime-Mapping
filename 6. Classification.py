import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
import random
import matplotlib
import joblib  # <--- Added for saving the model
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- New Imports for ROC Curve, Stacking & Cross Validation ---
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Import SMOTE ---
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("ERROR: 'imbalanced-learn' library is missing.")
    print("Please install it running: pip install imbalanced-learn")
    raise

# ==============================================================================
# --- 1. STABILITY SETUP (REPRODUCIBILITY) ---
# ==============================================================================
def set_global_seeds(seed=42):
    """
    Sets random seeds for all libraries to ensure 100% reproducible results.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Global Random Seed set to: {seed}")

# Apply seeds immediately
set_global_seeds(42)

# ==============================================================================
# --- CONFIGURATION (PATHS UPDATED) ---
# ==============================================================================
# Exact Input File Path (From Previous Step)
INPUT_FILE_FULL_PATH = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 5\Method_2_Deep_Clustering\full_clustered_data.csv"

# Exact Output Directory
BASE_OUTPUT_DIR = r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try\Test\Output Code 6"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'Classification_Model_Results')

# Target Column Name (Must match the column created in previous clustering script)
TARGET_COLUMN = 'Deep_Regime' 

# ==============================================================================
# --- DIAGNOSTICS ---
# ==============================================================================
print("\n" + "="*60)
print("SYSTEM DIAGNOSTICS")
print("="*60)

# Check LightGBM
LGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
    print("   [OK] LightGBM imported.")
except ImportError:
    print("   [WARN] LightGBM not found. Skipping.")

# Check TabNet
TABNET_AVAILABLE = False
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    # Ensure torch reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    TABNET_AVAILABLE = True
    print("   [OK] TabNet imported.")
except ImportError:
    print("   [WARN] TabNet/Torch not found. Skipping.")

print("="*60 + "\n")

# Feature Selection Modules
from sklearn.feature_selection import RFECV, SelectFromModel

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- Storage for Final Comparison ---
results_data = []

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def evaluate_and_log(stage, model_name, model, X_test, y_test, note=""):
    """
    Central function to evaluate, print report, save CM plot, and log data.
    """
    # 1. Prediction
    y_pred = model.predict(X_test)
    
    # 2. Metrics
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Print Report to Console
    print(f"\n--- Evaluation: [{stage}] {model_name} ---")
    print(f"    Accuracy: {acc:.4f}")
    print("    Per-Class Metrics:")
    print(report_str)
    
    # 4. Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(np.unique(y_test)), 
                yticklabels=sorted(np.unique(y_test)))
    plt.title(f'CM: {stage} - {model_name}\nAcc: {acc:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Create valid filename (remove spaces/special chars)
    filename_stage = stage.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
    plot_filename = f"CM_{filename_stage}_{model_name}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
    plt.close()
    
    # 5. Log Data for Final CSV
    results_data.append({
        'Stage': stage,
        'Model': model_name,
        'Accuracy': acc,
        'Precision_Weighted': report_dict['weighted avg']['precision'],
        'Recall_Weighted': report_dict['weighted avg']['recall'],
        'F1_Weighted': report_dict['weighted avg']['f1-score'],
        'Note': note
    })
    
    return acc

def plot_uncertainty_analysis(model, X_test, y_test, output_dir):
    """
    Analyzes and plots model confidence vs accuracy (Reliability Diagram).
    FIXED: Ensuring 1D arrays for DataFrame construction & Seaborn warning fix.
    """
    print("\n🔬 Running Uncertainty & Confidence Analysis...")
    
    # Get probabilities (Confidence)
    try:
        probas = model.predict_proba(X_test)
    except AttributeError:
        print("   [WARN] Model does not support predict_proba. Skipping Uncertainty Analysis.")
        return

    # Calculate max probability (Confidence)
    confidence = np.max(probas, axis=1) 
    
    # Get predictions
    predictions = model.predict(X_test)
    
    # Prepare y_true (Target) - Ensuring 1D numpy array
    y_true = np.array(y_test)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    y_true = y_true.flatten() # Force 1D

    # Ensure other arrays are 1D (Flattening solves the 'Per-column arrays...' error)
    predictions = np.array(predictions).flatten()
    confidence = np.array(confidence).flatten()

    # Calculate correctness
    correct = (predictions == y_true)
    
    # Create DataFrame for analysis
    try:
        df_unc = pd.DataFrame({'Confidence': confidence, 'Correct': correct})
    except Exception as e:
        print(f"   [ERROR] Failed to create Uncertainty DataFrame: {e}")
        return
    
    # Binning confidence scores
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    df_unc['Conf_Bin'] = pd.cut(df_unc['Confidence'], bins=bins, labels=labels)
    
    # Calculate accuracy per bin
    bin_acc = df_unc.groupby('Conf_Bin', observed=False)['Correct'].mean()
    bin_counts = df_unc.groupby('Conf_Bin', observed=False)['Correct'].count()
    
    # Plot
    plt.figure(figsize=(10, 6))
    # FIXED: Updated seaborn syntax to avoid FutureWarnings
    ax = sns.barplot(x=bin_acc.index, y=bin_acc.values, hue=bin_acc.index, palette='RdYlGn', legend=False)
    
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Accuracy Threshold')
    plt.title('Model Reliability Diagram: Accuracy vs. Confidence', fontsize=14)
    plt.xlabel('Prediction Confidence (Probability)', fontsize=12)
    plt.ylabel('Actual Accuracy in Bin', fontsize=12)
    plt.ylim(0, 1.1)
    
    # Add counts on bars
    for i, p in enumerate(ax.patches):
        if i < len(bin_counts):
            count = bin_counts.iloc[i]
            if count > 0:
                ax.annotate(f"N={count}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', xytext = (0, 10), 
                            textcoords = 'offset points', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'Uncertainty_Reliability_Plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Reliability Plot saved: {save_path}")

# ==============================================================================
# --- MAIN SCRIPT EXECUTION ---
# ==============================================================================

print("--- Starting the Comprehensive Classification Analysis Script ---")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

try:
    # ==========================================================================
    # --- Step 1: Load and Prepare Data ---
    # ==========================================================================
    print(f"\n1. Loading data from: {INPUT_FILE_FULL_PATH}")
    if not os.path.exists(INPUT_FILE_FULL_PATH):
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE_FULL_PATH}")
        
    df = pd.read_csv(INPUT_FILE_FULL_PATH)
    print(f"   Data loaded. Shape: {df.shape}")
    
    # Validate Target Column
    if TARGET_COLUMN not in df.columns:
        possible_targets = ['Deep_Regime', 'Deep_Cluster', 'GMM_Regime']
        found = [col for col in possible_targets if col in df.columns]
        if found:
            print(f"   [WARN] Target '{TARGET_COLUMN}' not found. Using '{found[0]}' instead.")
            TARGET_COLUMN = found[0]
        else:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    y = df[TARGET_COLUMN]
    
    # --- CRITICAL: DROP LEAKAGE & METADATA ---
    # We must remove ADCP data (Answers) and Metadata (Location/Time)
    features_to_drop = [
        # Metadata
        'Date and Time', 'Station ID', 'Measurement No', 
        'Source File', 'Station', 'Base_Station', 'Distance To Sur', 
        'Distance From Shore', 'Season', 'Inside Protected Area', 'Depth GEBCO','Depth ETOP1', 'Location_Tag', 'Season_Name',
        
        # ADCP Data (Data Leakage - The Answers!)
        'Hm0', 'Tp', 'Hmax', 'Hmean', 'Tm02', 'Mean Wave Direction',
        'Wave_Power_Kw', 'Power_Fluctuation', 'Period_Fluctuation', 
        'Angle Fluctuation Std', 'Daily_Max_Hmax', 'Directional Spread',
        
        # Other Clustering Labels
        'GMM_Regime', 'KMeans_Regime', 'Deep_Cluster',
        
        # The Target itself
        TARGET_COLUMN
    ]
    
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    
    print(f"   Features kept for training ({len(X.columns)}): {list(X.columns)}")
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Split (Stratified for stability)
    # CRITICAL: We keep a clean copy of y_test (series) for later evaluations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing
    transformers = [('num', StandardScaler(), numerical_features)]
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    # Transform data
    X_train_full = preprocessor.fit_transform(X_train)
    X_test_full = preprocessor.transform(X_test)
    
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names_out = numerical_features + categorical_features

    # --- DEFINE MODELS ---
    base_models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
    }

    if LGBM_AVAILABLE:
        base_models['LightGBM'] = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    if TABNET_AVAILABLE:
        base_models['TabNet'] = TabNetClassifier(
            verbose=0, 
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax'
        )

    # ==========================================================================
    # --- Step 2: Phase 1 - Baseline (All Features) ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Baseline (All Features)")
    print("="*60)
    
    for name, model in base_models.items():
        try:
            model.fit(X_train_full, y_train)
            evaluate_and_log("1. Baseline", name, model, X_test_full, y_test, note="All Features")
        except Exception as e:
            print(f"   [!] Failed to train {name} in Baseline: {e}")

    # ==========================================================================
    # --- Step 3: Feature Selection ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Feature Selection Strategies")
    print("="*60)

    # --- Method A: RFECV ---
    print("   Running RFECV...")
    rf_selector = RandomForestClassifier(n_jobs=-1, random_state=42)
    rfecv = RFECV(estimator=rf_selector, step=1, cv=StratifiedKFold(3), scoring='f1_weighted', n_jobs=-1)
    rfecv.fit(X_train_full, y_train)
    mask_rfecv = rfecv.support_
    X_train_rfecv = X_train_full[:, mask_rfecv]
    X_test_rfecv = X_test_full[:, mask_rfecv]
    print(f"   -> RFECV kept {sum(mask_rfecv)} features.")

    # --- Method B: Statistical Threshold (Mean) ---
    print("   Running Threshold Selection (Mean)...")
    sfm_selector = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    sfm_selector.fit(X_train_full, y_train)
    sfm = SelectFromModel(estimator=sfm_selector, threshold='mean', prefit=True)
    mask_sfm = sfm.get_support()
    X_train_sfm = X_train_full[:, mask_sfm]
    X_test_sfm = X_test_full[:, mask_sfm]
    features_sfm = [f for f, s in zip(feature_names_out, mask_sfm) if s]
    print(f"   -> Threshold kept {len(features_sfm)} features.")

    # --- Evaluation ---
    for name, model in base_models.items():
        try:
            model.fit(X_train_sfm, y_train)
            evaluate_and_log("3. Threshold Selection", name, model, X_test_sfm, y_test, note=f"{len(features_sfm)} Features")
        except Exception as e:
            print(f"   [!] Failed to train {name} in Selection Phase: {e}")

    # ==========================================================================
    # --- Step 4: Balancing Strategies (On Best Features: Threshold) ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Balancing Strategies")
    print("="*60)

    # --- Strategy A: Class Weights ---
    print("   Testing Class Weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = {i: w for i, w in enumerate(class_weights)}
    
    weighted_models = {
        'RandomForest_W': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'XGBoost_W': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'CatBoost_W': CatBoostClassifier(random_state=42, verbose=0, class_weights=weights_dict)
    }
    
    if LGBM_AVAILABLE:
        weighted_models['LightGBM_W'] = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced')

    for name, model in weighted_models.items():
        model.fit(X_train_sfm, y_train)
        evaluate_and_log("4. Class Weights", name, model, X_test_sfm, y_test, note="Balanced Weights")

    # --- Strategy B: SMOTE ---
    print("   Testing SMOTE...")
    smote_models = {
        'RandomForest_S': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost_S': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), 
        'CatBoost_S': CatBoostClassifier(random_state=42, verbose=0)
    }
    if LGBM_AVAILABLE:
        smote_models['LightGBM_S'] = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    for name, model in smote_models.items():
        try:
            pipeline = ImbPipeline(steps=[('smote', SMOTE(random_state=42)), ('classifier', model)])
            pipeline.fit(X_train_sfm, y_train)
            evaluate_and_log("5. SMOTE", name, pipeline, X_test_sfm, y_test, note="Synthetic Data")
        except Exception as e:
            print(f"   [!] Failed SMOTE for {name}: {e}")

    # ==========================================================================
    # --- Step 5: Final Optimization & ROC ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 4: Optimization & ROC")
    print("="*60)
    
    # Tuning CatBoost with SMOTE
    pipeline_final = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', CatBoostClassifier(random_state=42, verbose=0))
    ])

    param_grid = {
        'classifier__iterations': [200, 500],
        'classifier__depth': [4, 6],
        'classifier__learning_rate': [0.05, 0.1]
    }
    
    print("   Performing GridSearchCV...")
    grid_search = GridSearchCV(pipeline_final, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train_sfm, y_train)
    
    print("   Best Params:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    evaluate_and_log("6. Optimized Model", "CatBoost_Tuned", grid_search, X_test_sfm, y_test, note="Light Tune")

    # --- ROC Curve ---
    print("   Generating ROC Curve...")
    # NOTE: we binarize y_test HERE only for plotting ROC, but keep original y_test for later
    y_test_bin = label_binarize(y_test, classes=sorted(np.unique(y_test)))
    n_classes = y_test_bin.shape[1]
    y_score = best_model.predict_proba(X_test_sfm)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(OUTPUT_DIR, 'Final_Model_ROC_Curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"   - ROC Curve saved to: {roc_path}")

    # ==========================================================================
    # --- Step 6: Stacking Ensemble ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 5: Stacking Ensemble")
    print("="*60)
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42, iterations=300, depth=6))
    ]
    
    if LGBM_AVAILABLE:
        estimators.append(('lgbm', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced')))
    
    final_estimator = LogisticRegression(max_iter=1000)
    
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3, n_jobs=-1)
    stacking_model.fit(X_train_sfm, y_train)
    
    evaluate_and_log("7. Stacking Ensemble", "Stacking_Enhanced", stacking_model, X_test_sfm, y_test, note="RF+XGB+Cat+LGBM")

    # ==========================================================================
    # --- Step 8: (NEW) ROBUSTNESS & UNCERTAINTY ANALYSIS ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 6: ROBUSTNESS & UNCERTAINTY (Q1 Requirements)")
    print("="*60)

    # 8.1 10-Fold Stratified Cross Validation on Best Base Model (e.g. Random Forest + SMOTE Pipeline)
    print("   Running 10-Fold Stratified Cross-Validation (Robustness Check)...")
    
    # We define a robust pipeline to check
    robust_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # We use X_train_sfm (Threshold features) because feature selection is already validated
    cv_scores = cross_val_score(robust_pipeline, X_train_sfm, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"   [ROBUSTNESS] Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"   [ROBUSTNESS] Std Deviation: {cv_scores.std():.4f}")
    print(f"   [INFO] Low Std Dev indicates high model stability.")
    
    # Save CV Report
    with open(os.path.join(OUTPUT_DIR, "Robustness_CrossVal_Report.txt"), "w") as f:
        f.write("ROBUSTNESS ANALYSIS (10-FOLD CV)\n")
        f.write("================================\n")
        f.write(f"Model: Random Forest + SMOTE\n")
        f.write(f"Features: Threshold Selection\n")
        f.write(f"Mean Accuracy: {cv_scores.mean():.4f}\n")
        f.write(f"Std Deviation: {cv_scores.std():.4f}\n")
        f.write(f"Scores per fold: {cv_scores}\n")

    # 8.2 Uncertainty Analysis (Reliability Diagram) on the Best Tuned Model
    # Using 'best_model' from Phase 4
    # Passing 'y_test' directly (which is now guaranteed to be 1D)
    plot_uncertainty_analysis(best_model, X_test_sfm, y_test, OUTPUT_DIR)

    # ==========================================================================
    # --- Step 9: Save the Best Model for Future Use (NEW) ---
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 7: SAVING THE BEST MODEL")
    print("="*60)
    
    # We choose Random Forest + SMOTE as the final robust model
    final_best_model_path = os.path.join(OUTPUT_DIR, 'Final_Model_RF_SMOTE.pkl')
    final_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    
    # We fit it on the FULL dataset (Train + Test) for maximum learning before deployment
    # Using X_train_sfm features (selected features)
    final_pipeline.fit(X_train_sfm, y_train) # Note: Ideally we fit on X_full if we had it prepared, but X_train is safe.
    
    joblib.dump(final_pipeline, final_best_model_path)
    joblib.dump(sfm, os.path.join(OUTPUT_DIR, 'Feature_Selector.pkl')) # Save selector too!
    
    print(f"   [SAVED] Model saved to: {final_best_model_path}")
    print(f"   [SAVED] Feature Selector saved to: Feature_Selector.pkl")
    print("   -> You can load this later using: model = joblib.load('filename.pkl')")

    # ==========================================================================
    # --- Step 10: Final Reports ---
    # ==========================================================================
    print("\n" + "="*60)
    print("Generating Final Reports...")
    print("="*60)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results_data)
    
    # Save CSV Report
    csv_path = os.path.join(OUTPUT_DIR, 'Model_Performance_Report.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"   - Report saved to: {csv_path}")

    # Generate Comparative Bar Plot
    plt.figure(figsize=(18, 10)) 
    sns.set_theme(style="whitegrid")
    
    bar_plot = sns.barplot(
        data=df_results, x='Stage', y='Accuracy', hue='Model', palette='viridis'
    )
    
    plt.title('Model Performance Evolution', fontsize=16)
    plt.xlabel('Processing Stage', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0.50, 0.90) 
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Algorithm')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, 'Model_Evolution_Comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"   - Evolution Plot saved to: {plot_path}")
    print("\n🎉 Full Comprehensive Analysis Finished Successfully!")

except FileNotFoundError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")