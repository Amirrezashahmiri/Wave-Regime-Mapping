#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
 WEC_Unified_Revision_Pipeline.py
==============================================================================
 UNIFIED, END-TO-END PIPELINE (Revision 1 — Ocean Engineering OE-D-26-08011)

 "A Fidelity-Gated Label Propagation Framework for Bathymetry-Conditioned
  Deep Clustering and Decision-Support Site Selection of Wave Energy
  Converters"

 This single file merges the previous 7 separate scripts (CMEMS download,
 ADCP preprocessing, ADCP-CMEMS merge + Fidelity Gate, feature engineering,
 clustering, classification, spatial mapping / site selection) AND adds the
 new analyses requested by the reviewers. Every reviewer-driven addition is
 tagged in the code with  [REV-Rx-Cy]  markers, where:

   [REV-R1-C1]  Fidelity-Gate threshold sensitivity analysis
                (Reviewer 1, comment 1 / Reviewer 2, Major 2)
   [REV-R1-C2]  Leave-One-Station-Out + Leave-One-Region-Out validation
                (Reviewer 1, comment 2 / Reviewer 3, limitations)
   [REV-R1-C4]  Cluster stability across random seeds + quantitative
                comparison of clustering methods (Reviewer 1, comment 4)
   [REV-R1-C5]  Additional independent storm events + non-storm (false-
                alarm) evaluation (Reviewer 1, comment 5)
   [REV-R1-C6]  True Pareto-front computation with explicit depth
                constraint + decision-support caveats (Reviewer 1, comment 6
                / Reviewer 2, Major 6)
   [REV-R2-C3]  Bootstrap 95% confidence intervals, repeated validation
                over multiple random seeds, and statistical significance
                tests for Gate-ON vs Gate-OFF (Reviewer 2, Major 3)
   [REV-R3-C1]  Complete autoencoder architecture / training logging
                (layers, latent dim, activations, loss, optimizer, LR,
                epochs actually trained, batch size, early stopping,
                initialization, seeds) (Reviewer 3, Methods 1)
   [REV-R3-C2]  Contiguous block-wise 10-fold CV with explicit,
                documented fold construction across stations and time
                (Reviewer 3, Methods 2)
   [REV-R3-C3]  Additional ensemble learning algorithms compared, with
                full hyper-parameter reporting for EVERY model incl. the
                stacking meta-learner (Reviewer 3, Results 2)
   [REV-R2-REP] Reproducibility log: software versions, hardware,
                computation time per stage (Reviewer 2, Methods 1)
   [REV-TABLES] Combined comprehensive result tables
                (Reviewer 3 "Key results should be combined into
                comprehensive tables"; Reviewer 2 Results 1)
   [REV-FIG10]  Reworked "model evolution" figure so that it reports
                Balanced Accuracy AND Cohen's Kappa (Reviewer 3:
                "Figure 10 does not clearly support the stated conclusion")

 NOTE ON SCOPE (per author instruction): ONLY reviewer-requested items were
 added/changed. Pre-existing behaviour that reviewers did not comment on is
 preserved as-is.

 HOW TO RUN
 ----------
   1) Edit the CONFIG block below (paths + RUN_STAGES switches).
   2) python WEC_Unified_Revision_Pipeline.py
   Stages write their outputs to disk exactly like the original pipeline, so
   you can also run stages one at a time by toggling RUN_STAGES.

 All revision outputs are written under  <REVISION_DIR>  in per-comment
 sub-folders, each figure/table/text in its own separate file.
==============================================================================
"""
from __future__ import annotations

# =============================================================================
# SECTION 0 — GLOBAL IMPORTS, CONFIG, SEEDS, REPRODUCIBILITY  [REV-R2-REP]
# =============================================================================
import os
os.environ['SCIPY_ARRAY_API'] = '1'
import re
import sys
import gc
import json
import time
import random
import pathlib
import platform
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from itertools import cycle, combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # non-interactive backend (large figures, no Tkinter)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.text  # needed by save_fig font enforcement
from matplotlib.colors import LogNorm
import seaborn as sns

import joblib
import xarray as xr

from scipy import stats as sps

from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             adjusted_mutual_info_score, roc_curve, auc,
                             roc_auc_score, accuracy_score, confusion_matrix,
                             classification_report, balanced_accuracy_score,
                             cohen_kappa_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import copernicusmarine as cm
    CM_AVAILABLE = True
except ImportError:
    CM_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WEC_REVISION")

# -----------------------------------------------------------------------------
# 0.1  CONFIG — EDIT PATHS HERE (kept consistent with the original scripts)
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(r"C:\Users\DFMRendering\Desktop\Wave Energy\New Try")

CONFIG = {
    # ---- Stage 1: CMEMS download -------------------------------------------
    "CMEMS_USERNAME": "",                       # <- CMEMS username (once)
    "CMEMS_PASSWORD": "",                       # <- CMEMS password (once)
    "CMEMS_OUT": ROOT / "Unified" / "S1_cmems_downloads",

    # ---- Stage 2: ADCP preprocessing ---------------------------------------
    "ADCP_INPUT_DIR": ROOT / "Test" / "ADCP",   # folder of station .xlsx files
    "ADCP_OUT": ROOT / "Unified" / "S2_adcp_clean",

    # ---- Stage 3: merge + fidelity gate ------------------------------------
    # Primary NC folder; if it is empty/missing, the code falls back
    # automatically to PATH_NC_LEGACY (the original download folder).
    "PATH_NC":   ROOT / "Unified" / "S1_cmems_downloads",
    "PATH_NC_LEGACY": ROOT / "Test" / "Output Code 1" / "cmems_downloads",
    "PATH_XLSX": ROOT / "Unified" / "S2_adcp_clean",
    "PATH_BATHY": ROOT / "Test" / "Output Code 1" / "cmems_downloads" /
                  "Bathymetry Statics" /
                  "cmems_mod_glo_wav_my_0.2deg_static_1765282455944.nc",
    "MERGE_OUT": ROOT / "Unified" / "S3_merge_fidelity",

    # ---- Stage 4: feature engineering / balancing --------------------------
    "FEATURE_OUT": ROOT / "Unified" / "S4_features",

    # ---- Stage 5: clustering ------------------------------------------------
    "CLUSTER_OUT": ROOT / "Unified" / "S5_clustering",

    # ---- Stage 6: classification -------------------------------------------
    "CLASSIF_OUT": ROOT / "Unified" / "S6_classification",

    # ---- Stage 7: spatial mapping / site selection -------------------------
    # [REV] repointed to the complete 17-variable regional download
    # (cmems_download_regional_grid.py) which replaces the old
    # Oman_Coast_Wave_Data_2023.nc that was missing 10 of the 22 trained
    # features (they were being zero-filled, collapsing Storm/Confused
    # predictions to ~0 on the spatial grid).
    "LOCAL_NC_FILE": ROOT / "Test" / "Output Code 7" /
                     "Oman_Coast_Wave_Data_FULL_17vars_2015_2025.nc",
    "MAPS_OUT": ROOT / "Unified" / "S7_maps",

    # ---- Revision outputs (per-reviewer-comment sub-folders) ---------------
    "REVISION_DIR": ROOT / "Unified" / "Revision_Outputs",
}

RUN_STAGES = {
    "S1_download":        False,  # per-station files unchanged & already on
                                  # disk (Stage 3 diagnostics confirmed all
                                  # 23 match) -> set True only for a true
                                  # from-scratch run (needs CMEMS login)
    "S2_adcp_preprocess": False,  # already run
    "S3_merge_fidelity":  False,
    "S4_features":        False,
    "S5_clustering":      False,
    "S6_classification":  False,
    "S7_maps":            False,  # results already correct
    "S8_master_report":   True,   # collect ALL results into ONE big txt
}

# -----------------------------------------------------------------------------
# 0.2  GLOBAL CONSTANTS (identical to the original pipeline unless tagged REV)
# -----------------------------------------------------------------------------
GLOBAL_SEED = 42

# Fidelity Gate thresholds (as in the manuscript)
MIN_CORR_THRESHOLD = 0.80          # Pearson R on Hm0
MAX_BIAS_THRESHOLD = 0.50          # |bias| on Hm0 (m)

# [REV-R1-C1] threshold grids for the sensitivity analysis
GATE_CORR_GRID = [0.70, 0.75, 0.80, 0.85, 0.90]
GATE_BIAS_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]

# [REV-R1-C4] number of random seeds for cluster-stability analysis
CLUSTER_STABILITY_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# [REV-R2-C3] repeated-validation seeds + bootstrap settings
REPEATED_EVAL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

N_CLUSTERS = 4
FEATURES_FOR_CLUSTERING = [
    'Hm0', 'Tp', 'Wave_Power_Kw', 'Angle Fluctuation Std',
    'Power_Fluctuation', 'Period_Fluctuation', 'Daily_Max_Hmax'
]
FEATURE_UNITS = {
    'Hm0': '(m)', 'Tp': '(s)', 'Wave_Power_Kw': '(kW/m)',
    'Angle Fluctuation Std': '(deg)', 'Power_Fluctuation': '(kW/m)',
    'Period_Fluctuation': '(s)', 'Daily_Max_Hmax': '(m)'
}

# Cyclone windows: Shaheen & Mekunu were in the submitted manuscript.
# [REV-R1-C5] Hikaa + Luban were ADDED as independent events, plus two
# quiet-season "non-storm" control windows for false-alarm quantification.
STORM_EVENTS = {
    "Shaheen": {"start": datetime(2021, 10, 3),  "end": datetime(2021, 10, 4),
                "region": "north"},
    "Mekunu":  {"start": datetime(2018, 5, 24),  "end": datetime(2018, 5, 27),
                "region": "south"},
    "Hikaa":   {"start": datetime(2019, 9, 23),  "end": datetime(2019, 9, 26),
                "region": "central"},                          # [REV-R1-C5]
    "Luban":   {"start": datetime(2018, 10, 10), "end": datetime(2018, 10, 14),
                "region": "south"},                            # [REV-R1-C5]
}
# [REV-R1-C5, non-circular false-alarm design] The false-alarm baseline is
# the model's background Storm-classification rate over every timestamp
# NOT within EXCLUSION_BUFFER_DAYS of a documented named storm (dates are
# independently known historical events — not derived from any wave
# variable the classifier consumes, so this cannot be circular). An
# earlier version selected "calm" windows by ranking CMEMS wave height
# itself, which is circular (VHM0 is a training feature that drives the
# Storm/Calm split) and was replaced with this design.
WINDOW_DAYS = 4              # matches the storm-event window length
EXCLUSION_BUFFER_DAYS = 14   # days around each storm excluded as "event"
STORM_CLASS = 1  # label index of the Storm regime in Deep_Regime

# Regions used for regional statistics + Leave-One-Region-Out  [REV-R1-C2]
REGION_BOUNDS = {
    "north":   lambda lat: lat > 22.5,
    "central": lambda lat: (lat >= 18.0) & (lat <= 22.5),
    "south":   lambda lat: lat < 18.0,
}

# Publication figure style (unchanged from the original scripts)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # Times-like math glyphs
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 26
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['legend.fontsize'] = 22
matplotlib.rcParams['legend.title_fontsize'] = 22
matplotlib.rcParams['figure.titlesize'] = 26
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.05
matplotlib.rcParams['savefig.dpi'] = 600           # journal-quality raster
matplotlib.rcParams['pdf.fonttype'] = 42           # embed TrueType (editable)
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['lines.linewidth'] = 3.0
matplotlib.rcParams['grid.alpha'] = 0.4
matplotlib.rcParams['grid.linestyle'] = '--'

# Snapshot of the publication style, so it can be re-imposed after any call
# that resets rcParams (notably seaborn's set_theme / set_style, which
# silently reverts font.family to a sans-serif default).
_PUB_STYLE = dict(matplotlib.rcParams)

def apply_publication_style():
    """[REV] Re-impose Times New Roman + sizing. Call after any seaborn
    theme change so every exported figure is typographically consistent."""
    for k, v in _PUB_STYLE.items():
        try:
            matplotlib.rcParams[k] = v
        except Exception:
            pass

def _verify_times_new_roman():
    """Warn once, loudly, if Times New Roman is not actually installed —
    otherwise matplotlib silently substitutes a fallback face and the
    figures ship with the wrong typeface."""
    # The per-element font enforcement in save_fig() triggers one findfont
    # lookup per text object, so a missing font would otherwise emit
    # thousands of identical warnings and bury the real log. Silence the
    # repeats; the single explicit report below is the signal that matters.
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*findfont.*')
    try:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        if 'Times New Roman' in names:
            print("[STYLE] Times New Roman found; figures will use it.")
        else:
            print("[STYLE] !! WARNING: 'Times New Roman' is NOT installed. "
                  "Matplotlib will substitute a fallback serif face.")
            print("[STYLE]    Windows: it ships with MS Office/Windows "
                  "(C:\\Windows\\Fonts\\times.ttf).")
            print("[STYLE]    If missing, install it, then delete the "
                  "matplotlib font cache and re-run:")
            print("[STYLE]      python -c \"import matplotlib as m,os,shutil;"
                  "shutil.rmtree(m.get_cachedir(),ignore_errors=True)\"")
    except Exception as e:
        print(f"[STYLE] font check skipped: {e}")
HIGH_DPI = 600   # [REV] journal-quality raster output

# [REV] Canonical regime colours. Colours are bound to PHYSICAL MEANING, not
# to Gaussian-mixture component order, so a figure's colour scheme can never
# permute between runs the way a colour-by-index scheme does. Chosen to be
# intuitive: red = danger (Storm), green = favourable (Golden Swell),
# blue = quiescent (Ambient/Calm), amber = caution (Confused Sea).
REGIME_NAMES_CANON = {0: 'Confused Sea', 1: 'Storm',
                      2: 'Ambient/Calm', 3: 'Golden Swell'}
REGIME_COLORS = {0: '#E8B21A',   # amber  - Confused Sea
                 1: '#D62728',   # red    - Storm
                 2: '#1F77B4',   # blue   - Ambient/Calm
                 3: '#2CA02C'}   # green  - Golden Swell

def regime_color(idx):
    return REGIME_COLORS.get(int(idx), '#7F7F7F')

def regime_label(idx):
    return f"{int(idx)} - {REGIME_NAMES_CANON.get(int(idx), 'Cluster')}"

# -----------------------------------------------------------------------------
# 0.3  SEEDS + REPRODUCIBILITY LOG  [REV-R2-REP] [REV-R3-C1]
# -----------------------------------------------------------------------------
def set_global_seeds(seed: int = GLOBAL_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
    print(f"[SEED] Global random seed set to {seed}")

set_global_seeds(GLOBAL_SEED)

STAGE_TIMINGS: Dict[str, float] = {}

class StageTimer:
    """Context manager that records wall-clock time per stage [REV-R2-REP]."""
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
        print("\n" + "=" * 78)
        print(f"  STAGE START: {self.name}")
        print("=" * 78)
        return self
    def __exit__(self, *exc):
        dt = time.time() - self.t0
        STAGE_TIMINGS[self.name] = dt
        print(f"  STAGE END:   {self.name}  ({dt:,.1f} s)")

def get_environment_report() -> str:
    """Software versions + hardware description  [REV-R2-REP]."""
    import sklearn, scipy
    lines = []
    lines.append("REPRODUCIBILITY / COMPUTATIONAL-ENVIRONMENT REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated:        {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Global seed:      {GLOBAL_SEED}")
    lines.append(f"Python:           {sys.version.split()[0]}")
    lines.append(f"Platform:         {platform.platform()}")
    lines.append(f"Processor:        {platform.processor()}")
    try:
        import psutil
        lines.append(f"Logical CPUs:     {psutil.cpu_count(logical=True)}")
        lines.append(f"RAM (GB):         {psutil.virtual_memory().total/1e9:.1f}")
    except Exception:
        lines.append(f"Logical CPUs:     {os.cpu_count()}")
    lines.append("-" * 60)
    lines.append("Library versions:")
    lines.append(f"  numpy         {np.__version__}")
    lines.append(f"  pandas        {pd.__version__}")
    lines.append(f"  scipy         {scipy.__version__}")
    lines.append(f"  scikit-learn  {sklearn.__version__}")
    lines.append(f"  matplotlib    {matplotlib.__version__}")
    lines.append(f"  seaborn       {sns.__version__}")
    lines.append(f"  xarray        {xr.__version__}")
    try:
        import xgboost; lines.append(f"  xgboost       {xgboost.__version__}")
    except Exception: pass
    try:
        import catboost; lines.append(f"  catboost      {catboost.__version__}")
    except Exception: pass
    if LGBM_AVAILABLE:
        import lightgbm; lines.append(f"  lightgbm      {lightgbm.__version__}")
    if TF_AVAILABLE:
        lines.append(f"  tensorflow    {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        lines.append(f"  TF GPUs:      {len(gpus)} ({[g.name for g in gpus]})")
    lines.append("-" * 60)
    lines.append("Wall-clock time per stage (s):")
    for k, v in STAGE_TIMINGS.items():
        lines.append(f"  {k:<42} {v:>10.1f}")
    return "\n".join(lines)

def ensure_dirs():
    for key in ["CMEMS_OUT", "ADCP_OUT", "MERGE_OUT", "FEATURE_OUT",
                "CLUSTER_OUT", "CLASSIF_OUT", "MAPS_OUT", "REVISION_DIR"]:
        pathlib.Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)
    # Per-comment revision sub-folders — every artefact in a separate file
    for sub in ["R1C1_FidelityGate_Sensitivity",
                "R1C2_LOSO_LORO_Validation",
                "R1C4_Cluster_Stability",
                "R1C5_Storm_And_FalseAlarm_Validation",
                "R1C6_Pareto_Site_Selection",
                "R2C3_CIs_RepeatedValidation_Stats",
                "R3C1_Model_Architecture_Hyperparams",
                "R3C2_Blockwise_CV",
                "R3C3_Extra_Ensembles",
                "Combined_Tables",
                "Reproducibility"]:
        (pathlib.Path(CONFIG["REVISION_DIR"]) / sub).mkdir(parents=True,
                                                           exist_ok=True)

def rev_dir(sub: str) -> pathlib.Path:
    return pathlib.Path(CONFIG["REVISION_DIR"]) / sub

def save_fig(fig, folder: pathlib.Path, name: str):
    """Save a figure as PNG (600 DPI) + vector PDF, each in its own file.
    [REV] Before saving, every text element in the figure is forced onto the
    publication serif family. This catches labels created by seaborn or by
    helper calls that captured a different font before the style was
    restored, so no exported figure can ship with a fallback typeface."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    serif = matplotlib.rcParams['font.serif']
    try:
        for txt in fig.findobj(match=matplotlib.text.Text):
            txt.set_family('serif')
            txt.set_fontname(serif[0] if isinstance(serif, (list, tuple))
                             else serif)
    except Exception:
        pass   # never let cosmetics break a run
    fig.savefig(folder / f"{name}.png", dpi=HIGH_DPI, bbox_inches='tight')
    fig.savefig(folder / f"{name}.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)
    gc.collect()

def save_text(folder: pathlib.Path, name: str, text: str):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / name, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   [TXT] saved -> {folder / name}")

# =============================================================================
# SECTION 1 — STAGE 1: CMEMS POINT DOWNLOADS (unchanged behaviour)
# =============================================================================
CMEMS_COMMON: dict = {
    "dataset_id": "cmems_mod_glo_wav_my_0.2deg_PT3H-i",
    "dataset_version": "202411",
    "variables": [
        "VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW",
        "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW",
        "VPED", "VSDX", "VSDY",
        "VTM01_SW1", "VTM01_SW2", "VTM01_WW",
        "VTM02", "VTM10", "VTPK",
    ],
    "coordinates_selection_method": "nearest",
    "netcdf_compression_level": 1,
    "disable_progress_bar": True,
}

STATIONS: dict = {
    "Barka":          dict(lon=58.0862,    lat=23.7700,
                           segments=[("2018-05-16T03:00:00", "2018-10-15T06:00:00")]),
    "Duqum":          dict(lon=57.8052167, lat=19.7719833,
                           segments=[("2019-02-15T09:00:00", "2019-04-15T06:00:00")]),
    "Fahal":          dict(lon=58.4991667, lat=23.6593667, segments=[
                           ("2017-11-23T12:00:00", "2017-12-24T09:00:00"),
                           ("2017-12-25T15:00:00", "2018-04-03T09:00:00"),
                           ("2018-04-10T06:00:00", "2018-07-02T09:00:00"),
                           ("2018-10-23T09:00:00", "2019-01-13T03:00:00")]),
    "Ghubrah":        dict(lon=58.4067500, lat=23.6212833,
                           segments=[("2018-03-06T09:00:00", "2018-05-15T06:00:00")]),
    "Masirah":        dict(lon=58.6789,    lat=20.1596833333333,
                           segments=[("2019-02-17T12:00:00", "2019-04-16T03:00:00")]),
    "Inshore_Suwayq": dict(lon=57.44221,   lat=23.94223,  segments=[
                           ("2021-10-07T06:00:00", "2022-01-27T09:00:00"),
                           ("2022-07-27T03:00:00", "2022-10-12T03:00:00")]),
    "Quriyat_North":  dict(lon=58.9260,    lat=23.2957,
                           segments=[("2018-11-08T09:00:00", "2019-01-10T09:00:00")]),
    "Quriyat_South":  dict(lon=58.925924,  lat=23.281023, segments=[
                           ("2021-07-08T09:00:00", "2021-10-12T03:00:00"),
                           ("2021-10-20T03:00:00", "2022-01-19T06:00:00"),
                           ("2022-02-10T03:00:00", "2022-03-31T12:00:00"),
                           ("2022-05-23T03:00:00", "2022-07-26T00:00:00")]),
    "Shywaimiya":     dict(lon=55.5511833, lat=17.8568333,
                           segments=[("2019-02-13T12:00:00", "2019-04-14T12:00:00")]),
    "Taqah":          dict(lon=54.3444833, lat=17.0157167,
                           segments=[("2019-02-11T12:00:00", "2019-04-11T06:00:00")]),
    "Raqqat_Suwayq":  dict(lon=57.49815,   lat=24.03608,  segments=[
                           ("2021-01-18T03:00:00", "2021-09-30T03:00:00"),
                           ("2021-10-07T09:00:00", "2022-01-27T03:00:00"),
                           ("2022-05-11T03:00:00", "2022-10-14T06:00:00")]),
    "Wudam_North":    dict(lon=57.61904,   lat=23.87524,
                           segments=[("2021-03-18T12:00:00", "2021-06-16T03:00:00")]),
    "Wudam_South":    dict(lon=57.59747,   lat=23.81999,
                           segments=[("2021-03-18T12:00:00", "2021-06-16T03:00:00")]),
    "Sawadi":         dict(lon=57.77933,   lat=23.80496,
                           segments=[("2022-01-27T06:00:00", "2022-05-11T03:00:00")]),
}

def stage1_cmems_download():
    if not CM_AVAILABLE:
        print("[S1] copernicusmarine not installed — skipping download stage.")
        return
    outdir = pathlib.Path(CONFIG["CMEMS_OUT"])
    outdir.mkdir(parents=True, exist_ok=True)
    if CONFIG["CMEMS_USERNAME"]:
        cm.login(CONFIG["CMEMS_USERNAME"], CONFIG["CMEMS_PASSWORD"])
    for site, meta in STATIONS.items():
        lon, lat = meta["lon"], meta["lat"]
        for idx, (t0, t1) in enumerate(meta["segments"], start=1):
            print(f"[S1] {site:<15} | segment {idx:<2} | {t0} -> {t1}")
            response = cm.subset(
                minimum_longitude=lon, maximum_longitude=lon,
                minimum_latitude=lat, maximum_latitude=lat,
                start_datetime=t0, end_datetime=t1,
                output_directory=outdir,
                output_filename=f"{site}_segment_{idx}.nc",
                **CMEMS_COMMON,
            )
            print(f"     saved -> {response.file_path}")
    print("[S1] All downloads finished.")

# =============================================================================
# SECTION 2 — STAGE 2: ADCP PREPROCESSING (unchanged scientific behaviour;
#             now loops over every station file in ADCP_INPUT_DIR)
# =============================================================================
def circular_mean(angles: pd.Series) -> float:
    if angles.isna().all():
        return np.nan
    rads = np.deg2rad(angles.dropna())
    return np.degrees(np.arctan2(np.sum(np.sin(rads)),
                                 np.sum(np.cos(rads)))) % 360

def circular_std(angles: pd.Series) -> float:
    """Yamartino-type circular standard deviation."""
    if angles.isna().all():
        return np.nan
    rad = np.deg2rad(angles.dropna())
    if len(rad) == 0:
        return np.nan
    R = np.sqrt(np.sum(np.sin(rad))**2 + np.sum(np.cos(rad))**2) / len(rad)
    if R == 0:
        return np.inf
    R = min(R, 1.0)
    return np.rad2deg(np.sqrt(-2 * np.log(R)))

def detect_and_handle_outliers(df: pd.DataFrame, columns: List[str],
                               method: str = 'ensemble', threshold: float = 3,
                               window_sizes: List[int] = [6, 12, 24]) -> pd.DataFrame:
    """Ensemble outlier detection (Hampel + Z-score + IQR, 2-of-3 voting)."""
    df_clean = df.copy()
    for col in columns:
        if df[col].notna().sum() < 10:
            continue
        best_window = window_sizes[0]
        min_outliers = float('inf')
        for w in window_sizes:
            rmed = df[col].rolling(w, center=True, min_periods=1).median()
            rmad = (df[col] - rmed).abs().rolling(w, center=True,
                                                  min_periods=1).median()
            cnt = ((df[col] - rmed).abs() > threshold * rmad).sum()
            if 0 < cnt < min_outliers:
                min_outliers = cnt
                best_window = w
        roll_med = df[col].rolling(best_window, center=True,
                                   min_periods=1).median()
        roll_mad = (df[col] - roll_med).abs().rolling(best_window, center=True,
                                                      min_periods=1).median()
        mask_hampel = (df[col] - roll_med).abs() > threshold * roll_mad
        z = (df[col] - df[col].mean()) / df[col].std()
        mask_z = z.abs() > threshold
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask_iqr = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
        if method == 'ensemble':
            final_mask = (mask_hampel.astype(int) + mask_z.astype(int)
                          + mask_iqr.astype(int)) >= 2
        else:
            final_mask = mask_hampel
        if final_mask.sum() > 0:
            df_clean.loc[final_mask, col] = roll_med[final_mask]
            logger.info(f"Outliers replaced in {col}: {final_mask.sum()}")
    return df_clean

def dineof_imputation(df: pd.DataFrame, numeric_cols: List[str],
                      max_iter: int = 50) -> pd.DataFrame:
    """Iterative SVD (DINEOF-style) gap filling."""
    valid_cols = [c for c in numeric_cols
                  if c in df.columns and df[c].notna().sum() > 5]
    if not valid_cols:
        return df
    data = df[valid_cols].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.fillna(data.mean()))
    missing_mask = np.isnan(df[valid_cols].values)
    prev_rmse = float('inf')
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(data_scaled, full_matrices=False)
        k = min(3, data_scaled.shape[1])
        s[k:] = 0
        recon = U @ np.diag(s) @ Vt
        rmse = np.sqrt(np.mean((data_scaled[~missing_mask]
                                - recon[~missing_mask]) ** 2))
        if abs(prev_rmse - rmse) < 1e-5:
            break
        prev_rmse = rmse
        data_scaled[missing_mask] = recon[missing_mask]
    imputed = scaler.inverse_transform(data_scaled)
    df_out = df.copy()
    for idx, col in enumerate(valid_cols):
        miss = df[col].isna()
        df_out.loc[miss, col] = imputed[miss.values, idx]
    return df_out

def sophisticated_imputation(df: pd.DataFrame, process_cols: List[str],
                             max_gap: int = 6) -> pd.DataFrame:
    directional_cols = ['Mean Wave Direction']
    linear_cols = [c for c in process_cols if c not in directional_cols]
    df = detect_and_handle_outliers(df, linear_cols, method='ensemble')
    for col in directional_cols:
        if col in df.columns and df[col].isna().any():
            rad = np.radians(df[col])
            df[f"{col}_sin"] = np.sin(rad)
            df[f"{col}_cos"] = np.cos(rad)
            for comp in [f"{col}_sin", f"{col}_cos"]:
                df[comp] = df[comp].interpolate(method='time', limit=max_gap,
                                                limit_area='inside')
            mask_valid = df[f"{col}_sin"].notna() & df[f"{col}_cos"].notna()
            df.loc[mask_valid & df[col].isna(), col] = np.degrees(np.arctan2(
                df.loc[mask_valid, f"{col}_sin"],
                df.loc[mask_valid, f"{col}_cos"])) % 360
            df = df.drop(columns=[f"{col}_sin", f"{col}_cos"])
    for col in linear_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method='time', limit=max_gap,
                                          limit_area='inside')
    strategies = {'Hm0': 'dineof', 'Tp': 'dineof', 'Hmax': 'dineof',
                  'Hmean': 'dineof', 'Tm02': 'dineof'}
    for col in linear_cols:
        if not df[col].isna().any():
            continue
        if strategies.get(col, 'hybrid') == 'dineof':
            try:
                vars_to_use = [col]
                if 'Hm0' in df.columns and col != 'Hm0':
                    vars_to_use.append('Hm0')
                if 'Tp' in df.columns and col != 'Tp':
                    vars_to_use.append('Tp')
                df = dineof_imputation(df, vars_to_use)
            except Exception as e:
                logger.warning(f"DINEOF failed for {col}: {e} -> KNN fallback")
                try:
                    df[col] = KNNImputer(n_neighbors=5).fit_transform(df[[col]])
                except Exception:
                    pass
        if df[col].isna().any():
            df[col] = df[col].ffill(limit=max_gap).bfill(limit=max_gap)
    return df

def preprocess_one_adcp_file(input_file: pathlib.Path, output_dir: pathlib.Path):
    base_name = input_file.stem
    df = pd.read_excel(input_file, sheet_name='WAVE')
    df['Date and Time'] = pd.to_datetime(df['Date and Time'], errors='coerce')
    df = df.dropna(subset=['Date and Time'])
    # Oman local time (UTC+4) -> UTC
    df['Date and Time'] = df['Date and Time'] - pd.Timedelta(hours=4)
    df = df.set_index('Date and Time').sort_index()

    # Raw ADCP files name the direction column 'MeanDir' (original Code 2
    # renames its circular-std aggregate to 'Angle Fluctuation Std' and adds
    # 'Mean Wave Direction' via the circular mean) — replicated exactly here.
    dir_src = next((c for c in ('MeanDir', 'Mean Wave Direction')
                    if c in df.columns), None)
    agg_rules = {c: 'mean' for c in df.select_dtypes(include=[np.number]).columns}
    if dir_src in agg_rules:
        agg_rules.pop(dir_src)
    df_resampled = df.resample('3h').agg(agg_rules)
    if dir_src is not None:
        df_resampled['Mean Wave Direction'] = (
            df[dir_src].resample('3h').apply(circular_mean))
        df_resampled['Angle Fluctuation Std'] = (
            df[dir_src].resample('3h').apply(circular_std))
    if 'Hm0' in df_resampled.columns:
        df_resampled = df_resampled.dropna(subset=['Hm0'])   # as in Code 2
    else:
        df_resampled = df_resampled.dropna(how='all')

    process_cols = [c for c in ['Hm0', 'Tp', 'Mean Wave Direction',
                                'Angle Fluctuation Std', 'Tm02', 'Hmax',
                                'Hmean']
                    if c in df_resampled.columns]

    # split into continuous segments at gaps > 24 h, clean each segment
    segment_id = (df_resampled.index.to_series().diff()
                  > pd.Timedelta(hours=24)).cumsum()
    df_resampled['segment_id'] = segment_id
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, segment) in enumerate(df_resampled.groupby('segment_id')):
        segment = segment.drop(columns='segment_id')
        if segment.empty:
            continue
        seg_raw = segment.copy()
        seg_clean = sophisticated_imputation(segment, process_cols, max_gap=6)
        out_name = output_dir / f'{base_name}_segment_{i+1}_resampled.xlsx'
        seg_clean.reset_index().to_excel(out_name, index=False)
        print(f"[S2]   saved -> {out_name}")
        for col in process_cols:
            if col not in seg_clean.columns:
                continue
            fig = plt.figure(figsize=(10, 5))
            plt.plot(seg_raw.index, seg_raw[col], label='Raw',
                     color='blue', alpha=0.5)
            plt.plot(seg_clean.index, seg_clean[col], label='Imputed',
                     color='red', alpha=0.5, linestyle='--')
            plt.title(f"{col} - {base_name} seg {i+1}")
            plt.legend()
            fig.savefig(plot_dir / f'{base_name}_{col}_seg{i+1}.png', dpi=150)
            plt.close(fig)

def stage2_adcp_preprocess():
    in_dir = pathlib.Path(CONFIG["ADCP_INPUT_DIR"])
    out_dir = pathlib.Path(CONFIG["ADCP_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.xls*"))
    if not files:
        print(f"[S2] No ADCP Excel files found in {in_dir}")
        return
    for f in files:
        print(f"[S2] Preprocessing {f.name} ...")
        try:
            preprocess_one_adcp_file(f, out_dir)
        except Exception as e:
            print(f"[S2]   ERROR on {f.name}: {e}")

# =============================================================================
# SECTION 3 — STAGE 3: ADCP–CMEMS MERGE + FIDELITY GATE
#             + [REV-R1-C1] GATE-THRESHOLD SENSITIVITY ANALYSIS
# =============================================================================
TAG_RE = re.compile(r"(?P<tag>[A-Za-z0-9_]+_segment_\d+)")

# Known spelling variants between the ADCP Excel names and the CMEMS
# station catalogue — both sides are normalized to the SAME canonical name.
TAG_ALIASES = {
    "Gubrah": "Ghubrah",     # ADCP file 'Gubrah.xlsx' vs catalogue 'Ghubrah'
    "Duqm":   "Duqum",
}

def extract_tag(name: str) -> Optional[str]:
    m = TAG_RE.search(name.replace(" ", "_"))
    if not m:
        return None
    tag = m.group("tag")
    station, _, seg = tag.rpartition("_segment_")
    station = TAG_ALIASES.get(station, station)
    return f"{station}_segment_{seg}"

def get_season(month: int) -> Optional[int]:
    if month in [12, 1, 2]: return 1
    if month in [3, 4, 5]:  return 2
    if month in [6, 7, 8]:  return 3
    if month in [9, 10, 11]: return 4
    return None

def stage3_merge_fidelity():
    PATH_NC = pathlib.Path(CONFIG["PATH_NC"])
    # automatic fallback to the original download folder if empty/missing
    if not PATH_NC.exists() or not any(PATH_NC.glob("*.nc")):
        legacy = pathlib.Path(CONFIG.get("PATH_NC_LEGACY", PATH_NC))
        if legacy.exists() and any(legacy.glob("*.nc")):
            print(f"[S3] PATH_NC has no .nc files -> falling back to legacy "
                  f"folder:\n[S3]   {legacy}")
            PATH_NC = legacy
    PATH_XLSX = pathlib.Path(CONFIG["PATH_XLSX"])
    PATH_BATHY = pathlib.Path(CONFIG["PATH_BATHY"])
    OUTPUT_DIR = pathlib.Path(CONFIG["MERGE_OUT"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- INPUT DIAGNOSTICS (printed every run) ----------------
    print("[S3] " + "-" * 66)
    print("[S3] INPUT DIAGNOSTICS")
    for label, p, pattern in [("PATH_NC   (CMEMS NetCDF)", PATH_NC, "*.nc"),
                              ("PATH_XLSX (clean ADCP)  ", PATH_XLSX,
                               "*.xls*")]:
        print(f"[S3] {label}: {p}")
        if not p.exists():
            print("[S3]   !! FOLDER DOES NOT EXIST !!")
            continue
        found = sorted(p.glob(pattern))
        print(f"[S3]   {len(found)} file(s) matching '{pattern}':")
        for f in found[:60]:
            print(f"[S3]     {f.name:<55s} -> tag: {extract_tag(f.name)}")
        others = [f.name for f in sorted(p.iterdir())
                  if f.is_file() and f not in found]
        if others:
            print(f"[S3]   other (non-matching) files: {others[:20]}")
    print("[S3] " + "-" * 66)

    nc_files = {extract_tag(f.name): f for f in PATH_NC.glob("*.nc")
                if extract_tag(f.name)} if PATH_NC.exists() else {}
    xlsx_files = {extract_tag(f.name): f for f in PATH_XLSX.glob("*.xls*")
                  if extract_tag(f.name)} if PATH_XLSX.exists() else {}
    tags = sorted(set(nc_files) & set(xlsx_files))
    print(f"[S3] NC tags   ({len(nc_files)}): {sorted(nc_files)}")
    print(f"[S3] XLSX tags ({len(xlsx_files)}): {sorted(xlsx_files)}")
    print(f"[S3] Intersection: {len(tags)} matching pair(s)")

    if not tags:
        # deep structure dump of one sample from each side, if available
        sample_nc = next(iter(PATH_NC.glob("*.nc")), None) \
            if PATH_NC.exists() else None
        if sample_nc is not None:
            print(f"\n[S3] ---- SAMPLE NetCDF STRUCTURE: {sample_nc.name} ----")
            try:
                with xr.open_dataset(sample_nc) as ds_s:
                    print(ds_s)
            except Exception as e:
                print(f"[S3]   could not open: {e}")
        sample_x = next(iter(PATH_XLSX.glob("*.xls*")), None) \
            if PATH_XLSX.exists() else None
        if sample_x is not None:
            print(f"\n[S3] ---- SAMPLE EXCEL STRUCTURE: {sample_x.name} ----")
            try:
                dfx = pd.read_excel(sample_x, engine="openpyxl", nrows=5)
                print("[S3]   columns:", list(dfx.columns))
                print(dfx.head().to_string())
            except Exception as e:
                print(f"[S3]   could not open: {e}")
        raise SystemExit(
            "\n[S3] No matching station-segment tags found.\n"
            "[S3] HINT: Stage 1 was skipped, so CONFIG['PATH_NC'] must point "
            "to the folder that ALREADY contains the per-station CMEMS "
            "NetCDF files (e.g. the output folder of the original download "
            "script), OR copy those .nc files into "
            f"{PATH_NC}.\n"
            "[S3] File names on both sides must contain the same "
            "'<Station>_segment_<n>' tag (spaces are treated as "
            "underscores).")
    print(f"[S3] Found {len(tags)} matching file pairs.")

    ds_bathy = xr.open_dataset(PATH_BATHY)

    best_match_stats = []
    all_merged_frames: Dict[str, pd.DataFrame] = {}

    physics_pairs = [("Hm0", "VHM0"), ("Tp", "VTPK"),
                     ("Tm02", "VTM02"), ("Mean Wave Direction", "VMDR")]

    for tag in tags:
        nc_path, xlsx_path = nc_files[tag], xlsx_files[tag]
        print(f"[S3] Processing {tag} ...")
        df_xls = pd.read_excel(xlsx_path, engine="openpyxl")
        if "Date and Time" not in df_xls.columns:
            warnings.warn(f"{xlsx_path.name}: missing 'Date and Time' - skipped.")
            continue
        df_xls["Date and Time"] = pd.to_datetime(df_xls["Date and Time"],
                                                 errors="coerce")
        df_xls = df_xls.dropna(subset=["Date and Time"]).sort_values("Date and Time")

        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            st_lat = (ds['latitude'].values.item() if 'latitude' in ds
                      else ds['lat'].values.item())
            st_lon = (ds['longitude'].values.item() if 'longitude' in ds
                      else ds['lon'].values.item())
            try:
                station_depth = ds_bathy['deptho'].sel(
                    latitude=st_lat, longitude=st_lon,
                    method='nearest').values.item()
            except KeyError:
                station_depth = ds_bathy['deptho'].sel(
                    lat=st_lat, lon=st_lon, method='nearest').values.item()
            df_nc = (ds.drop_vars([v for v in ("lat", "lon", "latitude",
                                               "longitude") if v in ds])
                       .to_dataframe().reset_index()
                       .drop(columns=[c for c in ("lat", "lon", "latitude",
                                                  "longitude")
                                      if c in ds.coords], errors='ignore'))

        df_nc["time"] = pd.to_datetime(df_nc["time"], errors="coerce")
        if df_nc["time"].dt.tz is not None:
            df_nc["time"] = df_nc["time"].dt.tz_localize(None)
        df_nc = (df_nc.dropna(subset=["time"]).sort_values("time")
                      .drop_duplicates("time"))

        df_merged = pd.merge(df_xls, df_nc, left_on="Date and Time",
                             right_on="time", how="inner").drop(columns=["time"])
        df_merged['Season'] = df_merged['Date and Time'].dt.month.apply(get_season)
        df_merged['Depth_CMEMS'] = station_depth

        # Physical validation stats (all four pairs stored [REV-R1-C1])
        current = {'tag': tag, 'n_samples': len(df_merged),
                   'hm0_corr': 0.0, 'hm0_bias': 100.0}
        for adcp_col, cmems_col in physics_pairs:
            if adcp_col in df_merged.columns and cmems_col in df_merged.columns:
                valid = df_merged[[adcp_col, cmems_col]].dropna()
                if not valid.empty:
                    corr = valid[adcp_col].corr(valid[cmems_col])
                    bias = (valid[cmems_col] - valid[adcp_col]).mean()
                    current[f"{adcp_col}_corr"] = corr
                    current[f"{adcp_col}_bias"] = bias
                    if adcp_col == "Hm0":
                        current['hm0_corr'] = corr
                        current['hm0_bias'] = bias
        best_match_stats.append(current)

        tagged = df_merged.copy()
        tagged.insert(0, "Location_Tag", tag)
        all_merged_frames[tag] = tagged

        # Correlation-matrix figure per station (unchanged style)
        numeric_df = df_merged.select_dtypes(include=[np.number])
        numeric_df = numeric_df.drop(columns=[c for c in ["Measurement No"]
                                              if c in numeric_df.columns],
                                     errors='ignore')
        numeric_df = numeric_df.loc[:, numeric_df.apply(pd.Series.nunique) > 1]
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            numeric_df.columns = [c.replace("_", " ") for c in numeric_df.columns]
            fig = plt.figure(figsize=(20, 18))
            corr_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            hm = sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                             vmax=1, vmin=-1, center=0, square=True,
                             linewidths=0.5, linecolor='black',
                             cbar_kws={"shrink": .75, "pad": 0.02})
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=26, width=2)
            cbar.set_label("Correlation Coefficient", fontsize=32,
                           fontweight='bold', labelpad=20)
            cbar.set_ticks(np.arange(-1, 1.1, 0.2))
            plt.xticks(rotation=45, ha='right', fontsize=26, fontweight='bold')
            plt.yticks(rotation=0, fontsize=26, fontweight='bold')
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / f"{tag}_correlation.png", dpi=HIGH_DPI,
                        bbox_inches='tight')
            plt.close(fig)

        df_merged.to_excel(OUTPUT_DIR / f"{tag}_merged.xlsx", index=False,
                           engine="openpyxl")

    stats_df = pd.DataFrame(best_match_stats)
    stats_df.to_csv(OUTPUT_DIR / "Station_Fidelity_Statistics.csv", index=False)

    # ---------------- Fidelity gate at the manuscript thresholds -------------
    passed, report_data = [], []
    for rec in best_match_stats:
        ok = (rec['hm0_corr'] >= MIN_CORR_THRESHOLD
              and abs(rec['hm0_bias']) <= MAX_BIAS_THRESHOLD)
        print(f"[S3]   {rec['tag']:<28} Corr {rec['hm0_corr']:.3f} "
              f"Bias {rec['hm0_bias']:.3f} -> {'PASS' if ok else 'REJECT'}")
        if ok:
            passed.append(all_merged_frames[rec['tag']])
            report_data.append(rec)

    if passed:
        global_df = pd.concat(passed, ignore_index=True)
        global_df = global_df.sort_values(by=['Location_Tag', 'Date and Time'])
        global_df.to_excel(OUTPUT_DIR / "Global_Best_Matches_Merged.xlsx",
                           index=False, engine="openpyxl")
        rep = ["Global Best Matches Report", "=" * 26,
               f"Criteria: Corr >= {MIN_CORR_THRESHOLD}, "
               f"Abs(Bias) <= {MAX_BIAS_THRESHOLD}", "",
               f"{'Location Tag':<30} | {'Hm0 Corr':<10} | {'Hm0 Bias':<10}",
               "-" * 55]
        for r in report_data:
            rep.append(f"{r['tag']:<30} | {r['hm0_corr']:<10.4f} | "
                       f"{r['hm0_bias']:<10.4f}")
        save_text(OUTPUT_DIR, "Global_Best_Matches_Report.txt", "\n".join(rep))
    else:
        print("[S3] WARNING: no station passed the fidelity gate.")

    # Unfiltered export (needed by the ablation study — unchanged behaviour)
    all_dfs = list(all_merged_frames.values())
    if all_dfs:
        unf = pd.concat(all_dfs, ignore_index=True)
        unf = unf.sort_values(by=['Location_Tag', 'Date and Time'])
        unf.to_excel(OUTPUT_DIR / "Global_All_Data_Unfiltered.xlsx",
                     index=False, engine="openpyxl")

    # -------------------------------------------------------------------------
    # [REV-R1-C1]  FIDELITY-GATE THRESHOLD SENSITIVITY ANALYSIS
    #   For every (corr, bias) threshold combination on a grid we report:
    #     - number of station-segments passing the gate
    #     - number of 3-hourly samples retained
    #     - fraction of the total dataset retained
    #   Downstream classification sensitivity is completed in Stage 6
    #   (function gate_downstream_sensitivity) using these tables.
    # -------------------------------------------------------------------------
    sens_dir = rev_dir("R1C1_FidelityGate_Sensitivity")
    rows = []
    total_samples = int(stats_df['n_samples'].sum())
    for c_thr in GATE_CORR_GRID:
        for b_thr in GATE_BIAS_GRID:
            mask = ((stats_df['hm0_corr'] >= c_thr)
                    & (stats_df['hm0_bias'].abs() <= b_thr))
            rows.append({
                'Corr_Threshold': c_thr,
                'Bias_Threshold_m': b_thr,
                'Stations_Passing': int(mask.sum()),
                'Samples_Retained': int(stats_df.loc[mask, 'n_samples'].sum()),
                'Fraction_Data_Retained':
                    stats_df.loc[mask, 'n_samples'].sum() / max(total_samples, 1),
                'Is_Manuscript_Setting':
                    (c_thr == MIN_CORR_THRESHOLD and b_thr == MAX_BIAS_THRESHOLD)
            })
    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(sens_dir / "Gate_Sensitivity_DataRetention.csv", index=False)

    # Heatmaps: stations passing & data retained
    for metric, label, fname in [
            ('Stations_Passing', 'Station segments passing the gate',
             'Gate_Sensitivity_Stations_Heatmap'),
            ('Fraction_Data_Retained', 'Fraction of samples retained',
             'Gate_Sensitivity_DataFraction_Heatmap')]:
        piv = sens_df.pivot(index='Bias_Threshold_m', columns='Corr_Threshold',
                            values=metric)
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(piv, annot=True,
                    fmt='.0f' if metric == 'Stations_Passing' else '.2f',
                    cmap='viridis', linewidths=1.0, linecolor='black',
                    annot_kws={'size': 22, 'weight': 'bold'},
                    cbar_kws={'label': label, 'shrink': 0.85}, ax=ax)
        ax.set_xlabel('Pearson-R threshold (Hm0)', fontweight='bold',
                      fontsize=26, labelpad=12)
        ax.set_ylabel('|Bias| threshold (m)', fontweight='bold',
                      fontsize=26, labelpad=12)
        # mark the manuscript operating point
        try:
            ci = list(piv.columns).index(MIN_CORR_THRESHOLD)
            bi = list(piv.index).index(MAX_BIAS_THRESHOLD)
            ax.add_patch(plt.Rectangle((ci, bi), 1, 1, fill=False,
                                       edgecolor='red', lw=5))
        except ValueError:
            pass
        save_fig(fig, sens_dir, fname)

    txt = ["FIDELITY-GATE THRESHOLD SENSITIVITY - DATA RETENTION  [REV-R1-C1]",
           "=" * 72,
           f"Grid: Corr in {GATE_CORR_GRID}; |Bias| in {GATE_BIAS_GRID} m",
           f"Manuscript operating point: Corr >= {MIN_CORR_THRESHOLD}, "
           f"|Bias| <= {MAX_BIAS_THRESHOLD} m (red box in heatmaps)", "",
           sens_df.to_string(index=False), "",
           "NOTE: downstream classification metrics for each gate setting are",
           "computed in Stage 6 and saved as Gate_Sensitivity_Downstream.csv."]
    save_text(sens_dir, "Gate_Sensitivity_DataRetention_Report.txt",
              "\n".join(txt))
    print("[S3] Fidelity-gate sensitivity (data retention) done.")

# =============================================================================
# SECTION 4 — STAGE 4: FEATURE ENGINEERING + STRATIFIED BALANCING (unchanged)
# =============================================================================
MINIMUM_GROUP_SIZE_THRESHOLD = 240
STATION_COLUMN = 'Location_Tag'
DATETIME_COLUMN = 'Date and Time'

def get_base_station_name(station_name) -> str:
    if not isinstance(station_name, str):
        return 'Unknown'
    return re.sub(r'(_segment)?_\d+$', '', station_name,
                  flags=re.IGNORECASE).strip()

def get_season_label(date) -> str:
    m = date.month
    if m in [3, 4, 5]: return 'Spring'
    if m in [6, 7, 8]: return 'Summer'
    if m in [9, 10, 11]: return 'Autumn'
    return 'Winter'

def add_rolling_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """24-h (8x3-h) rolling engineering features (unchanged formulas)."""
    df = df.sort_values(by=[group_col, DATETIME_COLUMN])
    df['Wave_Power_Kw'] = 0.5 * (df['Hm0'] ** 2) * (0.9 * df['Tp'])
    g = df.groupby(group_col)
    df['Power_Fluctuation'] = g['Wave_Power_Kw'].transform(
        lambda x: x.rolling(8, min_periods=1).std()).fillna(0)
    df['Daily_Max_Hmax'] = g['Hmax'].transform(
        lambda x: x.rolling(8, min_periods=1).max())
    df['Daily_Max_Hmax'] = df['Daily_Max_Hmax'].fillna(df['Hmax'])
    df['Period_Fluctuation'] = g['Tp'].transform(
        lambda x: x.rolling(8, min_periods=1).std()).fillna(0)
    for src, name in [('VHM0', 'Model_Height_Stability'),
                      ('VTM02', 'Model_Period_Stability'),
                      ('VHM0_SW1', 'Swell_Stability'),
                      ('VMDR', 'Direction_Stability')]:
        if src in df.columns:
            df[name] = g[src].transform(
                lambda x: x.rolling(8, min_periods=1).std()).fillna(0)
    return df

def stage4_features():
    in_path = pathlib.Path(CONFIG["MERGE_OUT"]) / "Global_Best_Matches_Merged.xlsx"
    out_dir = pathlib.Path(CONFIG["FEATURE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "Visualizations"
    plot_dir.mkdir(exist_ok=True)

    df = pd.read_excel(in_path, engine='openpyxl')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce')
    n0 = len(df)
    df.dropna(subset=[DATETIME_COLUMN, STATION_COLUMN, 'Hm0', 'Tp', 'Hmax'],
              inplace=True)
    print(f"[S4] Loaded {n0} rows -> {len(df)} after NaN cleaning.")

    df['Base_Station'] = df[STATION_COLUMN].apply(get_base_station_name)
    df = add_rolling_features(df, 'Base_Station')

    df['Season_Name'] = df[DATETIME_COLUMN].apply(get_season_label)
    strat_key = 'Stratify_Key'
    df[strat_key] = df['Base_Station'] + '_' + df['Season_Name']
    counts = df[strat_key].value_counts()
    valid_groups = counts[counts >= MINIMUM_GROUP_SIZE_THRESHOLD].index.tolist()
    if not valid_groups:
        raise ValueError("[S4] No groups met the minimum size threshold!")
    df_filtered = df[df[strat_key].isin(valid_groups)].copy()
    min_size = df_filtered[strat_key].value_counts().min()
    balanced = [g.sample(n=min_size, random_state=GLOBAL_SEED)
                for _, g in df_filtered.groupby(strat_key)]
    df_balanced = pd.concat(balanced).sort_values(
        by=['Base_Station', DATETIME_COLUMN])
    final_output = df_balanced.drop(columns=[strat_key])
    out_csv = out_dir / 'Final_Balanced_Data_with_Features_Rolling.csv'
    final_output.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[S4] Balanced dataset saved ({len(final_output)} rows) -> {out_csv}")

    # data-provenance figures (unchanged)
    sns.set_theme(style="whitegrid")
    apply_publication_style()   # [REV] seaborn resets fonts; restore Times
    fig = plt.figure(figsize=(12, 8))
    sns.scatterplot(data=final_output, x=DATETIME_COLUMN, y='Base_Station',
                    hue='Season_Name', palette='viridis', s=10, edgecolor=None)
    plt.title('Temporal Distribution of Final Balanced Data',
              fontsize=16, fontweight='bold')
    plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(plot_dir / 'Temporal_Distribution.png', dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    count_data = (final_output.groupby(['Base_Station', 'Season_Name'])
                  .size().reset_index(name='Count'))
    sns.barplot(data=count_data, x='Base_Station', y='Count',
                hue='Season_Name', palette='coolwarm')
    plt.title('Balanced Data Count per Location & Season',
              fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(plot_dir / 'Seasonal_Data_Balance.png', dpi=300)
    plt.close(fig)
    apply_publication_style()   # [REV] full restore, not just labelweight

# =============================================================================
# SECTION 5 — STAGE 5: CLUSTERING (K-Means / GMM / Deep AE+GMM)
#             + [REV-R1-C4] multi-seed stability + method-comparison table
#             + [REV-R3-C1] full autoencoder architecture/training logging
# =============================================================================
def build_autoencoder(input_dim: int, encoding_dim: int = 4,
                      learning_rate: float = 1e-3):
    """Symmetric 16-8-4-8-16 autoencoder (as reported in the manuscript).
    Glorot-uniform initialization (Keras Dense default), ReLU activations,
    linear output, MSE loss, Adam optimizer.  [REV-R3-C1]"""
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu')(input_layer)
    x = layers.Dense(8, activation='relu')(x)
    latent = layers.Dense(encoding_dim, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(latent)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(input_layer, out)
    enc = models.Model(input_layer, latent)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
               loss='mse')
    return ae, enc

def train_deep_clustering(data_normalized: np.ndarray, seed: int = GLOBAL_SEED,
                          epochs: int = 100, batch_size: int = 32,
                          patience: int = 15, verbose: int = 0):
    """Train AE, fit GMM on latent space; returns labels, encoder, gmm and a
    complete training-details dictionary for reviewer reporting [REV-R3-C1]."""
    set_global_seeds(seed)
    ae, enc = build_autoencoder(data_normalized.shape[1], encoding_dim=4,
                                learning_rate=1e-3)
    es = EarlyStopping(monitor='val_loss', patience=patience,
                       restore_best_weights=True)
    t0 = time.time()
    hist = ae.fit(data_normalized, data_normalized, epochs=epochs,
                  batch_size=batch_size, shuffle=True, validation_split=0.2,
                  callbacks=[es], verbose=verbose)
    train_time = time.time() - t0
    encoded = enc.predict(data_normalized, verbose=0)
    gmm_deep = GaussianMixture(n_components=N_CLUSTERS, random_state=seed,
                               covariance_type='full')
    labels = gmm_deep.fit_predict(encoded)
    # NOTE: raw GMM cluster indices are arbitrary. The mapping from cluster
    # index -> physical regime (Storm/Swell/Calm/Confused) is now assigned
    # DATA-DRIVEN in map_clusters_to_regimes() from the cluster physical
    # profiles, replacing the old hardcoded 1<->3 swap which silently
    # inverted the semantics whenever GMM initialization changed.

    details = {                                              # [REV-R3-C1]
        "architecture": "Input(7) -> Dense16(ReLU) -> Dense8(ReLU) -> "
                        "Latent4(ReLU) -> Dense8(ReLU) -> Dense16(ReLU) -> "
                        "Output7(linear)",
        "latent_dimension": 4,
        "activation_hidden": "ReLU",
        "activation_output": "linear",
        "loss": "MSE (reconstruction)",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "epochs_actually_trained": len(hist.history['loss']),
        "early_stopping": f"val_loss, patience={patience}, "
                          "restore_best_weights=True",
        "validation_split": 0.2,
        "weight_initialization": "Glorot uniform (Keras Dense default)",
        "bias_initialization": "zeros (Keras Dense default)",
        "random_seed": seed,
        "final_train_loss": float(hist.history['loss'][-1]),
        "final_val_loss": float(hist.history['val_loss'][-1]),
        "best_val_loss": float(np.min(hist.history['val_loss'])),
        "training_time_s": round(train_time, 2),
        "gmm_on_latent": {"n_components": N_CLUSTERS,
                          "covariance_type": "full", "random_state": seed},
        "cluster_index_convention": "raw GMM indices; canonical regime "
                                    "mapping assigned data-driven downstream",
    }
    return labels, enc, gmm_deep, details, hist

def plot_tsne_pair(data, labels, station_series, title, output_dir, method):
    """Combined cluster-view + station-view t-SNE panel (unchanged style)."""
    tsne = TSNE(n_components=2, random_state=GLOBAL_SEED, perplexity=30,
                max_iter=1000)
    emb = tsne.fit_transform(data)
    labels_num = np.asarray(labels).astype(int)
    # [REV-FIGFIX] colours bound to physical regime, not to cluster index
    point_colors = [regime_color(l) for l in labels_num]
    stations = station_series.astype('category')
    codes = stations.cat.codes
    names = [str(c).replace('_', ' ') for c in stations.cat.categories]
    cmap = matplotlib.colormaps['tab10' if len(names) <= 10 else 'tab20'].resampled(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16))
    ax1.scatter(emb[:, 0], emb[:, 1], c=point_colors, s=250, alpha=0.7,
                edgecolors='k', linewidth=1.5)
    ax1.set_title(f't-SNE Projection: {title}', fontweight='bold',
                  fontsize=52, y=1.02)
    ax1.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=42)
    ax1.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=42)
    ax1.tick_params(axis='both', labelsize=42)
    ax1.grid(True, linestyle='--', alpha=0.4)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=regime_color(i), markersize=24,
                          markeredgecolor='k',
                          label=REGIME_NAMES_CANON.get(int(i),
                                                       f'Cluster {i}'))
               for i in np.unique(labels_num)]
    ax1.legend(handles=handles, title='Regime', loc='best', fontsize=32,
               title_fontsize=36, frameon=True, edgecolor='k')
    sc = ax2.scatter(emb[:, 0], emb[:, 1], c=codes, cmap=cmap, s=250,
                     alpha=0.7, edgecolors='k', linewidth=1.5)
    ax2.set_title(f't-SNE Projection (station view): {title}',
                  fontweight='bold', fontsize=52, y=1.02)
    ax2.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=46)
    ax2.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=46)
    ax2.tick_params(axis='both', labelsize=42)
    ax2.grid(True, linestyle='--', alpha=0.4)
    cbar = fig.colorbar(sc, ax=ax2, ticks=np.arange(len(names)))
    cbar.set_label('Station', fontweight='bold', fontsize=48, labelpad=25)
    cbar.ax.set_yticklabels(names, fontsize=40)
    plt.subplots_adjust(wspace=0.35)
    fig.tight_layout(pad=3.0)
    save_fig(fig, output_dir, f'Combined_tSNE_{method}')
    return emb

def analyze_clusters(df, cluster_col, output_dir, data_for_metrics,
                     do_tsne=False):
    """Statistics, box plots, spider+heatmap panel (condensed original)."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = FEATURES_FOR_CLUSTERING
    labels = df[cluster_col]
    cluster_freq = labels.value_counts().sort_index()
    cluster_means = df.groupby(cluster_col)[features].mean()
    sil = silhouette_score(data_for_metrics, labels)
    ch = calinski_harabasz_score(data_for_metrics, labels)
    db = davies_bouldin_score(data_for_metrics, labels)

    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Index',
                   'Davies-Bouldin Index'],
        'Score': [sil, ch, db],
        'Interpretation': ['Higher is better', 'Higher is better',
                           'Lower is better']}).set_index('Metric')
    with pd.ExcelWriter(output_dir / 'cluster_statistics.xlsx') as writer:
        cluster_freq.to_frame('Frequency').to_excel(writer,
                                                    sheet_name='Cluster Frequency')
        cluster_means.to_excel(writer, sheet_name='Cluster Means')
        metrics_df.to_excel(writer, sheet_name='Evaluation Metrics')

    for feature in features:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x=cluster_col, y=feature, data=df, linewidth=3,
                    fliersize=6, medianprops={'linewidth': 3, 'color': 'red'})
        ax.set_ylabel(f'{feature} {FEATURE_UNITS.get(feature, "")}',
                      fontweight='bold', fontsize=26)
        ax.set_xlabel('Regime (Cluster)', fontweight='bold', fontsize=26)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        save_fig(fig, output_dir, f'boxplot_{feature}')

    # spider + heatmap panel
    scaler_means = StandardScaler()
    cm_norm = scaler_means.fit_transform(cluster_means)
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    crosstab = pd.crosstab(df['Base_Station'], df[cluster_col])
    crosstab.index = crosstab.index.str.replace('_', ' ')
    fig = plt.figure(figsize=(40, 18))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.55)
    ax1 = plt.subplot(gs[0], projection='polar')
    for i in range(df[cluster_col].nunique()):
        vals = cm_norm[i].tolist(); vals += vals[:1]
        ax1.plot(angles, vals, label=regime_label(i), linewidth=5.0,
                 color=regime_color(i))
        ax1.fill(angles, vals, alpha=0.18, color=regime_color(i))
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(features, fontsize=34, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=28)
    # [REV-FIGFIX] single title offset only (pad AND y previously stacked,
    # pushing the title off-canvas), and legend placed BELOW the polar axes
    # so its long regime names cannot escape the figure edge.
    ax1.set_title('Normalized Regime Profiles', fontsize=52,
                  fontweight='bold', pad=55)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2,
               fontsize=32, frameon=True, edgecolor='black')
    ax2 = plt.subplot(gs[1])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='viridis',
                linewidths=2.5, annot_kws={'size': 40, 'weight': 'bold'},
                cbar=True, ax=ax2, cbar_kws={'shrink': 0.85})
    ax2.set_title('Regime Distribution by Station', fontweight='bold',
                  fontsize=52, pad=25)
    ax2.set_xlabel('Regime (Cluster)', fontweight='bold', fontsize=44)
    ax2.set_ylabel('Station', fontweight='bold', fontsize=44)
    ax2.tick_params(axis='y', rotation=30, labelsize=38)
    ax2.tick_params(axis='x', labelsize=38)
    # [REV-FIGFIX] explicit margins instead of tight_layout(), which cannot
    # handle a polar axes plus an out-of-axes legend and silently clipped
    # the top of the figure.
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.04, right=0.97)
    save_fig(fig, output_dir, f'Combined_Spider_Heatmap_{cluster_col}')

    if do_tsne:
        plot_tsne_pair(data_for_metrics, labels, df['Base_Station'],
                       cluster_col, output_dir, cluster_col)

    df.to_csv(output_dir / 'full_clustered_data.csv', index=False,
              encoding='utf-8-sig')
    return {'Silhouette': sil, 'Calinski_Harabasz': ch, 'Davies_Bouldin': db,
            'Frequencies': cluster_freq.to_dict()}

def map_clusters_to_regimes(df_phys: pd.DataFrame, raw_labels: np.ndarray):
    """[REV] Assign physical regime semantics to arbitrary cluster indices
    from the per-cluster physical profiles (deterministic, seed-independent):
      Storm    = cluster with highest mean Wave_Power_Kw
      Confused = of the remaining, highest mean 'Angle Fluctuation Std'
      Swell    = of the remaining two, highest mean Tp
      Calm     = the last remaining cluster
    Canonical output indices follow the manuscript convention:
      1 = Storm, 3 = Swell, 2 = Calm, 0 = Confused.
    Returns (mapped_labels, mapping_dict, profile_table)."""
    prof = (df_phys.assign(_c=raw_labels)
            .groupby('_c')[['Hm0', 'Tp', 'Wave_Power_Kw',
                            'Angle Fluctuation Std']].mean())
    remaining = list(prof.index)
    storm = prof.loc[remaining, 'Wave_Power_Kw'].idxmax()
    remaining.remove(storm)
    confused = prof.loc[remaining, 'Angle Fluctuation Std'].idxmax()
    remaining.remove(confused)
    swell = prof.loc[remaining, 'Tp'].idxmax()
    remaining.remove(swell)
    calm = remaining[0]
    mapping = {int(storm): 1, int(swell): 3, int(calm): 2, int(confused): 0}
    mapped = np.vectorize(mapping.get)(raw_labels)
    prof = prof.copy()
    prof['Assigned_Regime'] = [
        {1: 'Storm', 3: 'Swell', 2: 'Calm', 0: 'Confused'}[mapping[int(c)]]
        for c in prof.index]
    prof['Canonical_Index'] = [mapping[int(c)] for c in prof.index]
    return mapped, mapping, prof.reset_index().rename(
        columns={'_c': 'Raw_GMM_Cluster'})

def compare_kmeans_deep_tsne(data, labels_kmeans, labels_deep, output_dir):
    """[REV-RESTORED] Side-by-side t-SNE of K-Means vs Deep Clustering on a
    SHARED embedding — this is manuscript Figure 4. It was present in the
    original per-script code but was inadvertently dropped when the scripts
    were unified, which is why the figure was missing from the output tree.
    Colours are now bound to physical regime meaning (REGIME_COLORS) rather
    than to raw cluster index, so the two panels are directly comparable and
    the scheme cannot permute between runs."""
    print("[S5] Generating comparison figure: K-Means vs Deep Clustering ...")
    tsne = TSNE(n_components=2, random_state=GLOBAL_SEED, perplexity=30,
                max_iter=1000)
    emb = tsne.fit_transform(data)
    lk = np.asarray(labels_kmeans).astype(int)
    ld = np.asarray(labels_deep).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16))
    for ax, lab, ttl in ((ax1, lk, 't-SNE Projection: K-Means Regimes'),
                         (ax2, ld, 't-SNE Projection: Deep Clustering '
                                   '(Autoencoder + GMM)')):
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=[regime_color(v) for v in lab],
                   s=250, alpha=0.7, edgecolors='k', linewidth=1.5)
        ax.set_title(ttl, fontweight='bold', fontsize=52, y=1.02)
        ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=48,
                      labelpad=15)
        ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=48,
                      labelpad=15)
        ax.tick_params(axis='both', labelsize=42)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.5)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=regime_color(i), markersize=24,
                              markeredgecolor='k',
                              label=REGIME_NAMES_CANON.get(int(i),
                                                           f'Cluster {i}'))
                   for i in np.unique(lab)]
        ax.legend(handles=handles, title='Regime', loc='best', fontsize=32,
                  title_fontsize=34, frameon=True, edgecolor='k')
    fig.subplots_adjust(wspace=0.28, top=0.90, bottom=0.10,
                        left=0.06, right=0.98)
    save_fig(fig, output_dir, 'Comparison_KMeans_vs_Deep')

def stage5_clustering():
    in_csv = (pathlib.Path(CONFIG["FEATURE_OUT"])
              / 'Final_Balanced_Data_with_Features_Rolling.csv')
    unfiltered_path = (pathlib.Path(CONFIG["MERGE_OUT"])
                       / "Global_All_Data_Unfiltered.xlsx")
    out_dir = pathlib.Path(CONFIG["CLUSTER_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    missing = [f for f in FEATURES_FOR_CLUSTERING if f not in df.columns]
    if missing:
        print(f"[S5] Available columns in {in_csv.name}:")
        print("[S5]   " + ", ".join(df.columns))
        raise SystemExit(
            f"[S5] Missing clustering feature(s): {missing}\n"
            "[S5] HINT: these are created in Stages 2-4. Re-run the "
            "pipeline from Stage 2 (S2/S3/S4 = True) so the ADCP files are "
            "re-processed with the corrected direction-column handling "
            "('MeanDir' -> 'Angle Fluctuation Std' / 'Mean Wave "
            "Direction').")
    df_clean = df.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
    df_clean.reset_index(drop=True, inplace=True)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df_clean[FEATURES_FOR_CLUSTERING])
    print(f"[S5] Data ready for clustering: {data_normalized.shape}")

    # ------------------ BIC / silhouette k-selection (unchanged) -------------
    opt_dir = out_dir / 'Optimization_Analysis'
    opt_dir.mkdir(exist_ok=True)
    k_range = range(2, 9)
    bic_scores, sil_scores = [], []
    for k in k_range:
        g = GaussianMixture(n_components=k, random_state=GLOBAL_SEED,
                            covariance_type='full')
        labs = g.fit_predict(data_normalized)
        bic_scores.append(g.bic(data_normalized))
        sil_scores.append(silhouette_score(data_normalized, labs))
    # numeric export of the k-selection curves (source of the BIC figure);
    # includes the relative BIC reduction quoted in the manuscript
    ksel = pd.DataFrame({'k': list(k_range), 'BIC': bic_scores,
                         'Silhouette': sil_scores})
    ksel['BIC_Reduction_vs_k2_%'] = ((ksel['BIC'].iloc[0] - ksel['BIC'])
                                     / abs(ksel['BIC'].iloc[0]) * 100)
    ksel['BIC_Reduction_vs_prev_%'] = (-ksel['BIC'].diff()
                                       / ksel['BIC'].shift().abs() * 100)
    ksel.to_csv(opt_dir / 'K_Selection_BIC_Silhouette_Scores.csv',
                index=False)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(list(k_range), bic_scores, marker='o', linewidth=5, markersize=18,
            color='#2b83ba', markerfacecolor='white', markeredgewidth=4,
            label='BIC (lower is better)')
    ax.axvline(x=4, color='#d7191c', linestyle='--', linewidth=4,
               label='Selected $k=4$')
    ax.set_xlabel('Number of clusters ($k$)', fontweight='bold', fontsize=42)
    ax.set_ylabel('BIC Score', fontweight='bold', fontsize=42)
    ax.legend(fontsize=28, frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.6)
    save_fig(fig, opt_dir, 'BIC_Elbow_Plot')
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(list(k_range), sil_scores, marker='s', linewidth=5, markersize=18,
            color='#4dac26', markerfacecolor='white', markeredgewidth=4,
            label='Silhouette (higher is better)')
    ax.set_xlabel('Number of clusters ($k$)', fontweight='bold', fontsize=42)
    ax.set_ylabel('Silhouette score', fontweight='bold', fontsize=42)
    ax.legend(fontsize=28, frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.6)
    save_fig(fig, opt_dir, 'Silhouette_Plot')

    # ------------------ the three clustering methods (unchanged) -------------
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=GLOBAL_SEED, n_init=10)
    kmeans_labels = kmeans.fit_predict(data_normalized)
    df_k = df_clean.copy(); df_k['KMeans_Regime'] = kmeans_labels
    m_k = analyze_clusters(df_k, 'KMeans_Regime',
                           out_dir / 'Method_0_Baseline_KMeans',
                           data_normalized, do_tsne=True)

    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=GLOBAL_SEED,
                          covariance_type='full')
    gmm_labels = gmm.fit_predict(data_normalized)
    df_g = df_clean.copy(); df_g['GMM_Regime'] = gmm_labels
    m_g = analyze_clusters(df_g, 'GMM_Regime', out_dir / 'Method_1_GMM',
                           data_normalized, do_tsne=False)

    if not TF_AVAILABLE:
        raise SystemExit("[S5] TensorFlow is required for deep clustering.")
    deep_labels_raw, encoder_model, gmm_deep, ae_details, hist = \
        train_deep_clustering(data_normalized, seed=GLOBAL_SEED)
    # [REV] data-driven cluster -> physical-regime assignment (replaces the
    # old hardcoded 1<->3 swap; robust to arbitrary GMM initialization)
    deep_labels, regime_mapping, regime_profile = \
        map_clusters_to_regimes(df_clean, deep_labels_raw)
    ae_details["cluster_index_convention"] = (
        "raw GMM clusters mapped to canonical regimes DATA-DRIVEN from the "
        "cluster physical profiles: Storm=max Wave_Power_Kw -> index 1, "
        "then Confused=max Angle Fluctuation Std -> 0, then Swell=max Tp "
        f"-> 3, Calm -> 2. Mapping this run: {regime_mapping}")
    regime_profile.to_csv(out_dir / 'Cluster_To_Regime_Mapping.csv',
                          index=False)
    save_text(rev_dir("R1C4_Cluster_Stability"),
              "Cluster_Physical_Interpretation.txt",
              "DATA-DRIVEN CLUSTER -> REGIME ASSIGNMENT  [REV-R1-C4]\n"
              + "=" * 64 + "\n"
              "Deterministic rules on the cluster physical profiles:\n"
              "  Storm    = highest mean Wave_Power_Kw          -> index 1\n"
              "  Confused = next, highest Angle Fluctuation Std -> index 0\n"
              "  Swell    = next, highest mean Tp               -> index 3\n"
              "  Calm     = remaining cluster                   -> index 2\n\n"
              f"Raw-GMM -> canonical mapping this run: {regime_mapping}\n\n"
              + regime_profile.to_string(index=False))
    print(f"[S5] Regime mapping (raw GMM -> canonical): {regime_mapping}")
    encoded_features = encoder_model.predict(data_normalized, verbose=0)
    df_d = df_clean.copy(); df_d['Deep_Regime'] = deep_labels
    m_d = analyze_clusters(df_d, 'Deep_Regime',
                           out_dir / 'Method_2_Deep_Clustering',
                           encoded_features, do_tsne=True)

    # [REV-RESTORED] manuscript Figure 4: K-Means vs Deep on one embedding.
    # Saved at the TOP level of the clustering folder so the path matches
    # the figure list given to the authors.
    compare_kmeans_deep_tsne(data_normalized, kmeans_labels, deep_labels,
                             out_dir)

    # ---------------------------------------------------------------------
    # [REV-R3-C1] save the complete autoencoder architecture/training log
    # ---------------------------------------------------------------------
    arch_dir = rev_dir("R3C1_Model_Architecture_Hyperparams")
    save_text(arch_dir, "Autoencoder_Training_Details.txt",
              "DEEP CLUSTERING - COMPLETE AUTOENCODER DETAILS  [REV-R3-C1]\n"
              + "=" * 66 + "\n"
              + json.dumps(ae_details, indent=2))
    with open(arch_dir / "Autoencoder_Training_Details.json", "w") as f:
        json.dump(ae_details, f, indent=2)
    # training curve figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(hist.history['loss'], label='Training loss', linewidth=3)
    ax.plot(hist.history['val_loss'], label='Validation loss', linewidth=3)
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=26)
    ax.set_ylabel('MSE reconstruction loss', fontweight='bold', fontsize=26)
    ax.legend(fontsize=22, frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.4)
    save_fig(fig, arch_dir, "Autoencoder_Loss_Curve")

    # ---------------------------------------------------------------------
    # [REV-R1-C4] (a) QUANTITATIVE COMPARISON TABLE of the three methods
    # ---------------------------------------------------------------------
    stab_dir = rev_dir("R1C4_Cluster_Stability")
    comp = pd.DataFrame([
        {'Method': 'K-Means (baseline)', 'Silhouette': m_k['Silhouette'],
         'Calinski_Harabasz': m_k['Calinski_Harabasz'],
         'Davies_Bouldin': m_k['Davies_Bouldin'],
         'Metric_Space': 'Standardized input features'},
        {'Method': 'GMM', 'Silhouette': m_g['Silhouette'],
         'Calinski_Harabasz': m_g['Calinski_Harabasz'],
         'Davies_Bouldin': m_g['Davies_Bouldin'],
         'Metric_Space': 'Standardized input features'},
        {'Method': 'Deep Clustering (AE + GMM)', 'Silhouette': m_d['Silhouette'],
         'Calinski_Harabasz': m_d['Calinski_Harabasz'],
         'Davies_Bouldin': m_d['Davies_Bouldin'],
         'Metric_Space': '4-D learned latent space'},
    ])
    comp.to_csv(stab_dir / "Clustering_Methods_Quantitative_Comparison.csv",
                index=False)

    # ---------------------------------------------------------------------
    # [REV-R1-C4] (b) CLUSTER CONSISTENCY ACROSS RANDOM SEEDS
    #   For each method we re-fit with CLUSTER_STABILITY_SEEDS and compute
    #   pairwise Adjusted Rand Index (ARI) and Adjusted Mutual Info (AMI)
    #   between every pair of seed runs. High mean ARI/AMI => stable regimes.
    # ---------------------------------------------------------------------
    print("[S5] [REV-R1-C4] Cluster-stability analysis across seeds ...")
    seed_labels = {'KMeans': [], 'GMM': [], 'DeepClustering': []}
    for s in CLUSTER_STABILITY_SEEDS:
        km = KMeans(n_clusters=N_CLUSTERS, random_state=s, n_init=10)
        seed_labels['KMeans'].append(km.fit_predict(data_normalized))
        gm = GaussianMixture(n_components=N_CLUSTERS, random_state=s,
                             covariance_type='full')
        seed_labels['GMM'].append(gm.fit_predict(data_normalized))
        d_labels, _, _, _, _ = train_deep_clustering(
            data_normalized, seed=s, epochs=100, batch_size=32, verbose=0)
        seed_labels['DeepClustering'].append(d_labels)
        print(f"      seed {s} done")
    set_global_seeds(GLOBAL_SEED)  # restore

    stab_rows, pair_rows = [], []
    for method, runs in seed_labels.items():
        aris, amis = [], []
        for (i, a), (j, b) in combinations(enumerate(runs), 2):
            ari = adjusted_rand_score(a, b)
            ami = adjusted_mutual_info_score(a, b)
            aris.append(ari); amis.append(ami)
            pair_rows.append({'Method': method, 'Seed_A':
                              CLUSTER_STABILITY_SEEDS[i],
                              'Seed_B': CLUSTER_STABILITY_SEEDS[j],
                              'ARI': ari, 'AMI': ami})
        stab_rows.append({'Method': method,
                          'N_Seeds': len(CLUSTER_STABILITY_SEEDS),
                          'Mean_ARI': np.mean(aris), 'Std_ARI': np.std(aris),
                          'Min_ARI': np.min(aris),
                          'Mean_AMI': np.mean(amis), 'Std_AMI': np.std(amis),
                          'Min_AMI': np.min(amis)})
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(stab_dir / "Cluster_Stability_Across_Seeds_Summary.csv",
                   index=False)
    pd.DataFrame(pair_rows).to_csv(
        stab_dir / "Cluster_Stability_Pairwise_ARI_AMI.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(len(stab_df))
    # [REV-FIGFIX] ARI and AMI are bounded above by 1, so a symmetric
    # +/- std whisker can overshoot the axis and get visually clipped
    # (K-Means: 0.896 + 0.204 = 1.10). The upper whisker is therefore
    # truncated at the theoretical maximum, which is also the statistically
    # correct depiction for a bounded index.
    def _clipped_err(mean, std):
        mean = np.asarray(mean, float); std = np.asarray(std, float)
        return np.vstack([std, np.minimum(std, 1.0 - mean)])
    ax.bar(x - 0.2, stab_df['Mean_ARI'], 0.4,
           yerr=_clipped_err(stab_df['Mean_ARI'], stab_df['Std_ARI']),
           capsize=8, label='Mean pairwise ARI', edgecolor='black',
           linewidth=1.5, color='#4C72B0')
    ax.bar(x + 0.2, stab_df['Mean_AMI'], 0.4,
           yerr=_clipped_err(stab_df['Mean_AMI'], stab_df['Std_AMI']),
           capsize=8, label='Mean pairwise AMI', edgecolor='black',
           linewidth=1.5, color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(stab_df['Method'], fontsize=22)
    ax.set_ylabel('Agreement across seeds', fontweight='bold', fontsize=26)
    ax.set_ylim(0, 1.18)          # headroom for the legend
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=2)
    ax.legend(fontsize=20, frameon=True, edgecolor='black',
              loc='upper right', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.subplots_adjust(top=0.95, bottom=0.12)
    save_fig(fig, stab_dir, "Cluster_Stability_Bar")

    save_text(stab_dir, "Cluster_Stability_Report.txt",
              "CLUSTER CONSISTENCY ACROSS RANDOM SEEDS  [REV-R1-C4]\n"
              + "=" * 64 + "\n"
              + f"Seeds tested: {CLUSTER_STABILITY_SEEDS}\n"
              + "Metric: pairwise Adjusted Rand Index (ARI) and Adjusted\n"
              + "Mutual Information (AMI) between regime labelings obtained\n"
              + "with different random seeds (identical data & settings).\n"
              + "ARI/AMI = 1 -> identical partitions; 0 -> chance agreement.\n\n"
              + stab_df.to_string(index=False) + "\n\n"
              + "Quantitative comparison with conventional clustering:\n"
              + comp.to_string(index=False))

    # ------------- apply trained deep model to the unfiltered data -----------
    if unfiltered_path.exists():
        df_unf = pd.read_excel(unfiltered_path, engine='openpyxl')
        df_unf['Base_Station'] = df_unf[STATION_COLUMN].apply(
            get_base_station_name)
        df_unf[DATETIME_COLUMN] = pd.to_datetime(df_unf[DATETIME_COLUMN],
                                                 errors='coerce')
        missing = [f for f in FEATURES_FOR_CLUSTERING
                   if f not in df_unf.columns]
        if missing:
            df_unf = add_rolling_features(df_unf, STATION_COLUMN)
            if ('Angle Fluctuation Std' in missing
                    and 'Angle Fluctuation Std' not in df_unf.columns):
                df_unf['Angle Fluctuation Std'] = \
                    df_clean['Angle Fluctuation Std'].median()
        still = [f for f in FEATURES_FOR_CLUSTERING if f not in df_unf.columns]
        if not still:
            unf_clean = df_unf.dropna(subset=FEATURES_FOR_CLUSTERING).copy()
            unf_clean.reset_index(drop=True, inplace=True)
            unf_norm = scaler.transform(unf_clean[FEATURES_FOR_CLUSTERING])
            unf_encoded = encoder_model.predict(unf_norm, verbose=0)
            unf_labels_raw = gmm_deep.predict(unf_encoded)
            # apply the SAME data-driven mapping as the training labels
            unf_clean['Deep_Regime'] = np.vectorize(
                regime_mapping.get)(unf_labels_raw)
            unf_clean.to_csv(out_dir / 'unfiltered_deep_clustered_data.csv',
                             index=False, encoding='utf-8-sig')
            print("[S5] Unfiltered dataset labeled with trained deep model.")
        else:
            print(f"[S5] Unfiltered data missing features {still}; skipped.")
    else:
        print("[S5] Unfiltered file not found; ablation input skipped.")

# =============================================================================
# SECTION 6 — STAGE 6: CLASSIFICATION (label propagation)
#   + [REV-R3-C3] extra ensemble algorithms + per-model tuning details
#   + [REV-R3-C2] contiguous block-wise 10-fold CV (explicit construction)
#   + [REV-R1-C2] Leave-One-Station-Out + Leave-One-Region-Out
#   + [REV-R2-C3] bootstrap CIs, repeated seeds, Gate ON/OFF significance
#   + [REV-R1-C1] downstream gate-threshold sensitivity
#   + [REV-FIG10] reworked model-evolution figure (BalAcc + Kappa)
#   + [REV-TABLES] combined comprehensive tables
# =============================================================================
FEATURES_TO_DROP = [
    'Date and Time', 'Station ID', 'Measurement No', 'Source File', 'Station',
    'Base_Station', 'Distance To Sur', 'Distance From Shore', 'Season',
    'Inside Protected Area', 'Depth GEBCO', 'Depth ETOP1', 'Location_Tag',
    'Season_Name', 'Hm0', 'Tp', 'Hmax', 'Hmean', 'Tm02', 'Mean Wave Direction',
    'Wave_Power_Kw', 'Power_Fluctuation', 'Period_Fluctuation',
    'Angle Fluctuation Std', 'Daily_Max_Hmax', 'Directional Spread',
    'GMM_Regime', 'KMeans_Regime', 'Deep_Cluster', 'Deep_Regime',
    # ADCP-only measured columns that are NOT available on the CMEMS grid
    # at deployment (the original Code-2 aggregation never exported them);
    # they must not enter the classifier so that the trained model matches
    # the 22 CMEMS-deployable features used in Stage 7 and the manuscript.
    'H10', 'H3', 'Tz', 'DirTp', 'SprTp',
]
TARGET_COLUMN = 'Deep_Regime'

def station_region(base_station: str) -> str:
    """Map a base-station name to north/central/south using its latitude."""
    meta = STATIONS.get(base_station.replace(' ', '_'),
                        STATIONS.get(base_station))
    if meta is None:
        for k, v in STATIONS.items():
            if k.lower().replace('_', '') == str(base_station).lower().replace(
                    '_', '').replace(' ', ''):
                meta = v
                break
    if meta is None:
        return 'unknown'
    lat = meta['lat']
    if lat > 22.5:
        return 'north'
    if lat < 18.0:
        return 'south'
    return 'central'

def clf_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Cohen_Kappa': cohen_kappa_score(y_true, y_pred),
        'F1_Macro': f1_score(y_true, y_pred, average='macro'),
        'F1_Weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    if y_proba is not None:
        try:
            out['AUC_Macro_OvR'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            out['AUC_Macro_OvR'] = np.nan
    return out

def build_stacking(weights_dict) -> StackingClassifier:
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200,
                                      random_state=GLOBAL_SEED, n_jobs=-1,
                                      class_weight='balanced')),
        ('xgb', XGBClassifier(eval_metric='mlogloss',
                              random_state=GLOBAL_SEED)),
        ('cat', CatBoostClassifier(verbose=0, random_state=GLOBAL_SEED,
                                   iterations=300, depth=6,
                                   class_weights=weights_dict)),
    ]
    if LGBM_AVAILABLE:
        estimators.append(('lgbm', LGBMClassifier(random_state=GLOBAL_SEED,
                                                  n_jobs=-1, verbose=-1,
                                                  class_weight='balanced')))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000,
                                           class_weight='balanced'),
        cv=3, n_jobs=-1)

def dump_all_hyperparameters(models_dict: dict, stacking_model, out_folder):
    """[REV-R3-C3] Save the COMPLETE hyper-parameter set of every model
    (base learners + tuned models + stacking meta-learner) as JSON + txt."""
    def clean(params):
        return {k: (str(v) if not isinstance(
            v, (int, float, str, bool, type(None))) else v)
                for k, v in params.items()}
    payload = {name: clean(m.get_params(deep=False))
               for name, m in models_dict.items()}
    if stacking_model is not None:
        payload['StackingEnsemble'] = clean(
            stacking_model.get_params(deep=False))
        payload['Stacking_MetaLearner_LogisticRegression'] = clean(
            stacking_model.final_estimator.get_params(deep=False))
        for est_name, est in stacking_model.estimators:
            payload[f'Stacking_Base_{est_name}'] = clean(
                est.get_params(deep=False))
    with open(pathlib.Path(out_folder) /
              "All_Models_Complete_Hyperparameters.json", "w") as f:
        json.dump(payload, f, indent=2)
    lines = ["COMPLETE HYPER-PARAMETERS OF ALL MODELS  [REV-R3-C3]",
             "=" * 66, ""]
    for name, params in payload.items():
        lines.append(f"### {name}")
        for k, v in sorted(params.items()):
            lines.append(f"    {k} = {v}")
        lines.append("")
    save_text(out_folder, "All_Models_Complete_Hyperparameters.txt",
              "\n".join(lines))

def make_blockwise_folds(df_meta: pd.DataFrame, n_folds: int = 10):
    """[REV-R3-C2] Contiguous block-wise fold construction, explicitly:
      1. Within EACH station, samples are sorted chronologically.
      2. Each station's time series is cut into `n_folds` contiguous,
         (near-)equal-length temporal blocks (no shuffling).
      3. Fold k = union of block k from every station.
    => every fold contains one contiguous temporal segment from every
    station, so the CV simultaneously probes temporal stability (unseen
    contiguous time blocks) and is balanced across all spatial nodes.
    Returns fold assignment array + a documentation table."""
    fold_assign = np.full(len(df_meta), -1, dtype=int)
    doc_rows = []
    for st, g in df_meta.groupby('Base_Station'):
        g_sorted = g.sort_values('Date and Time')
        idx = g_sorted.index.to_numpy()
        blocks = np.array_split(idx, n_folds)
        for k, block in enumerate(blocks):
            fold_assign[df_meta.index.get_indexer(block)] = k
            if len(block) > 0:
                doc_rows.append({
                    'Base_Station': st, 'Fold': k, 'N_Samples': len(block),
                    'Block_Start': str(df_meta.loc[block,
                                                   'Date and Time'].min()),
                    'Block_End': str(df_meta.loc[block,
                                                 'Date and Time'].max())})
    return fold_assign, pd.DataFrame(doc_rows)

def bootstrap_ci(y_true, y_pred, y_proba, metric: str,
                 n_boot: int = N_BOOTSTRAP, level: float = CI_LEVEL,
                 seed: int = GLOBAL_SEED):
    """[REV-R2-C3] Percentile bootstrap CI on the held-out test set."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    stats_list = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        try:
            if metric == 'Balanced_Accuracy':
                stats_list.append(balanced_accuracy_score(yt, yp))
            elif metric == 'Cohen_Kappa':
                stats_list.append(cohen_kappa_score(yt, yp))
            elif metric == 'AUC_Macro_OvR' and y_proba is not None:
                stats_list.append(roc_auc_score(yt, y_proba[idx],
                                                multi_class='ovr',
                                                average='macro'))
        except ValueError:
            continue
    if not stats_list:
        return np.nan, np.nan, np.nan
    lo = np.percentile(stats_list, (1 - level) / 2 * 100)
    hi = np.percentile(stats_list, (1 + level) / 2 * 100)
    return float(np.mean(stats_list)), float(lo), float(hi)

def stage6_classification():
    in_csv = (pathlib.Path(CONFIG["CLUSTER_OUT"])
              / 'Method_2_Deep_Clustering' / 'full_clustered_data.csv')
    unf_csv = (pathlib.Path(CONFIG["CLUSTER_OUT"])
               / 'unfiltered_deep_clustered_data.csv')
    out_dir = pathlib.Path(CONFIG["CLASSIF_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_data = []

    df = pd.read_csv(in_csv)
    print(f"[S6] Loaded {df.shape} labeled samples.")
    y = df[TARGET_COLUMN]
    meta = df[['Base_Station', 'Date and Time']].copy()
    meta['Date and Time'] = pd.to_datetime(meta['Date and Time'],
                                           errors='coerce')
    drop_cols = [c for c in FEATURES_TO_DROP if c in df.columns]
    X = df.drop(columns=drop_cols)
    print(f"[S6] Features kept ({len(X.columns)}): {list(X.columns)}")
    # Persist the exact ordered training feature list so Stage 7 rebuilds
    # the grid feature matrix with identical columns/order.
    with open(out_dir / 'Training_Features.json', 'w') as fh:
        json.dump(list(X.columns), fh, indent=2)

    num_feats = X.select_dtypes(include=np.number).columns.tolist()
    cat_feats = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test, meta_train, meta_test = \
        train_test_split(X, y, meta, test_size=0.2,
                         random_state=GLOBAL_SEED, stratify=y)

    transformers = [('num', StandardScaler(), num_feats)]
    if cat_feats:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'),
                             cat_feats))
    preprocessor = ColumnTransformer(transformers=transformers,
                                     remainder='passthrough')
    X_train_full = preprocessor.fit_transform(X_train)
    X_test_full = preprocessor.transform(X_test)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names_out = num_feats + cat_feats

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train), y=y_train)
    weights_dict = {i: w for i, w in enumerate(class_weights)}

    # ---------------- base models (original + [REV-R3-C3] additions) --------
    base_models = {
        'RandomForest': RandomForestClassifier(random_state=GLOBAL_SEED,
                                               n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=GLOBAL_SEED,
                                 eval_metric='mlogloss'),
        'CatBoost': CatBoostClassifier(random_state=GLOBAL_SEED, verbose=0),
        # ---- [REV-R3-C3] additional ensemble learners for comparison ----
        # GradientBoosting / AdaBoost cannot handle NaN natively, so they are
        # wrapped with a median imputer (documented in the hyper-param dump).
        'ExtraTrees': ExtraTreesClassifier(random_state=GLOBAL_SEED,
                                           n_jobs=-1),
        'GradientBoosting': SkPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', GradientBoostingClassifier(random_state=GLOBAL_SEED))]),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            random_state=GLOBAL_SEED),
        'AdaBoost': SkPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', AdaBoostClassifier(random_state=GLOBAL_SEED))]),
        'Bagging_RF': BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=50,
                                             random_state=GLOBAL_SEED),
            n_estimators=10, random_state=GLOBAL_SEED, n_jobs=-1),
    }
    if LGBM_AVAILABLE:
        base_models['LightGBM'] = LGBMClassifier(random_state=GLOBAL_SEED,
                                                 n_jobs=-1, verbose=-1)

    def evaluate_and_log(stage, name, model, X_te, y_te, note=""):
        y_pred = np.ravel(model.predict(X_te))   # CatBoost returns (n,1)
        try:
            y_proba = model.predict_proba(X_te)
        except Exception:
            y_proba = None
        m = clf_metrics(y_te, y_pred, y_proba)
        print(f"[S6]  [{stage}] {name}: BA={m['Balanced_Accuracy']:.4f} "
              f"Kappa={m['Cohen_Kappa']:.4f}")
        cmx = confusion_matrix(y_te, y_pred)
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(cmx, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(np.unique(y_te)),
                    yticklabels=sorted(np.unique(y_te)),
                    annot_kws={'size': 24, 'weight': 'bold'},
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=32, fontweight='bold')
        plt.ylabel('True Label', fontsize=32, fontweight='bold')
        plt.tight_layout()
        fstage = re.sub(r'[ .()]', '_', stage)
        fig.savefig(out_dir / f"CM_{fstage}_{name}.png", dpi=HIGH_DPI)
        plt.close(fig)
        results_data.append({'Stage': stage, 'Model': name, **m, 'Note': note})
        return m, y_pred, y_proba

    # Phase 1 — baseline, all features
    print("\n[S6] PHASE 1: Baseline (all features, incl. extra ensembles)")
    for name, model in base_models.items():
        try:
            model.fit(X_train_full, y_train)
            evaluate_and_log("Baseline", name, model, X_test_full, y_test,
                             "All Features")
        except Exception as e:
            print(f"[S6]  [!] {name} failed in baseline: {e}")

    # Phase 2 — feature selection (unchanged: SelectFromModel threshold=mean)
    print("\n[S6] PHASE 2: Feature selection (threshold=mean, RF importance)")
    sfm_est = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                     random_state=GLOBAL_SEED)
    sfm_est.fit(X_train_full, y_train)
    sfm = SelectFromModel(estimator=sfm_est, threshold='mean', prefit=True)
    mask_sfm = sfm.get_support()
    X_train_sfm = X_train_full[:, mask_sfm]
    X_test_sfm = X_test_full[:, mask_sfm]
    feats_sel = [f for f, s in zip(feature_names_out, mask_sfm) if s]
    print(f"[S6]  kept {len(feats_sel)} features: {feats_sel}")
    for name, model in base_models.items():
        try:
            model.fit(X_train_sfm, y_train)
            evaluate_and_log("Threshold Selection", name, model, X_test_sfm,
                             y_test, f"{len(feats_sel)} Features")
        except Exception as e:
            print(f"[S6]  [!] {name} failed in selection phase: {e}")

    # Phase 3 — cost-sensitive balancing (no SMOTE; physically valid)
    print("\n[S6] PHASE 3: Cost-sensitive balancing (class weights)")
    weighted_models = {
        'RandomForest_W': RandomForestClassifier(random_state=GLOBAL_SEED,
                                                 n_jobs=-1,
                                                 class_weight='balanced'),
        'XGBoost_W': XGBClassifier(random_state=GLOBAL_SEED,
                                   eval_metric='mlogloss'),
        'CatBoost_W': CatBoostClassifier(random_state=GLOBAL_SEED, verbose=0,
                                         class_weights=weights_dict),
        'ExtraTrees_W': ExtraTreesClassifier(random_state=GLOBAL_SEED,
                                             n_jobs=-1,
                                             class_weight='balanced'),
    }
    if LGBM_AVAILABLE:
        weighted_models['LightGBM_W'] = LGBMClassifier(
            random_state=GLOBAL_SEED, n_jobs=-1, verbose=-1,
            class_weight='balanced')
    for name, model in weighted_models.items():
        model.fit(X_train_sfm, y_train)
        evaluate_and_log("Cost-Sensitive", name, model, X_test_sfm, y_test,
                         "Balanced Weights")

    # Phase 4 — hyper-parameter tuning WITH DOCUMENTED GRIDS for every
    #           comparative model  [REV-R3-C3]
    print("\n[S6] PHASE 4: Hyper-parameter tuning (documented grids)")
    tuning_grids = {
        'RandomForest_Tuned': (
            RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1,
                                   class_weight='balanced'),
            {'n_estimators': [100, 200], 'max_depth': [None, 10, 20],
             'min_samples_leaf': [1, 3]}),
        'XGBoost_Tuned': (
            XGBClassifier(random_state=GLOBAL_SEED, eval_metric='mlogloss'),
            {'n_estimators': [200, 400], 'max_depth': [4, 6],
             'learning_rate': [0.05, 0.1]}),
        'CatBoost_Tuned': (
            CatBoostClassifier(random_state=GLOBAL_SEED, verbose=0,
                               class_weights=weights_dict),
            {'iterations': [200, 500], 'depth': [4, 6],
             'learning_rate': [0.05, 0.1]}),
        'ExtraTrees_Tuned': (
            ExtraTreesClassifier(random_state=GLOBAL_SEED, n_jobs=-1,
                                 class_weight='balanced'),
            {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
        'HistGB_Tuned': (
            HistGradientBoostingClassifier(random_state=GLOBAL_SEED),
            {'max_iter': [100, 200], 'max_depth': [None, 6],
             'learning_rate': [0.05, 0.1]}),
    }
    if LGBM_AVAILABLE:
        tuning_grids['LightGBM_Tuned'] = (
            LGBMClassifier(random_state=GLOBAL_SEED, n_jobs=-1, verbose=-1,
                           class_weight='balanced'),
            {'n_estimators': [200, 400], 'max_depth': [-1, 6],
             'learning_rate': [0.05, 0.1]})
    tuned_models, tuning_log = {}, []
    for name, (est, grid) in tuning_grids.items():
        gs = GridSearchCV(est, grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        gs.fit(X_train_sfm, y_train)
        tuned_models[name] = gs.best_estimator_
        tuning_log.append({'Model': name, 'Search': 'GridSearchCV',
                           'CV': '3-fold stratified',
                           'Scoring': 'f1_weighted',
                           'Grid': json.dumps(grid),
                           'Best_Params': json.dumps(gs.best_params_),
                           'Best_CV_Score': gs.best_score_})
        evaluate_and_log("Tuned", name, gs.best_estimator_, X_test_sfm,
                         y_test, "GridSearchCV")
    tuning_df = pd.DataFrame(tuning_log)
    tuning_df.to_csv(rev_dir("R3C3_Extra_Ensembles")
                     / "Hyperparameter_Tuning_Details_All_Models.csv",
                     index=False)
    best_model = tuned_models['CatBoost_Tuned']

    # ROC curve of the tuned CatBoost (unchanged style)
    y_test_bin = label_binarize(y_test, classes=sorted(np.unique(y_test)))
    y_score = best_model.predict_proba(X_test_sfm)
    fig = plt.figure(figsize=(16, 12))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    for i, color in zip(range(y_test_bin.shape[1]), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, color=color, lw=4.0,
                 label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=3.0, label='Random Classifier')
    plt.xlim([-0.02, 1.0]); plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=32, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=32, fontweight='bold')
    plt.legend(loc='lower right', fontsize=26, frameon=True,
               edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig.savefig(out_dir / 'Final_Model_ROC_Curve.png', dpi=HIGH_DPI)
    fig.savefig(out_dir / 'Final_Model_ROC_Curve.pdf', format='pdf')
    plt.close(fig)

    # Phase 5 — stacking ensemble (+ soft voting for comparison [REV-R3-C3])
    print("\n[S6] PHASE 5: Stacking (+ Voting) ensembles")
    stacking_model = build_stacking(weights_dict)
    stacking_model.fit(X_train_sfm, y_train)
    m_stack, y_pred_stack, y_proba_stack = evaluate_and_log(
        "Stacking Ensemble", "Stacking_Enhanced", stacking_model,
        X_test_sfm, y_test, "RF+XGB+Cat(+LGBM) -> LogisticRegression")

    voting_ests = [('rf', RandomForestClassifier(n_estimators=200,
                                                 random_state=GLOBAL_SEED,
                                                 n_jobs=-1,
                                                 class_weight='balanced')),
                   ('xgb', XGBClassifier(eval_metric='mlogloss',
                                         random_state=GLOBAL_SEED)),
                   ('cat', CatBoostClassifier(verbose=0,
                                              random_state=GLOBAL_SEED,
                                              iterations=300, depth=6,
                                              class_weights=weights_dict))]
    voting_model = VotingClassifier(estimators=voting_ests, voting='soft',
                                    n_jobs=-1)
    voting_model.fit(X_train_sfm, y_train)
    evaluate_and_log("Voting Ensemble", "SoftVoting_RF_XGB_Cat",
                     voting_model, X_test_sfm, y_test, "[REV-R3-C3]")

    # ---------------------------------------------------------------------
    # [REV-R3-R1 / Table 2] PER-CLASS DIAGNOSTIC METRICS
    # Table 2 of the manuscript reports per-class one-vs-rest AUC (and the
    # abstract quotes the Storm-class AUC). Only the macro AUC was exported
    # previously, so the per-class values behind Table 2 / Fig. 9 could not
    # be checked. They are now written out explicitly for BOTH the tuned
    # CatBoost (the model actually plotted in the ROC figure) and the final
    # stacking ensemble, together with per-class precision/recall/F1.
    # ---------------------------------------------------------------------
    # [REV] single source of truth: REGIME_NAMES_CANON (module level),
    # so the regime naming used in Table 2 can never drift from the naming
    # used in the figures.
    def per_class_report(model, name, X_te, y_te) -> pd.DataFrame:
        classes = sorted(np.unique(y_te))
        y_bin = label_binarize(y_te, classes=classes)
        proba = model.predict_proba(X_te)
        y_hat = np.ravel(model.predict(X_te))
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_te, y_hat, labels=classes, zero_division=0)
        rows = []
        for j, c in enumerate(classes):
            try:
                a = roc_auc_score(y_bin[:, j], proba[:, j])
            except Exception:
                a = np.nan
            rows.append({'Model': name, 'Class_Index': int(c),
                         'Regime': REGIME_NAMES_CANON.get(int(c), f'Class {c}'),
                         'ROC_AUC_OvR': round(float(a), 4),
                         'Precision': round(float(prec[j]), 4),
                         'Recall': round(float(rec[j]), 4),
                         'F1': round(float(f1[j]), 4),
                         'Support': int(sup[j])})
        return pd.DataFrame(rows)

    per_class_df = pd.concat([
        per_class_report(best_model, 'CatBoost_Tuned (ROC figure)',
                         X_test_sfm, y_test),
        per_class_report(stacking_model, 'Stacking_Enhanced (final)',
                         X_test_sfm, y_test)], ignore_index=True)
    per_class_df.to_csv(out_dir / 'Per_Class_Diagnostic_Metrics.csv',
                        index=False)
    save_text(out_dir, 'Per_Class_Diagnostic_Metrics.txt',
              "PER-CLASS DIAGNOSTIC METRICS (source of manuscript Table 2)\n"
              + "=" * 70 + "\n"
              "One-vs-rest ROC AUC plus precision/recall/F1 per wave regime.\n"
              "Class indices follow the canonical regime mapping:\n"
              "  0 = Confused Sea, 1 = Storm, 2 = Ambient/Calm, "
              "3 = Golden Swell\n"
              "(see R1C4_Cluster_Stability/Cluster_Physical_"
              "Interpretation.txt)\n\n"
              + per_class_df.to_string(index=False))
    print("[S6] Per-class diagnostics:")
    print(per_class_df.to_string(index=False))

    # dump the complete hyper-parameters of everything  [REV-R3-C3]
    all_models_for_dump = {**base_models, **weighted_models, **tuned_models,
                           'SoftVoting_RF_XGB_Cat': voting_model}
    dump_all_hyperparameters(all_models_for_dump, stacking_model,
                             rev_dir("R3C1_Model_Architecture_Hyperparams"))

    # -------------------------------------------------------------------------
    # [REV-R3-C2]  CONTIGUOUS BLOCK-WISE 10-FOLD CV — EXPLICIT CONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n[S6] [REV-R3-C2] Contiguous block-wise 10-fold CV ...")
    bw_dir = rev_dir("R3C2_Blockwise_CV")
    X_all = preprocessor.transform(X)[:, mask_sfm]
    y_all = y.to_numpy()
    meta_idx = meta.reset_index(drop=True)
    fold_assign, fold_doc = make_blockwise_folds(meta_idx, n_folds=10)
    fold_doc.to_csv(bw_dir / "Blockwise_Fold_Construction_Table.csv",
                    index=False)
    bw_rows = []
    for k in range(10):
        te = fold_assign == k
        tr = ~te
        if te.sum() == 0:
            continue
        mdl = build_stacking(weights_dict)
        mdl.fit(X_all[tr], y_all[tr])
        y_hat = mdl.predict(X_all[te])
        try:
            y_pp = mdl.predict_proba(X_all[te])
        except Exception:
            y_pp = None
        m = clf_metrics(y_all[te], y_hat, y_pp)
        bw_rows.append({'Fold': k, 'N_Test': int(te.sum()), **m})
        print(f"[S6]   fold {k}: BA={m['Balanced_Accuracy']:.4f}")
    bw_df = pd.DataFrame(bw_rows)
    bw_df.to_csv(bw_dir / "Blockwise_10Fold_CV_Results.csv", index=False)
    bw_summary = bw_df[['Accuracy', 'Balanced_Accuracy', 'Cohen_Kappa',
                        'F1_Macro', 'AUC_Macro_OvR']].agg(['mean', 'std'])
    bw_summary.to_csv(bw_dir / "Blockwise_10Fold_CV_Summary.csv")
    save_text(bw_dir, "Blockwise_CV_Design_Description.txt",
              "CONTIGUOUS BLOCK-WISE 10-FOLD CROSS-VALIDATION  [REV-R3-C2]\n"
              + "=" * 66 + "\n"
              "Fold construction (exactly as implemented in "
              "make_blockwise_folds):\n"
              "  1. Within EACH station, samples are sorted chronologically.\n"
              "  2. Each station's series is cut into 10 contiguous,\n"
              "     (near-)equal-length temporal blocks (no shuffling).\n"
              "  3. Fold k is the union of block k from every station.\n"
              "Consequently every validation fold consists of one unseen,\n"
              "contiguous temporal segment from every spatial node, testing\n"
              "temporal stability while remaining balanced across stations.\n"
              "Model evaluated: the full stacking ensemble (identical\n"
              "hyper-parameters as the final model).\n\n"
              "Per-fold results:\n" + bw_df.to_string(index=False) + "\n\n"
              "Summary (mean/std):\n" + bw_summary.to_string() + "\n\n"
              "Per-station block boundaries are documented in\n"
              "Blockwise_Fold_Construction_Table.csv")

    # -------------------------------------------------------------------------
    # [REV-R1-C2]  LEAVE-ONE-STATION-OUT + LEAVE-ONE-REGION-OUT
    # -------------------------------------------------------------------------
    print("\n[S6] [REV-R1-C2] Leave-One-Station-Out validation ...")
    loso_dir = rev_dir("R1C2_LOSO_LORO_Validation")
    stations_all = meta_idx['Base_Station'].unique()
    loso_rows = []
    for st in stations_all:
        te = (meta_idx['Base_Station'] == st).to_numpy()
        tr = ~te
        if te.sum() < 30 or len(np.unique(y_all[tr])) < 2:
            continue
        mdl = build_stacking(weights_dict)
        mdl.fit(X_all[tr], y_all[tr])
        y_hat = mdl.predict(X_all[te])
        try:
            y_pp = mdl.predict_proba(X_all[te])
        except Exception:
            y_pp = None
        m = clf_metrics(y_all[te], y_hat, y_pp)
        loso_rows.append({'Held_Out_Station': st,
                          'Region': station_region(st),
                          'N_Test': int(te.sum()), **m})
        print(f"[S6]   LOSO {st:<16} BA={m['Balanced_Accuracy']:.4f}")
    loso_df = pd.DataFrame(loso_rows)
    loso_df.to_csv(loso_dir / "LOSO_Per_Station_Results.csv", index=False)
    loso_sum = loso_df[['Balanced_Accuracy', 'Cohen_Kappa',
                        'F1_Macro']].agg(['mean', 'std', 'min', 'max'])
    loso_sum.to_csv(loso_dir / "LOSO_Summary.csv")

    fig, ax = plt.subplots(figsize=(16, 9))
    order = loso_df.sort_values('Balanced_Accuracy')['Held_Out_Station']
    sns.barplot(data=loso_df, x='Held_Out_Station', y='Balanced_Accuracy',
                order=order, hue='Region', dodge=False, edgecolor='black',
                linewidth=1.5, ax=ax)
    ax.axhline(loso_df['Balanced_Accuracy'].mean(), color='red',
               linestyle='--', linewidth=3, label='Mean BA')
    ax.set_xlabel('Held-out station', fontweight='bold', fontsize=26)
    ax.set_ylabel('Balanced Accuracy', fontweight='bold', fontsize=26)
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=20, frameon=True, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    save_fig(fig, loso_dir, "LOSO_BalancedAccuracy_Bar")

    print("\n[S6] [REV-R1-C2] Leave-One-Region-Out validation ...")
    meta_idx['Region'] = meta_idx['Base_Station'].apply(station_region)
    loro_rows = []
    for reg in ['north', 'central', 'south']:
        te = (meta_idx['Region'] == reg).to_numpy()
        tr = ~te
        if te.sum() < 30 or len(np.unique(y_all[tr])) < 2:
            continue
        mdl = build_stacking(weights_dict)
        mdl.fit(X_all[tr], y_all[tr])
        y_hat = mdl.predict(X_all[te])
        try:
            y_pp = mdl.predict_proba(X_all[te])
        except Exception:
            y_pp = None
        m = clf_metrics(y_all[te], y_hat, y_pp)
        loro_rows.append({'Held_Out_Region': reg, 'N_Test': int(te.sum()),
                          **m})
        print(f"[S6]   LORO {reg:<8} BA={m['Balanced_Accuracy']:.4f}")
    loro_df = pd.DataFrame(loro_rows)
    loro_df.to_csv(loso_dir / "LORO_Per_Region_Results.csv", index=False)
    save_text(loso_dir, "LOSO_LORO_Report.txt",
              "SPATIAL GENERALIZATION - LOSO & LORO  [REV-R1-C2]\n"
              + "=" * 60 + "\n"
              "Leave-One-Station-Out: for each base station, the stacking\n"
              "ensemble is retrained on ALL other stations and evaluated on\n"
              "the fully withheld station (no temporal or spatial leakage).\n"
              "Leave-One-Region-Out: same protocol with north (>22.5N),\n"
              "central (18-22.5N) and south (<18N) held out in turn.\n\n"
              "LOSO per-station:\n" + loso_df.to_string(index=False) + "\n\n"
              "LOSO summary:\n" + loso_sum.to_string() + "\n\n"
              "LORO per-region:\n" + loro_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # [REV-R2-C3]  (a) BOOTSTRAP 95% CIs on the held-out test set
    # -------------------------------------------------------------------------
    print("\n[S6] [REV-R2-C3] Bootstrap confidence intervals ...")
    ci_dir = rev_dir("R2C3_CIs_RepeatedValidation_Stats")
    ci_rows = []
    for metric in ['Balanced_Accuracy', 'Cohen_Kappa', 'AUC_Macro_OvR']:
        mean_b, lo, hi = bootstrap_ci(y_test, y_pred_stack, y_proba_stack,
                                      metric)
        point = m_stack.get(metric, np.nan)
        ci_rows.append({'Model': 'Stacking_Enhanced', 'Metric': metric,
                        'Point_Estimate': point, 'Bootstrap_Mean': mean_b,
                        'CI95_Low': lo, 'CI95_High': hi,
                        'N_Bootstrap': N_BOOTSTRAP})
        print(f"[S6]   {metric}: {point:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(ci_dir / "Bootstrap_95CI_TestSet.csv", index=False)

    # -------------------------------------------------------------------------
    # [REV-R2-C3]  (b) REPEATED VALIDATION over multiple random seeds
    #   The full pipeline split->preprocess->select->stacking is repeated for
    #   each seed in REPEATED_EVAL_SEEDS; mean +/- std demonstrate stability.
    # -------------------------------------------------------------------------
    print("\n[S6] [REV-R2-C3] Repeated validation over random seeds ...")
    rep_rows = []
    unf_available = unf_csv.exists()
    if unf_available:
        df_unf = pd.read_csv(unf_csv)
        has_unf_labels = TARGET_COLUMN in df_unf.columns
    for s in REPEATED_EVAL_SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=s, stratify=y)
        prep_s = ColumnTransformer(transformers=transformers,
                                   remainder='passthrough')
        Xtr_t = prep_s.fit_transform(Xtr)
        Xte_t = prep_s.transform(Xte)
        sel_est = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                         random_state=s)
        sel_est.fit(Xtr_t, ytr)
        mask_s = SelectFromModel(sel_est, threshold='mean',
                                 prefit=True).get_support()
        cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
        wd = {i: w for i, w in enumerate(cw)}
        mdl = build_stacking(wd)
        mdl.fit(Xtr_t[:, mask_s], ytr)
        y_hat = mdl.predict(Xte_t[:, mask_s])
        y_pp = mdl.predict_proba(Xte_t[:, mask_s])
        m_on = clf_metrics(yte, y_hat, y_pp)
        row = {'Seed': s, 'Condition': 'Gate_ON_filtered', **m_on}
        rep_rows.append(row)

        # Gate OFF: evaluate the SAME trained model on the unfiltered data
        if unf_available and has_unf_labels:
            drop_u = [c for c in FEATURES_TO_DROP if c in df_unf.columns]
            X_unf = df_unf.drop(columns=drop_u)
            for col in set(X.columns) - set(X_unf.columns):
                X_unf[col] = X[col].median()
            X_unf = X_unf[X.columns]
            X_unf_t = prep_s.transform(X_unf)[:, mask_s]
            y_unf = df_unf[TARGET_COLUMN]
            y_hat_u = mdl.predict(X_unf_t)
            y_pp_u = mdl.predict_proba(X_unf_t)
            m_off = clf_metrics(y_unf, y_hat_u, y_pp_u)
            rep_rows.append({'Seed': s, 'Condition': 'Gate_OFF_unfiltered',
                             **m_off})
        print(f"[S6]   seed {s}: BA(on)={m_on['Balanced_Accuracy']:.4f}")
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(ci_dir / "Repeated_Validation_Across_Seeds.csv", index=False)
    rep_summary = (rep_df.groupby('Condition')
                   [['Balanced_Accuracy', 'Cohen_Kappa', 'AUC_Macro_OvR']]
                   .agg(['mean', 'std']))
    rep_summary.to_csv(ci_dir / "Repeated_Validation_Summary.csv")

    # -------------------------------------------------------------------------
    # [REV-R2-C3]  (c) STATISTICAL SIGNIFICANCE: Gate ON vs Gate OFF
    #   Paired (per-seed) t-test + Wilcoxon signed-rank on Balanced Accuracy
    # -------------------------------------------------------------------------
    stat_lines = ["STATISTICAL COMPARISON: FIDELITY GATE ON vs OFF "
                  "[REV-R2-C3]", "=" * 66]
    on_ba = rep_df.loc[rep_df['Condition'] == 'Gate_ON_filtered',
                       'Balanced_Accuracy'].to_numpy()
    off_ba = rep_df.loc[rep_df['Condition'] == 'Gate_OFF_unfiltered',
                        'Balanced_Accuracy'].to_numpy()
    if len(on_ba) == len(off_ba) and len(on_ba) >= 3:
        t_stat, p_t = sps.ttest_rel(on_ba, off_ba)
        try:
            w_stat, p_w = sps.wilcoxon(on_ba, off_ba)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        d = (np.mean(on_ba - off_ba)
             / (np.std(on_ba - off_ba, ddof=1) + 1e-12))
        stat_lines += [
            f"Seeds compared (paired): {REPEATED_EVAL_SEEDS}",
            f"Gate ON  Balanced Accuracy: {on_ba.mean():.4f} "
            f"+/- {on_ba.std(ddof=1):.4f}",
            f"Gate OFF Balanced Accuracy: {off_ba.mean():.4f} "
            f"+/- {off_ba.std(ddof=1):.4f}",
            f"Mean difference: {np.mean(on_ba - off_ba):.4f}",
            f"Paired t-test:        t = {t_stat:.3f}, p = {p_t:.3e}",
            f"Wilcoxon signed-rank: W = {w_stat}, p = {p_w:.3e}",
            f"Cohen's d (paired):   {d:.3f}",
            "",
            "Interpretation: p < 0.05 indicates the performance advantage of",
            "the Fidelity Gate is statistically significant and not an",
            "artefact of a particular train/test split."]
    else:
        stat_lines.append("Gate-OFF evaluations unavailable "
                          "(unfiltered labels missing).")
    save_text(ci_dir, "GateON_vs_GateOFF_Statistical_Tests.txt",
              "\n".join(stat_lines))

    # -------------------------------------------------------------------------
    # ABLATION STUDY (Gate ON vs OFF, single-seed) — as in the manuscript,
    # now augmented with the seed-wise statistics above.
    # -------------------------------------------------------------------------
    if unf_available and has_unf_labels:
        drop_u = [c for c in FEATURES_TO_DROP if c in df_unf.columns]
        X_unf = df_unf.drop(columns=drop_u)
        for col in set(X.columns) - set(X_unf.columns):
            X_unf[col] = X[col].median()
        X_unf = X_unf[X.columns]
        X_unf_t = preprocessor.transform(X_unf)[:, mask_sfm]
        y_unf = df_unf[TARGET_COLUMN]
        y_hat_u = stacking_model.predict(X_unf_t)
        pp_u = stacking_model.predict_proba(X_unf_t)
        m_off1 = clf_metrics(y_unf, y_hat_u, pp_u)
        conf_off = float(np.mean(np.max(pp_u, axis=1)))
        conf_on = float(np.mean(np.max(y_proba_stack, axis=1)))
        abl = ["ABLATION STUDY: IMPACT OF THE FIDELITY GATE", "=" * 60,
               "1. EPISTEMIC UNCERTAINTY (mean max-probability confidence):",
               f"   Gate ON  (quality-controlled): {conf_on:.4f}",
               f"   Gate OFF (raw/unfiltered):     {conf_off:.4f}", "",
               "2. CLASSIFICATION METRICS:",
               "   Gate ON  : "
               f"Acc {m_stack['Accuracy']:.4f} | "
               f"BA {m_stack['Balanced_Accuracy']:.4f} | "
               f"Kappa {m_stack['Cohen_Kappa']:.4f}",
               "   Gate OFF : "
               f"Acc {m_off1['Accuracy']:.4f} | "
               f"BA {m_off1['Balanced_Accuracy']:.4f} | "
               f"Kappa {m_off1['Cohen_Kappa']:.4f}", "",
               "3. Seed-wise significance tests: see",
               "   R2C3_CIs_RepeatedValidation_Stats/"
               "GateON_vs_GateOFF_Statistical_Tests.txt"]
        save_text(out_dir, "Fidelity_Gate_Ablation_Study.txt", "\n".join(abl))

    # -------------------------------------------------------------------------
    # [REV-R1-C1] DOWNSTREAM GATE-THRESHOLD SENSITIVITY
    #   For every (corr, bias) gate setting: rebuild the training pool from
    #   the unfiltered labeled data using per-station fidelity statistics,
    #   train a cost-sensitive Random Forest (computational proxy for the
    #   full stack) and report held-out metrics.
    # -------------------------------------------------------------------------
    stats_csv = (pathlib.Path(CONFIG["MERGE_OUT"])
                 / "Station_Fidelity_Statistics.csv")
    if unf_available and has_unf_labels and stats_csv.exists():
        print("\n[S6] [REV-R1-C1] Downstream gate-threshold sensitivity ...")
        st_stats = pd.read_csv(stats_csv)
        sens_rows = []
        drop_u = [c for c in FEATURES_TO_DROP if c in df_unf.columns]
        for c_thr in GATE_CORR_GRID:
            for b_thr in GATE_BIAS_GRID:
                tags_ok = st_stats.loc[
                    (st_stats['hm0_corr'] >= c_thr)
                    & (st_stats['hm0_bias'].abs() <= b_thr), 'tag'].tolist()
                pool = df_unf[df_unf['Location_Tag'].isin(tags_ok)].copy()
                if (len(pool) < 400
                        or pool[TARGET_COLUMN].nunique() < N_CLUSTERS):
                    sens_rows.append({'Corr_Threshold': c_thr,
                                      'Bias_Threshold_m': b_thr,
                                      'Stations': len(tags_ok),
                                      'Samples': len(pool),
                                      'Balanced_Accuracy': np.nan,
                                      'Cohen_Kappa': np.nan,
                                      'Note': 'insufficient data/classes'})
                    continue
                Xp = pool.drop(columns=[c for c in drop_u
                                        if c in pool.columns])
                for col in set(X.columns) - set(Xp.columns):
                    Xp[col] = X[col].median()
                Xp = Xp[X.columns]
                yp = pool[TARGET_COLUMN]
                Xtr, Xte, ytr, yte = train_test_split(
                    Xp, yp, test_size=0.2, random_state=GLOBAL_SEED,
                    stratify=yp)
                prep_g = ColumnTransformer(transformers=transformers,
                                           remainder='passthrough')
                Xtr_t = prep_g.fit_transform(Xtr)
                Xte_t = prep_g.transform(Xte)
                rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=GLOBAL_SEED,
                                            class_weight='balanced')
                rf.fit(Xtr_t, ytr)
                m = clf_metrics(yte, rf.predict(Xte_t),
                                rf.predict_proba(Xte_t))
                sens_rows.append({'Corr_Threshold': c_thr,
                                  'Bias_Threshold_m': b_thr,
                                  'Stations': len(tags_ok),
                                  'Samples': len(pool),
                                  'Balanced_Accuracy':
                                      m['Balanced_Accuracy'],
                                  'Cohen_Kappa': m['Cohen_Kappa'],
                                  'Note': ''})
        sens_ds = pd.DataFrame(sens_rows)
        sdir = rev_dir("R1C1_FidelityGate_Sensitivity")
        sens_ds.to_csv(sdir / "Gate_Sensitivity_Downstream.csv", index=False)
        piv = sens_ds.pivot(index='Bias_Threshold_m',
                            columns='Corr_Threshold',
                            values='Balanced_Accuracy')
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(piv, annot=True, fmt='.3f', cmap='viridis',
                    linewidths=1.0, linecolor='black',
                    annot_kws={'size': 20, 'weight': 'bold'},
                    cbar_kws={'label': 'Balanced Accuracy (held-out)',
                              'shrink': 0.85}, ax=ax)
        ax.set_xlabel('Pearson-R threshold (Hm0)', fontweight='bold',
                      fontsize=26)
        ax.set_ylabel('|Bias| threshold (m)', fontweight='bold', fontsize=26)
        try:
            ci_ = list(piv.columns).index(MIN_CORR_THRESHOLD)
            bi_ = list(piv.index).index(MAX_BIAS_THRESHOLD)
            ax.add_patch(plt.Rectangle((ci_, bi_), 1, 1, fill=False,
                                       edgecolor='red', lw=5))
        except ValueError:
            pass
        save_fig(fig, sdir, "Gate_Sensitivity_Downstream_BA_Heatmap")
        save_text(sdir, "Gate_Sensitivity_Downstream_Report.txt",
                  "DOWNSTREAM SENSITIVITY OF THE FIDELITY-GATE THRESHOLDS\n"
                  "[REV-R1-C1 / R2 Major-2]\n" + "=" * 66 + "\n"
                  "Protocol: for each (Corr, |Bias|) setting the training\n"
                  "pool is rebuilt from stations passing that gate; a cost-\n"
                  "sensitive Random Forest (proxy for the full stack, seed\n"
                  f"{GLOBAL_SEED}) is trained on 80% and evaluated on a\n"
                  "stratified 20% hold-out of that pool. The red box marks\n"
                  "the manuscript operating point (0.80 / 0.50 m).\n\n"
                  + sens_ds.to_string(index=False))

    # -------------------------------------------------------------------------
    # Robustness (unchanged 10-fold shuffled CV of class-weighted RF)
    # -------------------------------------------------------------------------
    robust = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                    random_state=GLOBAL_SEED,
                                    class_weight='balanced')
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_SEED)
    cv_scores = cross_val_score(robust, X_train_sfm, y_train, cv=cv,
                                scoring='accuracy', n_jobs=-1)
    save_text(out_dir, "Robustness_CrossVal_Report.txt",
              "ROBUSTNESS ANALYSIS (10-FOLD SHUFFLED CV)\n"
              "=========================================\n"
              "Model: Random Forest (class-weighted)\n"
              f"Mean Accuracy: {cv_scores.mean():.4f}\n"
              f"Std Deviation: {cv_scores.std():.4f}\n"
              f"Scores per fold: {cv_scores}\n"
              "NOTE: the leakage-safe contiguous block-wise CV requested by\n"
              "Reviewer 3 is reported separately in R3C2_Blockwise_CV/.")

    # Reliability / uncertainty diagram (unchanged)
    # NOTE: CatBoost.predict returns a 2-D (n,1) array -> ravel both sides.
    probas = best_model.predict_proba(X_test_sfm)
    confidence = np.max(probas, axis=1)
    correct = (np.ravel(best_model.predict(X_test_sfm))
               == np.asarray(y_test).ravel())
    df_unc = pd.DataFrame({'Confidence': confidence, 'Correct': correct})
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels_b = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    df_unc['Conf_Bin'] = pd.cut(df_unc['Confidence'], bins=bins,
                                labels=labels_b)
    bin_acc = df_unc.groupby('Conf_Bin', observed=False)['Correct'].mean()
    bin_counts = df_unc.groupby('Conf_Bin', observed=False)['Correct'].count()
    fig = plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=bin_acc.index, y=bin_acc.values, hue=bin_acc.index,
                     palette='RdYlGn', legend=False, edgecolor='black',
                     linewidth=1.5)
    plt.axhline(y=0.9, color='red', linestyle='--', linewidth=3.5,
                label='90% Accuracy Threshold')
    plt.xlabel('Prediction Confidence Bin', fontsize=32, fontweight='bold')
    plt.ylabel('Observed Accuracy', fontsize=32, fontweight='bold')
    plt.ylim(0, 1.05)
    for i, p in enumerate(ax.patches):
        if i < len(bin_counts) and bin_counts.iloc[i] > 0:
            ax.annotate(f'n={bin_counts.iloc[i]}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', fontsize=22,
                        fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=26,
               edgecolor='black')
    plt.tight_layout()
    fig.savefig(out_dir / 'Uncertainty_Reliability_Plot.png', dpi=HIGH_DPI)
    fig.savefig(out_dir / 'Uncertainty_Reliability_Plot.pdf', format='pdf')
    plt.close(fig)

    # Save final model + selector (unchanged)
    final_pipe = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                        random_state=GLOBAL_SEED,
                                        class_weight='balanced')
    final_pipe.fit(X_train_sfm, y_train)
    joblib.dump(final_pipe, out_dir / 'Final_Model_RF_Balanced.pkl')
    joblib.dump(stacking_model, out_dir / 'Final_Stacking_Model.pkl')
    joblib.dump(sfm, out_dir / 'Feature_Selector.pkl')
    joblib.dump(preprocessor, out_dir / 'Preprocessor.pkl')

    # -------------------------------------------------------------------------
    # [REV-FIG10] REWORKED MODEL-EVOLUTION FIGURE: Balanced Accuracy + Kappa
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(out_dir / 'Model_Performance_Report.csv', index=False)
    df_plot = df_results[~df_results['Model'].str.contains(
        'LightGBM|TabNet', case=False, na=False)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(26, 20), sharex=True)
    for ax_i, metric, ylab in [(axes[0], 'Balanced_Accuracy',
                                'Balanced Accuracy'),
                               (axes[1], 'Cohen_Kappa', "Cohen's Kappa")]:
        sns.barplot(data=df_plot, x='Stage', y=metric, hue='Model',
                    palette='viridis', edgecolor='black', linewidth=1.2,
                    ax=ax_i)
        ax_i.set_ylabel(ylab, fontsize=34, fontweight='bold', labelpad=20)
        ax_i.set_xlabel('')
        ax_i.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax_i.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                    title='Algorithm', fontsize=18, title_fontsize=22,
                    frameon=True, edgecolor='black')
    axes[1].set_xlabel('Processing Stage', fontsize=34, fontweight='bold',
                       labelpad=22)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=24)
    fig.tight_layout()
    fig.savefig(out_dir / 'Model_Evolution_Comparison_BA_Kappa.png',
                dpi=HIGH_DPI)
    fig.savefig(out_dir / 'Model_Evolution_Comparison_BA_Kappa.pdf',
                format='pdf')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # [REV-TABLES] COMBINED COMPREHENSIVE RESULTS WORKBOOK
    # -------------------------------------------------------------------------
    tables_dir = rev_dir("Combined_Tables")
    combined = tables_dir / "Comprehensive_Results_Tables.xlsx"
    with pd.ExcelWriter(combined) as writer:
        df_results.to_excel(writer, sheet_name='All_Model_Performance',
                            index=False)
        per_class_df.to_excel(writer, sheet_name='Per_Class_Table2',
                              index=False)
        tuning_df.to_excel(writer, sheet_name='Tuning_Details', index=False)
        bw_df.to_excel(writer, sheet_name='Blockwise_10Fold_CV', index=False)
        loso_df.to_excel(writer, sheet_name='LOSO_Stations', index=False)
        loro_df.to_excel(writer, sheet_name='LORO_Regions', index=False)
        ci_df.to_excel(writer, sheet_name='Bootstrap_95CI', index=False)
        rep_df.to_excel(writer, sheet_name='Repeated_Seeds', index=False)
        try:
            sens_ds.to_excel(writer, sheet_name='Gate_Sensitivity',
                             index=False)
        except NameError:
            pass
    print(f"[S6] Combined comprehensive tables -> {combined}")

# =============================================================================
# SECTION 7 — STAGE 7: SPATIAL MAPPING, STORM VALIDATION, SITE SELECTION
#   + [REV-R1-C5] extra cyclones (Hikaa, Luban) + non-storm false-alarm audit
#   + [REV-R1-C6] explicit Pareto front with depth constraint + caveats
# =============================================================================
LON_STEP, LAT_STEP = 2.0, 2.0

def calculate_engineering_features_grid(ds) -> pd.DataFrame:
    df = ds.to_dataframe().reset_index().dropna()
    df = df.sort_values(by=['latitude', 'longitude', 'time'])
    df['loc_id'] = df['latitude'].astype(str) + "_" + df['longitude'].astype(str)
    g = df.groupby('loc_id')
    df['Model_Height_Stability'] = g['VHM0'].transform(
        lambda x: x.rolling(8, min_periods=1).std()).fillna(0)
    df['Model_Period_Stability'] = g['VTPK'].transform(
        lambda x: x.rolling(8, min_periods=1).std()).fillna(0)
    df['Direction_Stability'] = g['VMDR'].transform(
        lambda x: x.rolling(8, min_periods=1).std()).fillna(0)
    if 'Swell_Stability' not in df.columns:
        df['Swell_Stability'] = df['Model_Height_Stability']
    return df

def add_bathymetry_grid(df, bathy_path) -> pd.DataFrame:
    if bathy_path is None or not os.path.exists(bathy_path):
        df['Depth_CMEMS'] = 100
        return df
    try:
        ds_b = xr.open_dataset(bathy_path)
        if 'deptho' in ds_b:
            bathy_df = ds_b['deptho'].to_dataframe().reset_index().dropna()
            df['lat_round'] = df['latitude'].round(1)
            df['lon_round'] = df['longitude'].round(1)
            bathy_df['lat_round'] = bathy_df['latitude'].round(1)
            bathy_df['lon_round'] = bathy_df['longitude'].round(1)
            merged = pd.merge(df, bathy_df[['lat_round', 'lon_round',
                                            'deptho']],
                              on=['lat_round', 'lon_round'], how='left')
            merged.rename(columns={'deptho': 'Depth_CMEMS'}, inplace=True)
            merged['Depth_CMEMS'] = merged['Depth_CMEMS'].fillna(1000)
            return merged
    except Exception as e:
        print(f"[S7]  bathymetry read failed: {e}")
    df['Depth_CMEMS'] = 100
    return df

def format_spatial_axis(ax, pivot_table):
    lon_vals = pivot_table.columns.values
    lat_vals = pivot_table.index.values
    lon_ticks = np.arange(lon_vals.min(), lon_vals.max(), LON_STEP)
    lat_ticks = np.arange(lat_vals.min(), lat_vals.max(), LAT_STEP)
    lon_idx = [np.abs(lon_vals - t).argmin() for t in lon_ticks]
    lat_idx = [np.abs(lat_vals - t).argmin() for t in lat_ticks]
    ax.set_xticks(lon_idx)
    ax.set_xticklabels([f"{lon_vals[i]:.0f}" for i in lon_idx], fontsize=22)
    ax.set_yticks(lat_idx)
    ax.set_yticklabels([f"{lat_vals[i]:.0f}" for i in lat_idx], fontsize=22)
    ax.set_xlabel('Longitude (deg E)', fontweight='bold', fontsize=26)
    ax.set_ylabel('Latitude (deg N)', fontweight='bold', fontsize=26)

# ---------------------------------------------------------------------------
# [REV-R1-C6, DOMAIN FIX] Omani coastal-waters mask.
# The grid domain (52-60 E, 16-26.4 N) covers far more than Oman: it also
# includes the Persian Gulf (UAE/Qatar) and the Iranian Makran coast. A
# depth<=100 m filter ALONE therefore admits shallow cells that are NOT in
# Omani waters -- e.g. 25.4 N / 59.6 E lies off Chabahar (Iran) and
# 24.0 N / 53.6 E lies off Abu Dhabi. Because this study is explicitly about
# WEC siting for OMAN, candidate sites must additionally lie within
# MAX_DIST_TO_OMAN_COAST_KM of the Omani coastline, approximated by the
# waypoint polyline below (Musandam -> Batinah -> Muscat -> Ra's al Hadd ->
# Masirah -> Duqm -> Shuwaymiyah -> Salalah -> Yemen border).
OMAN_COAST_WAYPOINTS = [
    (26.38, 56.28), (26.10, 56.40), (25.80, 56.30),   # Musandam
    (24.40, 56.70), (24.10, 56.95), (23.90, 57.40),   # Batinah
    (23.75, 57.90), (23.62, 58.30), (23.60, 58.60),   # Muscat area
    (23.30, 58.95), (22.90, 59.40), (22.55, 59.80),   # Ra's al Hadd
    (22.00, 59.75), (21.50, 59.40), (21.00, 59.05),   # NE Arabian Sea
    (20.60, 58.90), (20.20, 58.55), (19.95, 58.10),   # Masirah area
    (19.70, 57.70), (19.35, 57.55), (19.00, 57.30),   # Duqm
    (18.60, 56.75), (18.25, 56.30), (18.00, 55.90),   # Al Wusta
    (17.90, 55.50), (17.60, 55.15), (17.30, 54.80),   # Shuwaymiyah
    (17.05, 54.35), (16.95, 54.00), (16.90, 53.60),   # Salalah
    (16.70, 53.20), (16.65, 53.05),                   # Yemen border
]
# Neighbouring (non-Omani) coasts, used for an equidistance / median-line
# test: a cell is treated as Omani only if it is closer to the Omani coast
# than to any of these. This is the standard approximation of a maritime
# median line and cleanly removes Persian-Gulf and Iranian-Makran cells
# that a distance-only rule would otherwise admit.
FOREIGN_COAST_WAYPOINTS = [
    (25.65, 57.77), (25.45, 59.05), (25.45, 59.60),   # Iran: Jask -> Chabahar
    (25.30, 60.00), (26.00, 57.20), (26.55, 56.50),   # Iran: Makran -> Hormuz
    (25.12, 56.33), (25.35, 56.35),                   # UAE east (Fujairah)
    (25.80, 55.95), (25.30, 55.30), (24.90, 54.90),   # RAK / Dubai
    (24.47, 54.37), (24.20, 53.70), (24.05, 52.60),   # Abu Dhabi
    (24.50, 51.50), (25.30, 51.50),                   # Qatar
    (16.50, 52.80), (15.90, 52.20),                   # Yemen
]
MAX_DIST_TO_OMAN_COAST_KM = 150.0

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorized over lat1/lon1)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = (np.sin(dp / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2)
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def distance_to_oman_coast_km(lats, lons) -> np.ndarray:
    """Minimum great-circle distance (km) from each point to the Omani
    coastline waypoint polyline."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    dmin = np.full(lats.shape, np.inf)
    for (wlat, wlon) in OMAN_COAST_WAYPOINTS:
        dmin = np.minimum(dmin, _haversine_km(lats, lons, wlat, wlon))
    return dmin

def distance_to_foreign_coast_km(lats, lons) -> np.ndarray:
    """Minimum great-circle distance (km) to the nearest NON-Omani coast
    (Iran, UAE, Qatar, Yemen) — used for the median-line test."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    dmin = np.full(lats.shape, np.inf)
    for (wlat, wlon) in FOREIGN_COAST_WAYPOINTS:
        dmin = np.minimum(dmin, _haversine_km(lats, lons, wlat, wlon))
    return dmin

def plot_spatial_heatmap(grid_stats, metric, label, folder, filename,
                         cmap='viridis', highlight_max=False, use_log=False,
                         star_point=None):
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot = grid_stats.pivot(index='latitude', columns='longitude',
                             values=metric)
    plot_args = {'cmap': cmap, 'cbar_kws': {'label': label, 'shrink': 0.8}}
    if use_log:
        pmin = pivot[pivot > 0].min().min()
        eps = pmin / 10.0 if pd.notna(pmin) else 0.01
        pivot_plot = pivot.replace(0.0, eps)
        plot_args['norm'] = LogNorm(vmin=eps, vmax=pivot_plot.max().max())
        label += " (Log Scale)"
        plot_args['cbar_kws']['label'] = label
    else:
        pivot_plot = pivot
    sns.heatmap(pivot_plot, ax=ax, **plot_args)
    ax.invert_yaxis()
    format_spatial_axis(ax, pivot)
    star = star_point
    if highlight_max and star is None:
        coastal = grid_stats[grid_stats['Depth_CMEMS'] <= 100]
        if 'In_Oman_Waters' in coastal.columns:      # [REV-R1-C6 domain fix]
            coastal = coastal[coastal['In_Oman_Waters']]
        if not coastal.empty:
            star = coastal.loc[coastal[metric].idxmax()]
    if star is not None:
        try:
            r = pivot.index.get_loc(star['latitude'])
            c = pivot.columns.get_loc(star['longitude'])
            ax.scatter(c + 0.5, r + 0.5, color='gold', marker='*', s=500,
                       edgecolors='black', linewidth=2.5, zorder=10,
                       label='Pareto-optimal site')
            ax.legend(loc='upper right', frameon=True, facecolor='white',
                      edgecolor='black', fontsize=24)
        except KeyError:
            pass
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label(label, fontsize=24, fontweight='bold', labelpad=20)
    save_fig(fig, folder, filename)

def pareto_front(df: pd.DataFrame, maximize: str, minimize: str) -> pd.Series:
    """[REV-R1-C6] Boolean mask of the non-dominated set:
    a cell is Pareto-optimal if no other cell has BOTH higher `maximize`
    AND lower `minimize`."""
    vals = df[[maximize, minimize]].to_numpy()
    n = len(vals)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        better = ((vals[:, 0] >= vals[i, 0]) & (vals[:, 1] <= vals[i, 1])
                  & ((vals[:, 0] > vals[i, 0]) | (vals[:, 1] < vals[i, 1])))
        if better.any():
            dominated[i] = True
    return pd.Series(~dominated, index=df.index)

def storm_window_stats(df_full, model, selector_mask_or_obj, required_feats,
                       t0, t1, preprocessor=None):
    """Grid storm-probabilities inside a time window."""
    mask = (df_full['time'] >= t0) & (df_full['time'] <= t1)
    sub = df_full[mask]
    if sub.empty:
        return pd.DataFrame()
    recs = []
    for (lat, lon), grp in sub.groupby(['latitude', 'longitude']):
        X_loc = grp[required_feats]
        # [REV-BUGFIX] apply the SAME StandardScaler used at training time
        # before feature selection — the model was trained on standardized
        # features; skipping this step silently saturates every split in
        # the tree-based model and collapses predictions to one class.
        X_loc = (preprocessor.transform(X_loc) if preprocessor is not None
                 else X_loc.values)
        try:
            X_sel = selector_mask_or_obj.transform(X_loc)
        except AttributeError:
            X_sel = X_loc[:, selector_mask_or_obj]
        preds = model.predict(X_sel)
        recs.append({'latitude': lat, 'longitude': lon, 'Total': len(preds),
                     'Storm': int((preds == STORM_CLASS).sum())})
    out = pd.DataFrame(recs)
    out['Storm_Risk'] = out['Storm'] / out['Total'] * 100
    return out

def stage7_maps():
    OUTPUT_DIR = pathlib.Path(CONFIG["MAPS_OUT"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(CONFIG["CLASSIF_OUT"]) / 'Final_Model_RF_Balanced.pkl'
    selector_path = pathlib.Path(CONFIG["CLASSIF_OUT"]) / 'Feature_Selector.pkl'
    preprocessor_path = pathlib.Path(CONFIG["CLASSIF_OUT"]) / 'Preprocessor.pkl'
    if not model_path.exists() or not selector_path.exists():
        raise FileNotFoundError("[S7] model/selector missing — run Stage 6.")
    model = joblib.load(model_path)
    selector = joblib.load(selector_path)
    # [REV-BUGFIX] The saved model/selector were fit on STANDARDIZED
    # features (preprocessor = StandardScaler via ColumnTransformer,
    # fit in Stage 6). Predicting on raw physical-unit grid values without
    # re-applying this exact scaler saturates every split threshold in the
    # tree-based model, collapsing predictions to a single class across the
    # whole grid. This preprocessor MUST be applied before selector.transform.
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "[S7] Preprocessor.pkl missing — re-run Stage 6 with the "
            "current code so the fitted StandardScaler is saved.")
    preprocessor = joblib.load(preprocessor_path)
    print(f"[S7] Preprocessor loaded from {preprocessor_path.name} "
          f"(will be applied before feature selection)")

    feat_json = pathlib.Path(CONFIG["CLASSIF_OUT"]) / 'Training_Features.json'
    if feat_json.exists():
        with open(feat_json) as fh:
            required = json.load(fh)
        print(f"[S7] Training feature list loaded ({len(required)} features) "
              f"from {feat_json.name}")
    elif hasattr(selector, 'feature_names_in_'):
        required = list(selector.feature_names_in_)
    else:
        required = ['VHM0', 'VHM0_SW1', 'VHM0_SW2', 'VHM0_WW', 'VMDR',
                    'VMDR_SW1', 'VMDR_SW2', 'VMDR_WW', 'VPED', 'VSDX',
                    'VSDY', 'VTM01_SW1', 'VTM01_SW2', 'VTM01_WW', 'VTM02',
                    'VTM10', 'VTPK', 'Depth_CMEMS', 'Model_Height_Stability',
                    'Model_Period_Stability', 'Swell_Stability',
                    'Direction_Stability']
    n_expected = getattr(selector, 'n_features_in_', len(required))
    if len(required) != n_expected:
        raise SystemExit(
            f"[S7] Feature-count mismatch: selector expects {n_expected} "
            f"features but the feature list has {len(required)}.\n"
            f"[S7] HINT: re-run Stage 6 with the current code so the saved "
            f"model/selector and Training_Features.json are regenerated "
            f"consistently (S6_classification = True).")

    ds = xr.open_dataset(CONFIG["LOCAL_NC_FILE"])
    nc_vars = sorted(ds.data_vars)
    print(f"[S7] Loaded NetCDF: {CONFIG['LOCAL_NC_FILE']}")
    print(f"[S7] Variables present in file ({len(nc_vars)}): {nc_vars}")
    missing_upfront = [c for c in required if c not in nc_vars]
    if missing_upfront:
        print(f"[S7] WARNING: {len(missing_upfront)} training features are "
              f"NOT variables in this NetCDF: {missing_upfront}")
        print("[S7]   If any of these look like they SHOULD be present "
              "under a different spelling/case, check the list above and "
              "rename before re-running (median-imputation below is a "
              "fallback, not a substitute for the real signal).")
    df_full = calculate_engineering_features_grid(ds)
    df_full = add_bathymetry_grid(df_full, CONFIG["PATH_BATHY"])

    # ------------------------------------------------------------------
    # Variables absent from the regional NetCDF are held at their
    # TRAINING MEDIANS (not 0). Zero (e.g. a 0-second wave period) is far
    # outside the training distribution and previously silenced the
    # classifier at deployment; median-holding keeps the model
    # in-distribution while the spatial signal comes from the variables
    # that ARE present in the product.
    # ------------------------------------------------------------------
    med_json = pathlib.Path(CONFIG["CLASSIF_OUT"]) / \
        'Training_Feature_Medians.json'
    if med_json.exists():
        with open(med_json) as fh:
            train_medians = json.load(fh)
    else:  # compute on-the-fly from the balanced training dataset
        bal_csv = pathlib.Path(CONFIG["FEATURE_OUT"]) / \
            'Final_Balanced_Data_with_Features_Rolling.csv'
        train_medians = {}
        if bal_csv.exists():
            df_bal = pd.read_csv(bal_csv)
            train_medians = {c: float(df_bal[c].median())
                             for c in required if c in df_bal.columns}
            with open(med_json, 'w') as fh:
                json.dump(train_medians, fh, indent=2)
    imputed_log = []
    for col in set(required) - set(df_full.columns):
        val = float(train_medians.get(col, 0.0))
        df_full[col] = val
        imputed_log.append((col, val))
        print(f"[S7]  missing grid variable {col} -> held at training "
              f"median {val:.4f}")
    if imputed_log:
        save_text(OUTPUT_DIR, "Grid_Missing_Variables_Imputation.txt",
                  "GRID VARIABLES ABSENT FROM THE REGIONAL NETCDF\n"
                  + "=" * 56 + "\n"
                  "These features are held constant at their TRAINING\n"
                  "MEDIANS during spatial prediction (documented for the\n"
                  "Methods section):\n\n"
                  + "\n".join(f"  {c:<24s} = {v:.4f}"
                              for c, v in sorted(imputed_log)))
    df_full['time'] = pd.to_datetime(df_full['time'])

    print("[S7] Predicting regimes over the full grid ...")
    # [REV-R1-C5, non-circular false-alarm design] the false-alarm baseline
    # is now computed as the model's background Storm-classification rate
    # over every timestamp that is NOT within EXCLUSION_BUFFER_DAYS of one
    # of the four independently-documented, citable named storms. This adds
    # zero extra model calls (reuses the predictions already computed for
    # the main grid pass) and is not circular: the exclusion mask is built
    # purely from the storms' historical dates, never from any wave
    # variable the classifier itself consumes.
    t_min = df_full['time'].min()
    block_bucket = (lambda t: t_min + pd.Timedelta(
        days=WINDOW_DAYS * ((t - t_min).days // WINDOW_DAYS)))
    grid_records, monthly_acc, block_acc = [], {}, {}
    grouped = df_full.groupby(['latitude', 'longitude'])
    for i, ((lat, lon), grp) in enumerate(grouped, 1):
        if i % 50 == 0:
            print(f"[S7]   location {i}/{len(grouped)}")
        X_loc = grp[required]  # DataFrame, exact training column order
        X_loc_scaled = preprocessor.transform(X_loc)   # [REV-BUGFIX]
        X_sel = selector.transform(X_loc_scaled)
        preds = model.predict(X_sel)
        probs = model.predict_proba(X_sel)
        counts = {k: int((preds == k).sum()) for k in range(4)}
        times = pd.to_datetime(grp['time'].values)
        event_mask = np.zeros(len(times), dtype=bool)
        for ev in STORM_EVENTS.values():
            buf = pd.Timedelta(days=EXCLUSION_BUFFER_DAYS)
            event_mask |= ((times >= ev['start'] - buf)
                           & (times <= ev['end'] + buf))
        ne_preds = preds[~event_mask]
        ne_total = len(ne_preds)
        ne_storm = int((ne_preds == STORM_CLASS).sum())
        grid_records.append({
            'latitude': lat, 'longitude': lon,
            'Depth_CMEMS': grp['Depth_CMEMS'].iloc[0],
            'Total_Count': len(preds),
            'Swell_Count': counts[3], 'Storm_Count': counts[1],
            'Calm_Count': counts[2], 'Confused_Count': counts[0],
            'Avg_Confidence': float(np.max(probs, axis=1).mean()),
            'NonEvent_Total_Count': ne_total,
            'NonEvent_Storm_Count': ne_storm})
        for t, p in zip(times, preds):
            mk = t.strftime('%Y-%m')
            d = monthly_acc.setdefault(mk, {'Total': 0, 'Calm': 0, 'Swell': 0,
                                            'Storm': 0, 'Confused': 0})
            d['Total'] += 1
            d[{2: 'Calm', 3: 'Swell', 1: 'Storm', 0: 'Confused'}[int(p)]] += 1
        for t, p in zip(times[~event_mask], ne_preds):
            bk = block_bucket(t)
            bd = block_acc.setdefault(bk, {'Total': 0, 'Storm': 0})
            bd['Total'] += 1
            bd['Storm'] += int(p == STORM_CLASS)
        if i % 100 == 0:
            gc.collect()

    grid_stats = pd.DataFrame(grid_records)
    grid_stats['Swell_Potential'] = (grid_stats['Swell_Count']
                                     / grid_stats['Total_Count'] * 100)
    grid_stats['Storm_Risk'] = (grid_stats['Storm_Count']
                                / grid_stats['Total_Count'] * 100)
    grid_stats['NonEvent_FalseAlarm_Rate_%'] = (
        grid_stats['NonEvent_Storm_Count']
        / grid_stats['NonEvent_Total_Count'].replace(0, np.nan) * 100)
    grid_stats.to_csv(OUTPUT_DIR / "Oman_Spatial_Analysis_FULL.csv",
                      index=False)

    monthly = []
    for mk, d in sorted(monthly_acc.items()):
        monthly.append({'month_key': mk, **d,
                        **{f'{k}_Pct': d[k] / d['Total'] * 100
                           for k in ['Calm', 'Swell', 'Storm', 'Confused']}})
    monthly_stats = pd.DataFrame(monthly)
    # ------------------------------------------------------------------
    # [REV-R1-C5] SEASONAL (MONSOON) CLIMATOLOGY VALIDATION
    # Independent, citable physical check on the off-cyclone Storm-regime
    # incidence: the Arabian Sea is dominated by the SOUTHWEST (summer)
    # MONSOON, roughly June-September, which is well documented to produce
    # the region's most energetic sea states. If the model's Storm-regime
    # incidence peaks in those months, the background incidence quantified
    # above is largely GENUINE monsoon forcing rather than false alarms.
    # This is verifiable against the published monsoon literature, unlike
    # asserting that particular calendar windows were "calm".
    # ------------------------------------------------------------------
    if not monthly_stats.empty:
        monthly_stats.to_csv(OUTPUT_DIR / "Monthly_Regime_Statistics.csv",
                             index=False)
        ms = monthly_stats.copy()
        ms['month'] = pd.to_datetime(ms['month_key']).dt.month
        clim = (ms.groupby('month')[['Calm_Pct', 'Swell_Pct', 'Storm_Pct',
                                     'Confused_Pct']]
                .mean().reset_index())
        clim['Month_Name'] = clim['month'].map(
            {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
        clim.to_csv(rev_dir("R1C5_Storm_And_FalseAlarm_Validation")
                    / "Monthly_Climatology_Regime_Incidence.csv", index=False)
        sw_monsoon = [6, 7, 8, 9]
        sw = clim[clim['month'].isin(sw_monsoon)]['Storm_Pct'].mean()
        non_sw = clim[~clim['month'].isin(sw_monsoon)]['Storm_Pct'].mean()
        ratio = sw / non_sw if non_sw > 0 else np.inf
        peak_month = clim.loc[clim['Storm_Pct'].idxmax()]
        consistent = peak_month['month'] in sw_monsoon
        save_text(rev_dir("R1C5_Storm_And_FalseAlarm_Validation"),
                  "Monsoon_Seasonality_Validation.txt",
                  "SEASONAL (MONSOON) VALIDATION OF STORM-REGIME INCIDENCE\n"
                  "  [REV-R1-C5]\n" + "=" * 66 + "\n"
                  "Physical rationale: the Arabian Sea / Gulf of Oman is\n"
                  "dominated by the SOUTHWEST (summer) MONSOON, ~June-\n"
                  "September, which is well documented as the season of the\n"
                  "region's most energetic sea states. If the model's\n"
                  "Storm-regime incidence OUTSIDE named cyclones peaks in\n"
                  "those months, that incidence is largely genuine monsoon\n"
                  "forcing rather than spurious false alarms. This test is\n"
                  "checkable against the published monsoon literature -- it\n"
                  "does not require asserting that any particular calendar\n"
                  "window was 'calm'.\n\n"
                  "Mean Storm-regime incidence by calendar month (%):\n"
                  + clim[['Month_Name', 'Storm_Pct', 'Swell_Pct',
                          'Calm_Pct', 'Confused_Pct']].to_string(index=False)
                  + f"\n\nSW-monsoon months (Jun-Sep) mean Storm%: {sw:.2f}\n"
                  f"Other months mean Storm%:                {non_sw:.2f}\n"
                  f"Monsoon / non-monsoon ratio:             {ratio:.2f}x\n"
                  f"Peak Storm month: {peak_month['Month_Name']} "
                  f"({peak_month['Storm_Pct']:.2f}%)\n\n"
                  "VERDICT: " + ("CONSISTENT with SW-monsoon forcing -- the "
                                 "off-cyclone Storm\nincidence follows the "
                                 "documented monsoon cycle, supporting a\n"
                                 "physical (not spurious) origin."
                                 if consistent else
                                 "NOT aligned with the SW monsoon peak -- "
                                 "the off-cyclone Storm\nincidence needs "
                                 "further scrutiny in the Discussion."))
        print(f"[S7]  Monsoon seasonality check: Jun-Sep Storm "
              f"{sw:.2f}% vs other {non_sw:.2f}% ({ratio:.2f}x), peak = "
              f"{peak_month['Month_Name']} -> "
              f"{'CONSISTENT' if consistent else 'NOT CONSISTENT'}")
        # climatology figure
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['#e74c3c' if m in sw_monsoon else '#95a5a6'
                  for m in clim['month']]
        ax.bar(clim['Month_Name'], clim['Storm_Pct'], color=colors,
               edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Month', fontweight='bold', fontsize=26)
        ax.set_ylabel('Storm-regime incidence (%)', fontweight='bold',
                      fontsize=26)
        ax.tick_params(labelsize=20)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(handles=[
            mpatches.Patch(facecolor='#e74c3c', edgecolor='black',
                           label='SW monsoon (Jun-Sep)'),
            mpatches.Patch(facecolor='#95a5a6', edgecolor='black',
                           label='Other months')],
            fontsize=20, frameon=True, edgecolor='black')
        save_fig(fig, rev_dir("R1C5_Storm_And_FalseAlarm_Validation"),
                 "Monsoon_Seasonality_StormIncidence")

    # ------------------------------------------------------------------
    # [REV-R1-C6] TRUE PARETO FRONT with explicit depth AND Oman-waters
    # constraints. Depth alone is NOT sufficient: the grid domain also
    # covers the Persian Gulf and the Iranian Makran coast, whose shallow
    # cells are not valid candidates for an Oman siting study.
    # ------------------------------------------------------------------
    pareto_dir = rev_dir("R1C6_Pareto_Site_Selection")
    grid_stats['Dist_To_Oman_Coast_km'] = distance_to_oman_coast_km(
        grid_stats['latitude'].values, grid_stats['longitude'].values)
    grid_stats['Dist_To_Foreign_Coast_km'] = distance_to_foreign_coast_km(
        grid_stats['latitude'].values, grid_stats['longitude'].values)
    grid_stats['In_Oman_Waters'] = (
        (grid_stats['Dist_To_Oman_Coast_km'] <= MAX_DIST_TO_OMAN_COAST_KM)
        & (grid_stats['Dist_To_Oman_Coast_km']
           < grid_stats['Dist_To_Foreign_Coast_km']))
    grid_stats.to_csv(OUTPUT_DIR / "Oman_Spatial_Analysis_FULL.csv",
                      index=False)   # rewrite with the new mask columns
    depth_ok = grid_stats['Depth_CMEMS'] <= 100
    n_depth_only = int(depth_ok.sum())
    coastal = grid_stats[depth_ok & grid_stats['In_Oman_Waters']].copy()
    print(f"[S7] Site-selection candidates: {n_depth_only} cells pass depth "
          f"<=100 m; {len(coastal)} of those are also within "
          f"{MAX_DIST_TO_OMAN_COAST_KM:.0f} km of the Omani coast "
          f"({n_depth_only - len(coastal)} shallow cells excluded as "
          f"non-Omani waters).")
    save_text(pareto_dir, "Oman_Waters_Mask_Report.txt",
              "OMANI COASTAL-WATERS MASK FOR SITE SELECTION  [REV-R1-C6]\n"
              + "=" * 66 + "\n"
              "The regional grid domain spans 52-60 E / 16-26.4 N, which\n"
              "also covers the Persian Gulf (UAE/Qatar) and the Iranian\n"
              "Makran coast. Filtering on water depth ALONE admits shallow\n"
              "cells outside Omani waters (e.g. 25.4 N / 59.6 E off\n"
              "Chabahar, Iran; 24.0 N / 53.6 E off Abu Dhabi), which are\n"
              "not valid candidates for an Oman WEC siting study.\n\n"
              "Candidate cells must therefore satisfy ALL of:\n"
              "  (1) water depth <= 100 m, and\n"
              f"  (2) within {MAX_DIST_TO_OMAN_COAST_KM:.0f} km of the "
              "Omani coastline\n"
              "      (waypoint polyline: Musandam -> Batinah -> Muscat ->\n"
              "       Ra's al Hadd -> Masirah -> Duqm -> Shuwaymiyah ->\n"
              "       Salalah -> Yemen border), and\n"
              "  (3) closer to the Omani coast than to any neighbouring\n"
              "      coast (Iran, UAE, Qatar, Yemen) -- i.e. on the Omani\n"
              "      side of an approximate maritime median line.\n\n"
              f"Cells passing depth only:            {n_depth_only}\n"
              f"Cells passing depth AND Oman mask:   {len(coastal)}\n"
              f"Excluded as non-Omani waters:        "
              f"{n_depth_only - len(coastal)}\n")
    best_site = None
    if not coastal.empty:
        coastal['Pareto_Optimal'] = pareto_front(coastal, 'Swell_Potential',
                                                 'Storm_Risk')
        front = coastal[coastal['Pareto_Optimal']].sort_values(
            'Swell_Potential', ascending=False)
        front.to_csv(pareto_dir / "Pareto_Front_Coastal_Cells.csv",
                     index=False)
        best_site = front.iloc[0]  # highest swell on the front
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.scatterplot(data=coastal, x='Storm_Risk', y='Swell_Potential',
                        hue='Avg_Confidence', size='Avg_Confidence',
                        sizes=(50, 400), palette='plasma', alpha=0.85,
                        edgecolor='w', linewidth=0.5, ax=ax, legend='brief')
        fr = front.sort_values('Storm_Risk')
        ax.plot(fr['Storm_Risk'], fr['Swell_Potential'], color='black',
                linewidth=3, linestyle='--', marker='D', markersize=12,
                label='Pareto front (non-dominated)')
        ax.scatter(best_site['Storm_Risk'], best_site['Swell_Potential'],
                   color='cyan', marker='*', s=700, edgecolors='black',
                   linewidth=2.5, zorder=10, label='Selected Pareto site')
        ax.text(best_site['Storm_Risk'] + 0.15, best_site['Swell_Potential'],
                f" Lat: {best_site['latitude']}\n Lon: {best_site['longitude']}"
                f"\n Depth: {best_site['Depth_CMEMS']:.0f} m",
                fontsize=20, fontweight='bold', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                          lw=1.5, alpha=0.9))
        ax.set_xlabel('Storm Risk (% of time) - lower is better',
                      fontweight='bold', fontsize=26)
        ax.set_ylabel('Swell Potential (% of time) - higher is better',
                      fontweight='bold', fontsize=26)
        ax.grid(True, linestyle='--', alpha=0.5)
        # [REV-FIGFIX] the legend previously sat outside the axes at
        # bbox_to_anchor=(1.05, 1) and was cropped on export. It is now
        # placed inside the empty upper-left region of the trade-off cloud,
        # with explicit headroom reserved above the data.
        ax.set_ylim(0, max(coastal['Swell_Potential'].max() * 1.30, 30))
        ax.legend(loc='upper left', frameon=True, edgecolor='black',
                  fontsize=16, framealpha=0.92, ncol=1)
        fig.subplots_adjust(top=0.96, right=0.97, left=0.11, bottom=0.11)
        save_fig(fig, pareto_dir, "Pareto_Front_Site_Selection_Matrix")
        save_text(pareto_dir, "Pareto_Site_Selection_Report.txt",
                  "BI-OBJECTIVE PARETO SITE SELECTION  [REV-R1-C6 / R2-M6]\n"
                  + "=" * 66 + "\n"
                  "Objectives: maximize Swell Potential, minimize Storm "
                  "Risk.\nHard constraints: (1) water depth <= 100 m "
                  "(coastal deployability)\n"
                  f"                  (2) within "
                  f"{MAX_DIST_TO_OMAN_COAST_KM:.0f} km of the Omani "
                  "coastline\n                      (excludes Persian-Gulf "
                  "and Iranian-Makran\n                      shallow cells "
                  "inside the grid domain -- see\n                      "
                  "Oman_Waters_Mask_Report.txt)\n"
                  f"Non-dominated Omani coastal cells: {len(front)}\n\n"
                  "Selected Pareto-optimal site (max swell on the front):\n"
                  f"  Lat/Lon:          {best_site['latitude']}, "
                  f"{best_site['longitude']}\n"
                  f"  Depth:            {best_site['Depth_CMEMS']:.1f} m\n"
                  f"  Dist. to Oman coast: "
                  f"{best_site['Dist_To_Oman_Coast_km']:.1f} km\n"
                  f"  Swell Potential:  {best_site['Swell_Potential']:.2f} %\n"
                  f"  Storm Risk:       {best_site['Storm_Risk']:.2f} %\n"
                  f"  Model Confidence: {best_site['Avg_Confidence']:.3f}\n\n"
                  "EXPLICIT DECISION-SUPPORT CAVEATS (for the manuscript):\n"
                  "This ranking is a DECISION-SUPPORT tool, not a final\n"
                  "siting decision. Practical WEC deployment additionally\n"
                  "depends on WEC-specific power matrices and extreme loads,\n"
                  "seabed/geotechnical conditions, marine protected areas,\n"
                  "navigation routes, grid accessibility, installation and\n"
                  "O&M costs — none of which are modeled here. The full\n"
                  "Pareto front is provided so planners can weigh these\n"
                  "unmodeled criteria across all non-dominated candidates.\n\n"
                  "Full front:\n" + front.to_string(index=False))

    # main spatial figures (unchanged, star = Pareto site)
    plot_spatial_heatmap(grid_stats, 'Swell_Potential', 'Swell Potential (%)',
                         OUTPUT_DIR, "Enhanced_Map_Swell", cmap='viridis',
                         star_point=best_site)
    plot_spatial_heatmap(grid_stats, 'Storm_Risk', 'Storm Risk (%)',
                         OUTPUT_DIR, "Enhanced_Map_Storm", cmap='magma')
    plot_spatial_heatmap(grid_stats, 'Storm_Risk', 'Storm Risk (%)',
                         OUTPUT_DIR, "Enhanced_Map_Storm_LogScale",
                         cmap='magma', use_log=True)
    plot_spatial_heatmap(grid_stats, 'Avg_Confidence',
                         'Average Model Confidence', OUTPUT_DIR,
                         "Enhanced_Map_Confidence", cmap='cividis')
    for name, col, cmapn in [('Swell', 'Swell_Count', 'viridis'),
                             ('Storm', 'Storm_Count', 'magma'),
                             ('Calm', 'Calm_Count', 'cividis'),
                             ('Confused', 'Confused_Count', 'plasma')]:
        grid_stats[f'{name}_Freq'] = (grid_stats[col]
                                      / grid_stats['Total_Count'] * 100)
        plot_spatial_heatmap(grid_stats, f'{name}_Freq',
                             f'{name} Frequency (%)', OUTPUT_DIR,
                             f"Validation_Map_{name}", cmap=cmapn)

    # regime frequency + temporal trend figures (unchanged)
    totals = {'Calm': grid_stats['Calm_Count'].sum(),
              'Swell': grid_stats['Swell_Count'].sum(),
              'Storm': grid_stats['Storm_Count'].sum(),
              'Confused': grid_stats['Confused_Count'].sum()}
    stats_df = pd.DataFrame(list(totals.items()), columns=['Regime', 'Count'])
    stats_df['Percentage'] = stats_df['Count'] / stats_df['Count'].sum() * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    _freq_colors = {'Calm': regime_color(2), 'Swell': regime_color(3),
                    'Storm': regime_color(1), 'Confused': regime_color(0)}
    bars = sns.barplot(data=stats_df, x='Regime', y='Percentage',
                       palette=[_freq_colors.get(r, '#7F7F7F')
                                for r in stats_df['Regime']],
                       hue='Regime', legend=False, ax=ax)
    for b in bars.containers:
        ax.bar_label(b, fmt='%.1f%%', padding=5, fontsize=24,
                     fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontweight='bold', fontsize=28)
    ax.set_xlabel('Wave Regime', fontweight='bold', fontsize=28)
    ax.set_ylim(0, 100)
    save_fig(fig, OUTPUT_DIR, "Stat_Regime_Frequency")

    if not monthly_stats.empty:
        dates = pd.to_datetime(monthly_stats['month_key'])
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.stackplot(dates, monthly_stats['Calm_Pct'],
                     monthly_stats['Swell_Pct'], monthly_stats['Storm_Pct'],
                     monthly_stats['Confused_Pct'],
                     labels=['Ambient/Calm', 'Golden Swell', 'Storm',
                             'Confused Sea'],
                     colors=[regime_color(2), regime_color(3),
                             regime_color(1), regime_color(0)], alpha=0.92)
        ax.set_ylabel('Regime Prevalence (%)', fontweight='bold', fontsize=28)
        ax.set_xlabel('Date', fontweight='bold', fontsize=28)
        ax.set_xlim(dates.min(), dates.max())
        ax.set_ylim(0, 100)
        # [REV-FIGFIX] legend moved from outside the axes (which was being
        # cropped) to a horizontal strip beneath the panel.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4,
                  frameon=True, edgecolor='black', fontsize=22)
        fig.subplots_adjust(top=0.95, bottom=0.30, left=0.08, right=0.98)
        save_fig(fig, OUTPUT_DIR, "Validation_Temporal_Trend")

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(dates, monthly_stats['Storm_Pct'], label='Storm Risk',
                linewidth=4.0, color='red', marker='o', markersize=10,
                markevery=6)
        ax.plot(dates, monthly_stats['Swell_Pct'], label='Swell Potential',
                linewidth=4.0, color='green', marker='s', markersize=10,
                markevery=6)
        ax.fill_between(dates, monthly_stats['Storm_Pct'], alpha=0.2,
                        color='red')
        ax.fill_between(dates, monthly_stats['Swell_Pct'], alpha=0.2,
                        color='green')
        ax.set_xlabel('Date', fontweight='bold', fontsize=30)
        ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=30)
        ax.legend(fontsize=28, loc='upper left', frameon=True,
                  edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.5)
        save_fig(fig, OUTPUT_DIR, "Fig10_Model_Evolution_Comparison")

    # ------------------------------------------------------------------
    # [REV-R1-C5] STORM + NON-STORM VALIDATION (independent events and
    #             false-alarm quantification)
    # ------------------------------------------------------------------
    st_dir = rev_dir("R1C5_Storm_And_FalseAlarm_Validation")
    region_masks = {r: fn(grid_stats['latitude']) for r, fn in
                    REGION_BOUNDS.items()}
    # [REV-R1-C5, unified] background rate EXCLUDES the storm-event samples
    # themselves (NonEvent_* columns), so the same, single background
    # definition is used both for the cyclone anomaly ratio below AND for
    # the false-alarm baseline further down -- avoids a storm's own days
    # diluting its own background, and keeps both quantities consistent.
    def _region_nonevent_rate(mask):
        t_ = grid_stats.loc[mask, 'NonEvent_Total_Count'].sum()
        s_ = grid_stats.loc[mask, 'NonEvent_Storm_Count'].sum()
        return (s_ / t_ * 100) if t_ else 0.001
    bg_risk = {r: _region_nonevent_rate(m) for r, m in region_masks.items()}
    bg_risk['all'] = (grid_stats['NonEvent_Storm_Count'].sum()
                      / grid_stats['NonEvent_Total_Count'].sum() * 100)

    val_rows = []
    for name, ev in STORM_EVENTS.items():
        sdf = storm_window_stats(df_full, model, selector, required,
                                 ev['start'], ev['end'], preprocessor)
        if sdf.empty:
            val_rows.append({'Event': name, 'Type': 'Storm',
                             'Region': ev['region'], 'Status':
                             'NO DATA IN WINDOW'})
            save_text(st_dir, f"Validation_Report_{name}.txt",
                      f"VALIDATION REPORT: CYCLONE {name.upper()}\n"
                      + "=" * 50 + "\nStatus: N/A (no data in the loaded "
                      "NetCDF for this window)")
            continue
        reg = ev['region']
        zone = sdf[REGION_BOUNDS[reg](sdf['latitude'])] \
            if reg in REGION_BOUNDS else sdf
        if zone.empty:
            zone = sdf
        max_risk = zone['Storm_Risk'].max()
        bg = max(bg_risk.get(reg, 0.001), 0.001)
        ar = max_risk / bg
        detected = max_risk > 10.0 or ar > 10.0
        val_rows.append({'Event': name, 'Type': 'Storm', 'Region': reg,
                         'Window': f"{ev['start']:%Y-%m-%d} - "
                                   f"{ev['end']:%Y-%m-%d}",
                         'Max_Storm_Prob_%': round(max_risk, 2),
                         'Background_Risk_%': round(bg, 3),
                         'Anomaly_Ratio': round(ar, 1),
                         'Status': 'DETECTED' if detected else 'MISSED'})
        save_text(st_dir, f"Validation_Report_{name}.txt",
                  f"VALIDATION REPORT: CYCLONE {name.upper()}\n" + "=" * 50 +
                  f"\nWindow: {ev['start']} .. {ev['end']}"
                  f"\nImpact region: {reg}"
                  f"\nMax storm probability in event window: {max_risk:.2f}%"
                  f"\nClimatological background risk ({reg}): {bg:.3f}%"
                  f"\nAnomaly ratio: {ar:.1f}x"
                  f"\nStatus: "
                  + ('SUCCESS (significant anomaly detected)'
                     if detected else 'FAILED'))
        # per-event map
        fig, ax = plt.subplots(figsize=(14, 10))
        piv = sdf.pivot(index='latitude', columns='longitude',
                        values='Storm_Risk')
        sns.heatmap(piv, cmap='inferno', vmin=0, vmax=100,
                    cbar_kws={'label': 'Storm Probability (%)',
                              'shrink': 0.8}, ax=ax)
        ax.invert_yaxis()
        format_spatial_axis(ax, piv)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)
        cbar.set_label('Storm Probability (%)', fontsize=24,
                       fontweight='bold', labelpad=20)
        save_fig(fig, st_dir, f"Validation_Cyclone_{name}")

    # ------------------------------------------------------------------
    # [REV-R1-C5, non-circular false-alarm design] FALSE-ALARM BASELINE
    # Rather than asserting particular calendar windows were "calm" (which
    # would need an external citation), or ranking CMEMS wave height itself
    # to find low-Hs windows (circular — VHM0 is a training feature that
    # drives the Storm/Calm split), the false-alarm rate is computed as the
    # model's background Storm-classification rate over EVERY timestamp
    # that is NOT within EXCLUSION_BUFFER_DAYS of one of the four
    # independently-documented named storms. This uses predictions already
    # computed in the main grid pass above (grid_stats' NonEvent_* columns
    # and the block_acc accumulator) — no extra model calls, and no
    # dependency on any wave-derived variable for the exclusion itself.
    # ------------------------------------------------------------------
    ne_total_all = grid_stats['NonEvent_Total_Count'].sum()
    ne_storm_all = grid_stats['NonEvent_Storm_Count'].sum()
    overall_fa_rate = bg_risk['all']
    regional_fa = {r: v for r, v in bg_risk.items() if r != 'all'}
    for r, v in regional_fa.items():
        val_rows.append({'Event': f'NonEvent_Background_{r}',
                         'Type': 'Storm-regime incidence baseline',
                         'Region': r,
                         'Mean_Storm_Prob_%': round(v, 3),
                         'Status': 'BASELINE'})
    val_rows.append({'Event': 'NonEvent_Background_ALL',
                     'Type': 'Storm-regime incidence baseline',
                     'Region': 'all',
                     'Mean_Storm_Prob_%': round(overall_fa_rate, 3),
                     'Status': 'BASELINE'})

    block_df = pd.DataFrame([
        {'block_start': bk, 'block_end': bk + pd.Timedelta(days=WINDOW_DAYS),
         'Total': d['Total'], 'Storm': d['Storm'],
         'Storm_Pct': d['Storm'] / d['Total'] * 100 if d['Total'] else np.nan}
        for bk, d in block_acc.items()]).sort_values('block_start')
    block_df.to_csv(st_dir / "NonEvent_Block_FalseAlarm_Rates.csv",
                    index=False)
    worst = block_df.loc[block_df['Storm_Pct'].idxmax()] \
        if not block_df.empty else None
    if worst is not None:
        _wm = pd.Timestamp(worst['block_start']).month
        _in_monsoon = _wm in (6, 7, 8, 9)
        val_rows.append({
            'Event': 'NonEvent_MaxIncidence_Window',
            'Type': 'Storm-regime incidence (max non-event block)',
            'Region': 'all',
            'Window': f"{worst['block_start']:%Y-%m-%d} - "
                      f"{worst['block_end']:%Y-%m-%d}",
            'Mean_Storm_Prob_%': round(worst['Storm_Pct'], 3),
            'Status': ('HIGHEST NON-EVENT INCIDENCE (within SW-monsoon '
                       'season)' if _in_monsoon else
                       'HIGHEST NON-EVENT INCIDENCE (outside SW-monsoon '
                       'season)')})

    save_text(st_dir, "FalseAlarm_Baseline_Report.txt",
              "STORM-REGIME BACKGROUND INCIDENCE (NON-CIRCULAR BASELINE)\n"
              "  [REV-R1-C5]\n" + "=" * 64 +
              "\nMethod: Storm-classification rate computed over every "
              "grid-cell\ntimestep OUTSIDE a "
              f"{EXCLUSION_BUFFER_DAYS}-day buffer around each of the four\n"
              "independently-documented named storms (Shaheen, Mekunu, "
              "Hikaa,\nLuban). The exclusion uses only those storms' "
              "historical dates --\nnever any wave variable the classifier "
              "itself consumes -- so this\nbaseline cannot be circular "
              "(unlike ranking CMEMS wave height to\nfind 'calm' windows).\n"
              "\nIMPORTANT INTERPRETIVE CAVEAT: away from the ADCP anchor\n"
              "stations, no independent ground truth exists for these grid\n"
              "cells. This quantity should therefore be read as the "
              "model's\nStorm-regime PREDICTION incidence outside named "
              "cyclones -- not\na proven false-positive/error rate.\n"
              "\nSEE ALSO Monsoon_Seasonality_Validation.txt: the seasonal\n"
              "breakdown shows this incidence peaks sharply during the\n"
              "documented SW (summer) monsoon, indicating that most of it\n"
              "reflects genuine monsoon-driven energetic sea states rather\n"
              "than spurious detections. Use both alongside the LOSO/LORO\n"
              "spatial-generalization results (R1C2) when discussing\n"
              "reliability away from monitored sites.\n"
              f"\nNon-event samples analyzed: {int(ne_total_all):,} "
              f"grid-cell-timesteps\n\n"
              f"Overall background Storm rate (all regions): "
              f"{overall_fa_rate:.3f}%\n"
              + "\n".join(f"  {r}: {v:.3f}%" for r, v in regional_fa.items())
              + f"\n\n{WINDOW_DAYS}-day non-event block statistics "
              f"({len(block_df)} blocks):\n"
              + (f"  mean Storm%: {block_df['Storm_Pct'].mean():.3f}\n"
                 f"  median Storm%: {block_df['Storm_Pct'].median():.3f}\n"
                 f"  95th pct Storm%: "
                 f"{block_df['Storm_Pct'].quantile(0.95):.3f}\n"
                 f"  max Storm%: {block_df['Storm_Pct'].max():.3f}\n"
                 if not block_df.empty else "  (no blocks)\n")
              + (f"\nHighest single-block incidence rate found (worst "
                 f"case among\nnon-event blocks, NOT a proven false alarm): "
                 f"{worst['block_start']:%Y-%m-%d} - "
                 f"{worst['block_end']:%Y-%m-%d} at "
                 f"{worst['Storm_Pct']:.2f}%\n" if worst is not None else ""))


    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(st_dir / "Storm_And_FalseAlarm_Validation_Summary.csv",
                  index=False)
    save_text(st_dir, "Storm_And_FalseAlarm_Validation_Summary.txt",
              "STORM DETECTION + FALSE-ALARM VALIDATION  [REV-R1-C5]\n"
              + "=" * 64 + "\n"
              "Independent tropical-cyclone events: Shaheen (2021, north),\n"
              "Mekunu (2018, south), Hikaa (2019, central; NEW) and\n"
              "Luban (2018, south; NEW). The false-alarm baseline is the\n"
              "background Storm-classification rate over every timestamp\n"
              "OUTSIDE a buffer around those four documented storms (see\n"
              "FalseAlarm_Baseline_Report.txt) -- non-circular, since the\n"
              "exclusion uses only the storms' historical dates, never any\n"
              "wave variable the classifier itself consumes.\n\n"
              + val_df.to_string(index=False))

    # ---- master summary report (unchanged content, updated Pareto site) ----
    lines = ["=" * 56, "        OMAN WAVE ENERGY ANALYSIS REPORT", "=" * 56,
             f"Date Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "-" * 50]
    if best_site is not None:
        lines += ["", "[PARETO-OPTIMAL SITE SELECTION]",
                  f"Constraints: depth <= 100 m AND Omani waters "
                  f"(<= {MAX_DIST_TO_OMAN_COAST_KM:.0f} km from the Omani "
                  f"coast, on the Omani side of the median line)",
                  f"Location: Lat {best_site['latitude']}, "
                  f"Lon {best_site['longitude']}",
                  f"Depth: {best_site['Depth_CMEMS']:.2f} m",
                  f"Distance to Omani coast: "
                  f"{best_site['Dist_To_Oman_Coast_km']:.1f} km",
                  f"Swell Potential: {best_site['Swell_Potential']:.2f}%",
                  f"Storm Risk: {best_site['Storm_Risk']:.2f}%",
                  f"Model Confidence: {best_site['Avg_Confidence']:.2f}"]
    for r in ['north', 'central', 'south']:
        m = region_masks[r]
        if m.any():
            lines += [f"{r.upper()}: Avg Swell "
                      f"{grid_stats.loc[m, 'Swell_Potential'].mean():.2f}% | "
                      f"Avg Storm Risk {bg_risk[r]:.2f}%"]
    save_text(OUTPUT_DIR, "Analysis_Report_Summary.txt", "\n".join(lines))
    print("[S7] Spatial stage complete.")

# =============================================================================
# SECTION 8 — MASTER REVISION REPORT + MAIN
# =============================================================================
# =============================================================================
# SECTION 7.5 — STAGE 8: COLLECT *ALL* RESULTS INTO ONE BIG TEXT FILE
#   Purpose: a single self-contained MASTER_RESULTS_FOR_REVIEW.txt with every
#   number needed to check the results against the manuscript and reviewer
#   comments (all txt reports, all csv tables, json details, xlsx sheets, and
#   numeric summaries of the figure-source data). No recomputation — it only
#   reads what previous stages already saved to disk.
# =============================================================================
MAX_TXT_CHARS = 60_000        # cap per embedded text file
MAX_CSV_ROWS = 250            # full dump if <= this many rows
SKIP_FILE_PATTERNS = (        # bulky raw-data files: summarized, not dumped
    'Global_All_Data_Unfiltered', 'Global_Best_Matches_Merged',
    'full_clustered_data', 'Final_Balanced_Data_with_Features_Rolling',
    'Unfiltered_With_Deep_Labels', 'Oman_Spatial_Analysis_FULL',
)
SKIP_EXTS = ('.png', '.pdf', '.pkl', '.nc', '.h5', '.keras', '.zip')

def _fmt_header(title: str, ch: str = "=") -> str:
    return f"\n\n{ch * 100}\n### {title}\n{ch * 100}\n"

def _embed_txt(p: pathlib.Path) -> str:
    try:
        txt = p.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return f"[could not read: {e}]"
    if len(txt) > MAX_TXT_CHARS:
        txt = (txt[:MAX_TXT_CHARS]
               + f"\n... [TRUNCATED — file has {len(txt):,} chars total]")
    return txt

def _embed_csv(p: pathlib.Path) -> str:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return f"[could not read csv: {e}]"
    parts = [f"shape = {df.shape[0]} rows x {df.shape[1]} cols"]
    if len(df) <= MAX_CSV_ROWS:
        parts.append(df.to_string(index=False))
    else:
        parts.append(f"(large table -> first 120 rows, last 30 rows, and "
                     f"numeric summary)")
        parts.append(df.head(120).to_string(index=False))
        parts.append("...")
        parts.append(df.tail(30).to_string(index=False))
        num = df.select_dtypes(include=np.number)
        if not num.empty:
            parts.append("\nNUMERIC SUMMARY (describe):")
            parts.append(num.describe().T.to_string())
    return "\n".join(parts)

def _embed_json(p: pathlib.Path) -> str:
    try:
        with open(p, encoding='utf-8') as fh:
            obj = json.load(fh)
        txt = json.dumps(obj, indent=2, default=str)
        return txt[:MAX_TXT_CHARS]
    except Exception as e:
        return f"[could not read json: {e}]"

def _embed_xlsx(p: pathlib.Path) -> str:
    parts = []
    try:
        xls = pd.ExcelFile(p)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            parts.append(f"\n--- SHEET: {sheet}  "
                         f"({df.shape[0]} rows x {df.shape[1]} cols) ---")
            if len(df) <= MAX_CSV_ROWS:
                parts.append(df.to_string(index=False))
            else:
                parts.append(df.head(120).to_string(index=False))
                parts.append(f"... [{len(df)} rows total]")
    except Exception as e:
        parts.append(f"[could not read xlsx: {e}]")
    return "\n".join(parts)

def _embed_any(p: pathlib.Path) -> str:
    s = p.suffix.lower()
    if s == '.csv':
        return _embed_csv(p)
    if s == '.json':
        return _embed_json(p)
    if s in ('.xlsx', '.xls'):
        return _embed_xlsx(p)
    return _embed_txt(p)

def _summarize_big_dataset(p: pathlib.Path) -> str:
    """Numeric-only summary for the bulky raw data files."""
    try:
        if p.suffix.lower() == '.csv':
            df = pd.read_csv(p)
        else:
            df = pd.read_excel(p)
    except Exception as e:
        return f"[could not read: {e}]"
    parts = [f"shape = {df.shape[0]} rows x {df.shape[1]} cols",
             "columns: " + ", ".join(map(str, df.columns))]
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        parts.append("\nNUMERIC SUMMARY (describe):")
        parts.append(num.describe().T.to_string())
    # class/label distributions if present
    for lab in ('Deep_Regime', 'GMM_Regime', 'KMeans_Regime', 'Season_Name',
                'Base_Station'):
        if lab in df.columns:
            vc = df[lab].value_counts()
            parts.append(f"\nVALUE COUNTS — {lab}:")
            parts.append(vc.to_string())
    return "\n".join(parts)

def stage8_collect_master_report():
    rev_root = pathlib.Path(CONFIG["REVISION_DIR"])
    out_path = rev_root / "MASTER_RESULTS_FOR_REVIEW.txt"
    chunks: List[str] = []
    included: set = set()

    chunks.append(
        "#" * 100 + "\n"
        "# MASTER RESULTS FILE FOR MANUSCRIPT-REVISION REVIEW\n"
        "# Manuscript OE-D-26-08011 — unified revision pipeline outputs\n"
        f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "# Contents: every txt/csv/json/xlsx result produced by Stages 3-7\n"
        "# plus numeric summaries of the figure-source datasets, grouped by\n"
        "# reviewer comment. Figures themselves are on disk as PNG/PDF; the\n"
        "# numbers behind them are embedded here.\n"
        + "#" * 100)

    def add_file(p: pathlib.Path, title: Optional[str] = None):
        if not p.exists() or str(p) in included:
            return
        included.add(str(p))
        chunks.append(_fmt_header(title or str(p)))
        chunks.append(f"[source: {p}]\n")
        chunks.append(_embed_any(p))

    def add_summary(p: pathlib.Path, title: str):
        if not p.exists() or str(p) in included:
            return
        included.add(str(p))
        chunks.append(_fmt_header(title))
        chunks.append(f"[source: {p} — summarized, raw table too large]\n")
        chunks.append(_summarize_big_dataset(p))

    S3 = pathlib.Path(CONFIG["MERGE_OUT"])
    S4 = pathlib.Path(CONFIG["FEATURE_OUT"])
    S5 = pathlib.Path(CONFIG["CLUSTER_OUT"])
    S6 = pathlib.Path(CONFIG["CLASSIF_OUT"])
    S7 = pathlib.Path(CONFIG["MAPS_OUT"])

    # ---- 0. environment / reproducibility -------------------------------
    chunks.append(_fmt_header("SECTION 0 — ENVIRONMENT & REPRODUCIBILITY "
                              "[R2 reproducibility]", "#"))
    add_file(rev_root / "Reproducibility" / "Environment_And_Timing_Report.txt")

    # ---- 1. fidelity gate (R1-C1 / R2-M2) --------------------------------
    chunks.append(_fmt_header("SECTION 1 — FIDELITY GATE + SENSITIVITY "
                              "[R1-C1 / R2-M2]", "#"))
    add_file(S3 / "Global_Best_Matches_Report.txt",
             "Station PASS/REJECT report (gate 0.80 / 0.50 m)")
    add_file(S3 / "Station_Fidelity_Statistics.csv",
             "Per-station physical validation statistics (all four pairs)")
    for p in sorted(rev_dir("R1C1_FidelityGate_Sensitivity").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)

    # ---- 2. data & features ----------------------------------------------
    chunks.append(_fmt_header("SECTION 2 — TRAINING DATA SUMMARIES", "#"))
    add_summary(S4 / "Final_Balanced_Data_with_Features_Rolling.csv",
                "Balanced training dataset (numeric summary + counts)")
    add_summary(S3 / "Global_Best_Matches_Merged.xlsx",
                "Gate-passing merged dataset (numeric summary)")
    add_summary(S3 / "Global_All_Data_Unfiltered.xlsx",
                "Unfiltered merged dataset for the ablation (summary)")

    # ---- 3. clustering (R1-C4, R3 AE details) ----------------------------
    chunks.append(_fmt_header("SECTION 3 — CLUSTERING, k-SELECTION, "
                              "STABILITY [R1-C4 / R3-M1]", "#"))
    for p in sorted(S5.rglob("*")):
        if (p.is_file() and p.suffix.lower() not in SKIP_EXTS
                and not any(s in p.name for s in SKIP_FILE_PATTERNS)):
            add_file(p)
    for p in sorted(rev_dir("R1C4_Cluster_Stability").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)
    for p in sorted(rev_dir("R3C1_Model_Architecture_Hyperparams").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)
    big_clustered = S5 / "Deep Clustering" / "full_clustered_data.csv"
    add_summary(big_clustered, "Deep-clustered dataset (label distribution)")

    # ---- 4. classification core results ----------------------------------
    chunks.append(_fmt_header("SECTION 4 — CLASSIFICATION RESULTS, ABLATION, "
                              "ROC/RELIABILITY SOURCE NUMBERS", "#"))
    for p in sorted(S6.glob("*")):
        if (p.is_file() and p.suffix.lower() not in SKIP_EXTS
                and not any(s in p.name for s in SKIP_FILE_PATTERNS)):
            add_file(p)

    # ---- 5. leakage-safe validation (R1-C2 / R3-M2) ----------------------
    chunks.append(_fmt_header("SECTION 5 — BLOCK-WISE CV + LOSO/LORO "
                              "[R1-C2 / R3-M2]", "#"))
    for folder in ("R3C2_Blockwise_CV", "R1C2_LOSO_LORO_Validation"):
        for p in sorted(rev_dir(folder).glob("*")):
            if p.suffix.lower() not in SKIP_EXTS:
                add_file(p)

    # ---- 6. statistics (R2-M3) -------------------------------------------
    chunks.append(_fmt_header("SECTION 6 — 95% CIs, REPEATED SEEDS, "
                              "SIGNIFICANCE TESTS [R2-M3]", "#"))
    for p in sorted(rev_dir("R2C3_CIs_RepeatedValidation_Stats").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)

    # ---- 7. extra ensembles + combined tables (R3) -----------------------
    chunks.append(_fmt_header("SECTION 7 — EXTRA ENSEMBLES + COMBINED "
                              "COMPREHENSIVE TABLES [R3]", "#"))
    for p in sorted(rev_dir("R3C3_Extra_Ensembles").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)
    add_file(rev_dir("Combined_Tables") / "Comprehensive_Results_Tables.xlsx",
             "Comprehensive results workbook (ALL sheets, as text)")

    # ---- 8. spatial mapping, storms, site selection ----------------------
    chunks.append(_fmt_header("SECTION 8 — SPATIAL RESULTS, STORM + "
                              "FALSE-ALARM VALIDATION, PARETO SITE "
                              "[R1-C5 / R1-C6]", "#"))
    add_file(S7 / "Analysis_Report_Summary.txt")
    grid_csv = S7 / "Oman_Spatial_Analysis_FULL.csv"
    if grid_csv.exists():
        try:
            g = pd.read_csv(grid_csv)
            chunks.append(_fmt_header("Grid statistics — numeric summary + "
                                      "key extremes (source of the spatial "
                                      "heatmap figures)"))
            chunks.append(f"[source: {grid_csv}]\n")
            chunks.append(f"grid cells: {len(g)}")
            chunks.append("\nNUMERIC SUMMARY:")
            chunks.append(g.select_dtypes(include=np.number)
                           .describe().T.to_string())
            coastal = g[g['Depth_CMEMS'] <= 100]
            chunks.append(f"\ncoastal cells (depth<=100m): {len(coastal)}")
            chunks.append("\nTOP 15 COASTAL CELLS BY SWELL POTENTIAL:")
            chunks.append(coastal.nlargest(15, 'Swell_Potential')
                          .to_string(index=False))
            chunks.append("\nTOP 15 CELLS BY STORM RISK (whole grid):")
            chunks.append(g.nlargest(15, 'Storm_Risk').to_string(index=False))
            included.add(str(grid_csv))
        except Exception as e:
            chunks.append(f"[grid summary failed: {e}]")
    for p in sorted(rev_dir("R1C5_Storm_And_FalseAlarm_Validation").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)
    for p in sorted(rev_dir("R1C6_Pareto_Site_Selection").glob("*")):
        if p.suffix.lower() not in SKIP_EXTS:
            add_file(p)

    # ---- 9. sweep: anything else not yet included ------------------------
    chunks.append(_fmt_header("SECTION 9 — SWEEP: ALL REMAINING OUTPUT "
                              "FILES NOT LISTED ABOVE", "#"))
    for root in (rev_root, S3, S4, S5, S6, S7):
        if not pathlib.Path(root).exists():
            continue
        for p in sorted(pathlib.Path(root).rglob("*")):
            if (p.is_file() and str(p) not in included
                    and p.suffix.lower() not in SKIP_EXTS
                    and p.name != out_path.name):
                if any(s in p.name for s in SKIP_FILE_PATTERNS):
                    add_summary(p, f"(summary) {p.name}")
                else:
                    add_file(p)

    text = "\n".join(chunks)
    out_path.write_text(text, encoding='utf-8')
    print(f"[S8] MASTER report written: {out_path}")
    print(f"[S8]   size: {out_path.stat().st_size / 1e6:.2f} MB, "
          f"{len(included)} files embedded")

def write_master_revision_report():
    rd = pathlib.Path(CONFIG["REVISION_DIR"])
    lines = [
        "=" * 78,
        " MASTER REVISION-OUTPUT INDEX (auto-generated by the unified "
        "pipeline)",
        "=" * 78,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Reviewer comment -> revision sub-folder mapping:",
        "  R1-C1 / R2-M2  Fidelity-gate sensitivity      -> "
        "R1C1_FidelityGate_Sensitivity/",
        "  R1-C2 / R3     LOSO + LORO validation         -> "
        "R1C2_LOSO_LORO_Validation/",
        "  R1-C4          Cluster stability (seeds)      -> "
        "R1C4_Cluster_Stability/",
        "  R1-C5          Storms + false alarms          -> "
        "R1C5_Storm_And_FalseAlarm_Validation/",
        "  R1-C6 / R2-M6  Pareto front + caveats         -> "
        "R1C6_Pareto_Site_Selection/",
        "  R2-M3          CIs, repeated seeds, stats     -> "
        "R2C3_CIs_RepeatedValidation_Stats/",
        "  R3-M1 / R3-R2  AE details + all hyperparams   -> "
        "R3C1_Model_Architecture_Hyperparams/",
        "  R3-M2          Block-wise 10-fold CV          -> R3C2_Blockwise_CV/",
        "  R3-R2          Extra ensembles + tuning       -> "
        "R3C3_Extra_Ensembles/",
        "  Tables         Combined comprehensive tables  -> Combined_Tables/",
        "  Reproducibility                              -> Reproducibility/",
        "",
        "Files present:",
    ]
    for p in sorted(rd.rglob("*")):
        if p.is_file():
            lines.append("  " + str(p.relative_to(rd)))
    save_text(rd, "MASTER_REVISION_INDEX.txt", "\n".join(lines))

def main():
    ensure_dirs()
    _verify_times_new_roman()   # [REV] fail loudly if the font is missing
    t_all = time.time()
    if RUN_STAGES["S1_download"]:
        with StageTimer("S1_CMEMS_download"):
            stage1_cmems_download()
    if RUN_STAGES["S2_adcp_preprocess"]:
        with StageTimer("S2_ADCP_preprocessing"):
            stage2_adcp_preprocess()
    if RUN_STAGES["S3_merge_fidelity"]:
        with StageTimer("S3_Merge_FidelityGate_Sensitivity"):
            stage3_merge_fidelity()
    if RUN_STAGES["S4_features"]:
        with StageTimer("S4_Feature_Engineering_Balancing"):
            stage4_features()
    if RUN_STAGES["S5_clustering"]:
        with StageTimer("S5_Clustering_Stability"):
            stage5_clustering()
    if RUN_STAGES["S6_classification"]:
        with StageTimer("S6_Classification_Validation"):
            stage6_classification()
    if RUN_STAGES["S7_maps"]:
        with StageTimer("S7_Spatial_Maps_SiteSelection"):
            stage7_maps()
    # [REV-BUGFIX] The reproducibility log must be written BEFORE the master
    # collator runs, otherwise Section 0 of MASTER_RESULTS_FOR_REVIEW.txt is
    # empty on a clean run (the file it embeds does not exist yet). This is
    # why the environment section was blank in the last output.
    save_text(rev_dir("Reproducibility"), "Environment_And_Timing_Report.txt",
              get_environment_report())
    if RUN_STAGES.get("S8_master_report", False):
        with StageTimer("S8_Collect_Master_Report"):
            stage8_collect_master_report()
    write_master_revision_report()
    print(f"\nALL DONE in {time.time() - t_all:,.1f} s. "
          f"Revision outputs: {CONFIG['REVISION_DIR']}")

if __name__ == "__main__":
    main()
