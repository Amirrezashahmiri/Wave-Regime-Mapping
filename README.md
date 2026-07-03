# 🌊 Bathy-Wave-Clustering (BWC) Framework

This repository contains the source code and implementation of the framework proposed in the following study:

> **A Fidelity-Gated Label Propagation Framework for Bathymetry-Conditioned Deep Clustering and Decision-Support Site Selection of Wave Energy Converters along the Omani Coast**

---

## 📖 Authors

- **Amirreza Shahmiri** — Sultan Qaboos University
- **Ali Ghorban Sarvi** — Iran University of Science and Technology
- **Dr. Mohammad Reza Nikoo** *(Corresponding Author)* — Sultan Qaboos University
- **Dr. Saleh Al-Saadi** — Sultan Qaboos University
- **Dr. Seyed Mostafa Siadatmousavi** — Iran University of Science and Technology

---

# 🚀 Overview

This project introduces the **Bathy-Wave-Clustering (BWC) framework**, a generalizable Environmental Decision Support System (EDSS) designed to address the **model-to-reality gap** in large-scale wave energy assessment and Wave Energy Converter (WEC) site selection.

By integrating:

- High-resolution **ADCP observations**
- **CMEMS reanalysis products**
- Bathymetry-Conditioned Deep Clustering
- Fidelity-Gated Label Propagation via Stacking Ensembles
- Bi-objective Pareto Optimization

the framework extracts intrinsic, physically meaningful wave regimes from sparse in-situ data and safely generalizes them across regional-scale marine environments without propagating spatial biases.

The proposed methodology enables:

- ✔ Robust, bathymetry-aware wave regime mapping
- ✔ Physically consistent spatial extrapolation via a strict Fidelity Gate
- ✔ Bi-objective energy-risk tradeoff analysis (Swell Potential vs. Storm Risk)
- ✔ Optimal WEC deployment site identification

---

# 🧠 Framework Architecture

```text
ADCP + CMEMS Data (Augmented with Static Bathymetry)
        │
        ▼
Preprocessing & DINEOF Reconstruction
        │
        ▼
Bathymetry-Conditioned Deep Embedded Clustering (DEC)
        │
        ▼
Fidelity-Gated Label Propagation (Anchor Stations Only)
        │
        ▼
Cost-Sensitive Stacking Ensemble (Regional Regime Mapping)
        │
        ▼
Bi-Objective Pareto Optimization
        │
        ▼
Optimal WEC Site Selection
```

---

# ⚙️ Methodology Pipeline

## 1️⃣ Data Acquisition & Fusion

Extraction and integration of:

- High-fidelity ADCP wave measurements
- CMEMS wave reanalysis products
- Nearest-neighbor interpolation of high-resolution static bathymetry to condition the feature space

---

## 2️⃣ Preprocessing

Data preprocessing includes:

- Automated outlier detection (Hampel, Z-score, IQR ensemble)
- Missing data reconstruction using physics-based DINEOF
- Stratified downsampling for seasonal balancing
- Calculation of a 24-hour rolling Stability Index (SI)

---

## 3️⃣ Bathymetry-Conditioned Deep Regime Discovery

Wave regimes are identified using an optimized Deep Embedded Clustering (DEC) autoencoder architecture:

```text
16 → 8 → 4 → 8 → 16
```

This latent representation explicitly disentangles coastal hydro-transformations (shoaling/refraction) from unadulterated deep-water wave propagation, yielding four physical sea states:

- Golden Swell
- Storm
- Confused Sea
- Ambient/Calm

---

## 4️⃣ Fidelity-Gated Label Propagation

A proxy-learning mechanism is developed to extrapolate learned regimes across the regional grid.

To prevent the silent propagation of reanalysis errors, training is strictly limited to anchor stations passing a physical Fidelity Gate:

- Pearson R ≥ 0.80
- \(H_{m0}\) bias ≤ 0.50 m

A cost-sensitive stacking ensemble consisting of:

- Random Forest
- XGBoost
- CatBoost
- LightGBM

is then used to predict regimes across the full domain.

---

## 5️⃣ Bi-Objective Pareto Optimization

Optimal WEC deployment zones are identified through a Pareto analysis balancing:

- **Swell Energy Potential (Reward)**
- **Storm State Probability (Risk)**

The framework quantifies this engineering tradeoff, delivering a risk-informed decision atlas for offshore infrastructure planning.

---

# 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_get_data.py
│   ├── 02_adcp_preprocessing.py
│   ├── 03_cmems_merged_adcp.py
│   ├── 04_balancing.py
│   ├── 05_clustering.py
│   ├── 06_classification.py
│   └── 07_final_output.py
│
├── Data/                 # Sample data for public access
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/Amirrezashahmiri/Wave-Regime-Mapping.git

cd Wave-Regime-Mapping
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the workflow sequentially:

```bash
python notebooks/01_get_data.py
python notebooks/02_adcp_preprocessing.py
python notebooks/03_cmems_merged_adcp.py
python notebooks/04_balancing.py
python notebooks/05_clustering.py
python notebooks/06_classification.py
python notebooks/07_final_output.py
```

---

# 📊 Key Features

- **Bathymetry-Conditioned Machine Learning**  
  Explicitly accounts for depth-induced wave transformations.

- **Fidelity-Gated Knowledge Transfer**  
  Prevents spatial bias propagation from coarse numerical models.

- **Cost-Sensitive Ensemble Learning**  
  Handles severe natural sea-state class imbalances without synthetic data distortion.

- **Dual-Event Validation**  
  Empirically validated against two extreme cyclones (Shaheen and Mekunu).

- **Generalizable EDSS**  
  A transferable sparse-to-grid modeling template for broader environmental domains (e.g., wind, hydrology).

---

# 📑 Citation

If you use this repository or framework in your research, please cite our paper:

```bibtex
@article{shahmiri2026fidelitygated,
  title={A Fidelity-Gated Label Propagation Framework for Bathymetry-Conditioned Deep Clustering and Decision-Support Site Selection of Wave Energy Converters along the Omani Coast},
  author={Shahmiri, Amirreza and Ghorban Sarvi, Ali and Nikoo, Mohammad Reza and Al-Saadi, Saleh and Siadatmousavi, Seyed Mostafa},
  journal={Environmental Modelling & Software (Under Review)},
  year={2026}
}
```

---

# 📜 License

This project is released under the MIT License.

---

# 🤝 Contributions

Contributions, suggestions, and collaborations are welcome.

Please feel free to open:

- Issues
- Pull requests
- Discussions

for improvements, bug reports, or research collaborations.

---

# ⭐ Acknowledgments

The authors acknowledge:

- The Sultan Qaboos University research group **DR/RG/28** ("Climate Change, Water, and Environmental Modeling")
- Iran University of Science and Technology
- Copernicus Marine Environment Monitoring Service (CMEMS)
