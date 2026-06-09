# 🌊 Physics-Aware Hybrid Deep Learning Framework for Large-Scale Wave Regime Mapping and Multi-Objective Site Selection of Wave Energy Converters

This repository contains the source code and implementation of the framework proposed in the following study:

> **Physics-Aware Hybrid Deep Learning Framework for Large-Scale Wave Regime Mapping and Multi-Objective Site Selection of Wave Energy Converters**

---

## 📖 Authors

- **Amirreza Shahmiri** — Sultan Qaboos University  
- **Ali Ghorban Sarvi** — Iran University of Science and Technology  
- **Mohammad Reza Nikoo** *(Corresponding Author)* — Sultan Qaboos University  
- **Saleh Al-Saadi** — Sultan Qaboos University  
- **Seyed Mostafa Siadatmousavi** — Iran University of Science and Technology  

---

# 🚀 Overview

This project introduces a **physics-aware hybrid deep learning framework** designed to address the **model-to-reality gap** in large-scale wave energy assessment and Wave Energy Converter (WEC) site selection.

By integrating:

- High-resolution **ADCP observations**
- **CMEMS reanalysis products**
- Deep clustering architectures
- Physics-based transfer learning
- Multi-objective optimization

the framework identifies realistic hydrodynamic regimes and generalizes them across regional-scale marine environments.

The proposed methodology enables:

✔ Robust wave regime mapping  
✔ Physically consistent spatial extrapolation  
✔ Energy-risk tradeoff analysis  
✔ Optimal WEC deployment site identification  

---

# 🧠 Framework Architecture

```text
ADCP + CMEMS Data
        │
        ▼
Preprocessing & DINEOF Reconstruction
        │
        ▼
Deep Embedded Clustering (DEC)
        │
        ▼
Physics-Aware Fidelity Transfer
        │
        ▼
Regional Wave Regime Mapping
        │
        ▼
Pareto-Based Multi-Objective Optimization
        │
        ▼
Optimal WEC Site Selection
```

---

# ⚙️ Methodology Pipeline

## 1️⃣ Data Acquisition

- Extraction and integration of:
  - ADCP wave measurements
  - CMEMS wave reanalysis products
- Temporal and spatial synchronization of datasets

---

## 2️⃣ Preprocessing

Data preprocessing includes:

- Outlier detection and removal
- Missing data reconstruction using **DINEOF**
- Feature normalization and balancing
- Hydrodynamic consistency checks

---

## 3️⃣ Deep Regime Discovery

Wave regimes are identified using:

### Deep Embedded Clustering (DEC)

with an autoencoder architecture:

```text
16 → 8 → 4 → 8 → 16
```

This latent representation enables the discovery of physically meaningful hydrodynamic patterns.

---

## 4️⃣ Fidelity-Gated Transfer Learning

A physics-aware transfer learning mechanism is developed to extrapolate learned regimes across the regional grid using:

- Physical similarity constraints
- Hydrodynamic fidelity filters
- Spatial consistency metrics

This step minimizes unrealistic model generalization.

---

## 5️⃣ Pareto Multi-Objective Optimization

Optimal WEC deployment zones are identified through Pareto optimization considering:

- Wave energy potential
- Storm risk exposure
- Regime persistence
- Operational reliability

The framework balances energy maximization against environmental and operational risks.

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
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Physics-Aware-Wave-Energy.git
cd Physics-Aware-Wave-Energy
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

- Physics-aware machine learning
- Deep embedded clustering
- Hybrid observational–reanalysis integration
- Transfer learning with fidelity constraints
- Large-scale wave regime mapping
- Multi-objective WEC site optimization
- Practical framework for renewable marine energy planning

---

# 🌍 Potential Applications

- Wave energy resource assessment
- Marine renewable energy planning
- Coastal engineering
- Offshore infrastructure design
- Climate-resilient energy systems
- Oceanographic regime classification

---

# 📑 Citation

If you use this repository in your research, please cite:

```bibtex
@article{shahmiri2026physicsaware,
  title={Physics-Aware Hybrid Deep Learning Framework for Large-Scale Wave Regime Mapping and Multi-Objective Site Selection of Wave Energy Converters},
  author={Shahmiri, Amirreza and Ghorban Sarvi, Ali and Nikoo, Mohammad Reza and Al-Saadi, Saleh and Siadatmousavi, Seyed Mostafa},
  journal={Under Review},
  year={2026}
}
```

---

# 📜 License

This project is released under the **MIT License**.

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

- Sultan Qaboos University
- Iran University of Science and Technology
- CMEMS program
- ADCP observational data providers

for supporting this research.
