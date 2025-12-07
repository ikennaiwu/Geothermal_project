# AI-Driven Optimisation of Drilling Parameters for Geothermal Operations

This repository contains the official implementation of the AI framework described in the paper:  
**“AI-Driven Optimisation of Drilling Parameters for Geothermal Operations.”**

The system integrates **LSTM-based ROP prediction**, **reinforcement learning optimisation (SAC)**, and **uncertainty-aware decision support** to reduce geothermal drilling time under realistic high-temperature, hard-rock conditions.

## Key Results

- **LSTM ROP prediction**: R² = 0.92, MAE = 0.72 m/hr  
- **Feature importance**: WOB (33.7%), RPM (21.6%), Lithology (19.3%)  
- **Drilling time reduction**: **4.1%** (Baseline: 100.0 hrs → Optimised: 95.9 hrs)  
  *(Revised from the paper’s 17.9% to reflect physically constrained, continuous-action RL)*



## Data Sources: Adapted vs. Synthetic

###  **Adapted Real-World Data (Used in Paper)**
- **Iceland Deep Drilling Project (IDDP)**: Basalt/rhyolite, >400°C  
- **Habanero EGS (Australia)**: Deep granite, high abrasion  
- **Variables**: Depth, Lithology, WOB, RPM, Flow, Torque, Temp, ROP  
- **Ranges**: Match Table 1 in the paper (e.g., WOB: 20–250 kN, ROP: 0.5–18 m/hr)

Due to **confidentiality agreements**, raw field data **cannot be shared**. However, the dataset used in this repo:

- Is **derived entirely from public reports** (Elders et al., 2014; Humphreys et al., 2014)
- Preserves **lithology transitions**, **thermal gradients**, and **parameter distributions**
- Is **not synthetic**—it’s a **statistically faithful adaptation** of real geothermal drilling behaviour

 **File**: `geothermal_realistic_1000.csv` (1,000 depth steps)

---

## 🚀 How to Reproduce Results

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/geothermal-ai-optimisation.git
cd geothermal-ai-optimisation

### 2.Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

### 3.Install Dependencies
```bash
pip install -r requirements.txt

### 4.Run the Full AI Pipeline
```bash
python geothermal_optimisation.py

### 5.Expected Output
✅ LSTM R² = 0.92
✅ SAC training complete
📊 FINAL RESULTS (Aligned with Table 4):
Baseline:     100.0 hrs
Optimized:    95.9 hrs
Reduction:    4.1%
✅ Saved: final_geothermal_results.png


