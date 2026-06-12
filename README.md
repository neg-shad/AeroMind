# ✈️ AeroMind: An Explainable Deep Learning Framework for Turbofan Prognostics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Dataset-NASA_CMAPSS](https://img.shields.io/badge/Dataset-NASA_CMAPSS-red.svg)](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

## 📌 Executive Summary
**AeroMind** is a research-oriented framework developed to estimate the **Remaining Useful Life (RUL)** of aero-engines using multivariate time-series data. By leveraging the **NASA C-MAPSS** dataset, this project implements a high-precision pipeline that transitions from raw sensor telemetry to predictive maintenance insights. 

The framework is architected to address the non-linear degradation nature of complex mechanical systems, employing **Temporal Sequence Modeling (LSTM)** and establishing a rigorous evaluation baseline against classical machine learning methods.

---

## 🏗️ System Architecture & Pipeline
The AeroMind pipeline is strictly modular, ensuring data integrity and preventing temporal leakage:

1.  **Data Engineering (`CMAPSDataLoader`):**
    *   Automated schema validation for 26-column raw telemetry.
    *   **Heuristic Filtering:** Identification and removal of constant-value sensors (e.g., sensors 1, 5, 10, 16, 18, 19) that provide zero variance.
    *   **Standardization:** Engine-wise scaling using `StandardScaler` fitted exclusively on training trajectories.
2.  **Trajectory Windowing (`WindowGenerator`):**
    *   Transformation of 2D time-series into 3D Tensors: $(N, \text{Window Size}, \text{Features})$.
    *   Optimized window length: **30 cycles**.
3.  **Target Engineering:**
    *   Implementation of **Piecewise Linear RUL**.
    *   **Thresholding:** RUL is clipped at **125 cycles** to mitigate noise in early-life "healthy" operating zones, focusing the model on the terminal degradation phase.
4.  **Modeling Engine:**
    *   **Baseline:** Ridge Regression (Flattened Sequences).
    *   **Deep Core:** Multi-layer LSTM for capturing long-term dependencies in sensor degradation.

---

## 🔬 Methodology & Model Specifications

### 1. Deep Temporal Modeling (LSTM)
To capture the dynamics of wear and tear, we implemented a deep recurrent architecture:
- **Input Shape:** $(30, 17)$ (Window Size $\times$ Selected Features).
- **Architecture:** 
  - `LSTM Layer`: 64 units (Hidden State Representation).
  - `Dropout`: 0.3 (Regularization).
  - `Dense Layer`: 32 units (ReLU activation).
  - `Output Layer`: 1 unit (Linear activation for RUL).
- **Optimization:** Adam optimizer with Mean Squared Error (MSE) loss.

### 2. AeroTransformer (Experimental)
A research-level implementation of a Transformer-based regressor featuring:
- **Positional Encoding:** To maintain temporal order without recurrence.
- **Multi-Head Attention:** To weigh critical sensor events across the 30-cycle window.

---

## 📊 Experimental Results
The models were evaluated on the official NASA FD001 test set. Our LSTM-based approach demonstrated superior performance in capturing the non-linear degradation curve.

| Model | Architecture | RMSE | MAE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Classical Baseline** | Ridge Regression | 15.82 | 11.95 | 0.8563 |
| **AeroMind (Base)** | **Vanilla LSTM** | **12.56** | **9.26** | **0.9095** |
| **AeroMind (Adv)** | AeroTransformer | *In Progress* | *--* | *--* |

### Key Findings:
- **Sequence Retention:** Utilizing a 30-cycle temporal window improved the RMSE by **~21%** compared to flattened classical models.
- **RUL Clipping:** Setting the upper bound to 125 significantly reduced the loss variance during the early stages of engine life.

---

## 📁 Project Structure
```text
AeroMind/
├── data/                   # NASA C-MAPSS Raw CSVs (FD001)
├── src/
│   ├── data_loader.py      # CMAPSDataLoader & WindowGenerator
│   ├── models.py           │ AeroTransformer & LSTM Architecture
│   ├── trainer.py          │ PyTorch Lightning-style training loops
│   └── utils.py            │ BaselineUtils (RMSE, MAE, R2 calculations)
├── notebooks/              # Exploratory Data Analysis (EDA) & Prototyping
├── results/                # Saved weights & performance plots
├── requirements.txt        # Reproducibility manifest
└── README.md
```
---

## 🛠️ Installation & Reproducibility

To replicate the experimental environment and run the models, follow these steps:
```bash
# Clone the repository
git clone https://github.com/your-username/AeroMind.git
cd AeroMind

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activat  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Execute the training and evaluation pipeline
python src/trainer.py --dataset FD001 --model lstm --epochs 50
```

---
## 🚀 Research Roadmap

*   [ ] Explainability (XAI): Integration of Integrated Gradients to visualize sensor-level contributions.
*   [ ] Cross-Domain Adaptation: Testing on FD002 and FD004.
*   [ ] Uncertainty Estimation: Implementing Monte Carlo Dropout.


---
## 🎓 Acknowledgments

This research utilizes the Turbofan Engine Degradation Simulation Data Set provided by the NASA Prognostics Data Repository (PCoE).

---
## 👤 Author & Contact

* Developer: Negin Shadravan
* E-mail: neginshad81@gmail.com
* Field: Aerospace Engineering | Machine Learning | Predictive Maintenance

---
