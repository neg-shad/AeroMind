# AeroMind ✈️
### Machine Learning Framework for Aircraft Engine Anomaly Detection

AeroMind is a modular machine learning framework designed to detect anomalies and predict degradation patterns in turbofan engines using multivariate time-series sensor data.

---

## 🔬 Project Motivation
Modern aircraft engines operate under highly dynamic conditions. Detecting subtle deviations from normal operational behavior is critical for:

*   **Safety:** Preventing catastrophic failures.
*   **Efficiency:** Reducing maintenance costs.
*   **Innovation:** Enabling predictive maintenance systems.

---

## 📊 Dataset
This project uses the **NASA CMAPSS Turbofan Engine Degradation Dataset**.

**Key characteristics:**
*   Simulated turbofan engine degradation cycles.
*   Multivariate sensor measurements (21 sensors).
*   3 Operational settings.
*   Total columns per row: **26**.

> [!IMPORTANT]
> The objective is to detect abnormal engine behavior and estimate remaining useful life (RUL).

---



## 🧭 Project Roadmap

Our development path follows a structured transition from classical heuristics to state-of-the-art deep learning architectures.

- [x] **Phase 0: Infrastructure**
  - [x] Repository initialization & Git strategy.
  - [x] Modular project structure definition.
  - [x] Professional Documentation (README.md).

- [x] **Phase 1: Environment Setup**
  - [x] Dependency management (`requirements.txt`).
  - [x] Virtual environment configuration.

- [ ] **Phase 2: Data Engineering (Active)**
  - [ ] Implementation of `data_loader.py` for CMAPSS parsing.
  - [ ] Multi-regime normalization using `StandardScaler`.
  - [ ] Exploratory Data Analysis (EDA) in Jupyter Notebooks.

- [ ] **Phase 3: Baseline Modeling**
  - [ ] Unsupervised Anomaly Detection using **Isolation Forest**.
  - [ ] Feature importance analysis for sensor health scores.
  - [ ] Performance evaluation against ground-truth RUL.

- [ ] **Phase 4: Advanced Architectures**
  - [ ] Temporal pattern extraction with **LSTM Autoencoders**.
  - [ ] Multi-head attention mechanisms for sensor-to-sensor correlations (**Transformers**).
  - [ ] Real-time anomaly scoring pipeline.

---

## 📜 License
*   This project is released under the MIT License.
---

## 🔗 Inspiration & Similar Projects
This project is inspired by state-of-the-art research in predictive maintenance. For further reading, explore these related repositories:

*   **[NASA-RUL-Prediction](https://github.com/biswajitsahoo1111/RUL_Prediction_using_Deep_Learning):** Deep learning approaches for Remaining Useful Life estimation.
*   **[Azure Predictive Maintenance](https://github.com/Azure-Samples/Predictive-Maintenance-using-LSTM):** Industrial-scale implementation of LSTM for turbofan monitoring.
*   **[PyOD](https://github.com/yzhao062/pyod):** A comprehensive toolkit for detecting outlying objects in multivariate data.

## 📚 References
*   A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository.
*   Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest".
---
