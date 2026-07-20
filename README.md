# Time-Series Model Comparator

A professional-grade PyTorch benchmarking framework designed to evaluate and compare the performance of various time-series forecasting architectures.

This suite provides a rigorous **"apples-to-apples"** comparison of predictive accuracy, computational scalability, and memory usage.

---

## Overview

This project benchmarks three distinct architectural paradigms for **Long-Term Time-Series Forecasting (LTSF):**

- **Recurrent (LSTM)** – Serving as the sample-efficient baseline.
- **Attention-based (Transformer)** – Analyzing self-attention mechanisms in time-series.
- **Frequency-domain (Autoformer)** – Testing decomposition-based forecasting for long-term seasonality.

---

## Key Features

- **Standardized Benchmarking**
  - Parameter parity across models for fair comparison.

- **Hardware Metrics**
  - Tracks VRAM usage (MB) and Latency (ms/batch) to expose architectural bottlenecks.

- **Automated Experimentation**
  - A modular orchestrator (`run_benchmark.py`) that executes full grids of datasets, horizons, and models.

- **Loss Visualizations**
  - Automatically generates loss curves for every training run.

---

## Repository Structure

```text
├── datasets/           # Data repository
├── data_loader/        # Sliding window dataset logic
├── models/             # FairLSTM, FairTransformer, and Autoformer
├── results/            # CSV metrics and training loss plots
├── run_benchmark.py    # Master experiment orchestrator
└── main.py             # Debugging and model-specific testing
```

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/Sahil2927/time-series-model-comparator.git
cd time-series-model-comparator
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Experiments

```bash
python run_benchmark.py
```

---

# Empirical Highlights

Initial benchmarks demonstrate that:

- **LSTM** remains highly competitive in accuracy for short-term horizons under constrained compute budgets.
- **Autoformer** exhibits superior scalability and predictive robustness on high-dimensional datasets such as **Traffic**.

---

# Goal

The objective of this project is to provide a fair and reproducible benchmarking framework for comparing modern time-series forecasting architectures under identical experimental settings.

The framework emphasizes:

- Fair model comparison
- Reproducible experimentation
- Hardware efficiency analysis
- Performance visualization
- Modular experimentation pipeline

---

# Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib

---

# License

This project is intended for research and educational purposes.
