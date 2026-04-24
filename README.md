# 🐦 BirdCLEF+ 2026 — Bird Sound Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This repository contains a complete solution for the **BirdCLEF+ 2026 Kaggle Competition** — a bioacoustics challenge focused on classifying bird species from raw audio recordings in real-world soundscapes.

The pipeline uses **Google's Perch** pre-trained audio embedding model to extract rich acoustic features, followed by a custom **MLP classification head** with **pseudo-label distillation** to improve generalisation on unlabeled soundscape data.

---

## 🏆 Competition Context

> **BirdCLEF+** is an annual Kaggle competition that challenges participants to identify bird species from passive acoustic monitoring recordings. The 2026 edition includes extended species coverage and soundscape audio from challenging real-world environments.

- **Evaluation Metric:** Macro-averaged ROC-AUC across all bird species
- **Platform:** Kaggle (T4 GPU environment)
- **Data:** Labelled training clips + large unlabeled soundscape recordings

---

## 🏗️ Solution Architecture

```
Audio Data
    ↓
Perch Embedding Extraction (GPU, ~30 min)
    ↓
MLP Classification Head Training (~20 min)
    ↓
Pseudo-Label Distillation on Soundscapes (~40 min)
    ↓
Final MLP Retrain on Augmented Dataset
    ↓
Inference + Submission Generation
```

---

## 🔬 Pipeline Details

### Cell 1 — EDA & Data Exploration
- Loaded and explored training metadata using Pandas
- Analysed species distribution, audio file counts, and soundscape structure
- Used `librosa` for audio loading and waveform inspection

### Cell 2 — Environment Setup + Model Scan
- Verified GPU availability (`torch.cuda.is_available()`)
- Scanned external datasets from Kaggle input directories
- Configured base paths for data loading

### Cell 3 — Perch Embedding Extraction
- Downloaded and installed **Google Perch** (a TF-Hub audio model trained on bird sounds)
- Extracted 1280-dimensional embeddings for all training audio clips
- Runtime: ~25–35 min on T4 GPU

### Cell 4 — MLP Training
- Trained a multi-layer perceptron on Perch embeddings
- Multi-label classification head (one output per species)
- Loss: Binary Cross-Entropy | Optimiser: Adam
- Runtime: ~15–20 min on T4 GPU

### Cell 5 — Pseudo-Label Distillation (Round 1)
- Generated soft predictions on all unlabeled soundscape recordings
- Selected high-confidence predictions as pseudo-labels
- Retrained MLP on combined labelled + pseudo-labelled data
- Runtime: ~35–40 min total

### Cell 6 — Inference + Submission
- CPU-compatible inference pipeline for Kaggle evaluation
- Generated species probability predictions for all test segments
- Formatted and validated submission file

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| PyTorch | MLP training |
| TensorFlow / TF-Hub | Perch model inference |
| Google Perch | Pre-trained bird audio embeddings |
| librosa | Audio loading and analysis |
| Pandas / NumPy | Data manipulation |
| Kaggle T4 GPU | Training acceleration |

---

## 📁 Project Structure

```
JOB/
│
└── birdclef-2026.ipynb    # Full 8-cell pipeline notebook
```

---

## 🚀 How to Run

This notebook is designed to run on **Kaggle** with GPU enabled.

1. Fork the notebook on Kaggle
2. Enable GPU (Settings → Accelerator → GPU T4)
3. Add the BirdCLEF+ 2026 competition dataset
4. Run all cells sequentially

---

## 👨‍💻 Author

**Homeswar Rao Nelakurthi**
[![GitHub](https://img.shields.io/badge/GitHub-homeshwarnelakurthi-181717?style=flat&logo=github)](https://github.com/homeshwarnelakurthi)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
