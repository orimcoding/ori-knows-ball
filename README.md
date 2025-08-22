# Ori Knows Ball ⚽️

[![Project Status](https://img.shields.io/badge/status-Pre--Alpha-orange)]()
[![Model Stage](https://img.shields.io/badge/model-training_in_progress-blue)]()
[![Version](https://img.shields.io/badge/version-0.1.0--dev-lightgrey)]()
[![License](https://img.shields.io/badge/license-MIT-black)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-informational)]()

Predicting the 2025–26 UEFA Champions League — from the **group stage** to the **final** — using machine learning (**Random Forest** classification).

---

## ✨ Features
- **Match-level pre-game predictions** (H/D/A)
- **Round-by-round simulation** (Monte Carlo from per-match probabilities)
- **Incremental upgrades**: Elo → rolling form → team stats (FBref)
- Optional inputs (future): **news**, **injuries**, **sentiment**

---

## 🧰 Tech Stack
- **Core**: Python (pandas, NumPy), scikit-learn
- **Data**: FBref (CSV) + ClubElo-style ratings (CSV with `from`/`to`)
- **Viz**: Matplotlib (static reports)
- **IO**: CSV / Parquet (for fast cached datasets)
- **App (TBD)**: Streamlit or Flask UI

---

## 📦 Repository Layout
- fbref_data/<season>/.csv # schedule, shooting, passing, ...
- data/elo_filtered/.csv # time-ranged Elo per team
- scripts/preview_clean_v1.py # schema check + cleaned schedule generation

## 🚦 Project Status
- **Scope**: 2025–26 UEFA Champions League (group → final)
- **Current focus**: data hygiene (schedule parsing), **per-match Elo join**, baseline **RandomForestClassifier**
- **Risks/Next**: team-name normalization, header flattening for FBref tables, leakage-proof rolling windows, calibration

---

## 🔖 Versioning
- Current: `0.1.0-dev` (data engineering/research/training)
