# EDS_[StudentNumber]_[Surname]
## Engineering Data Systems Pipeline
**Topic:** HVA-01 — Chiller Plant COP Variance  
**Pillar:** Pillar 9 — HVAC & Building Systems  
**Course:** Computer Programming | AY 2026  
**Dataset:** [Chiller Energy Data — Kaggle](https://www.kaggle.com/datasets/chillerenergy/chiller-energy-data)

---

## Core Engineering Formula
```
COP = Cooling Load (kW) / Chiller Power Input (kW)

COP > 4.0  →  High efficiency
COP 2.5–4  →  Normal operation  
COP < 2.5  →  Degraded / fault state
```

---

## Project Structure
```
EDS_Project/
├── main.py                  ← Full Python pipeline (5 OOP modules)
├── install_packages.py      ← Run this first to install all libraries
├── requirements.txt         ← Required libraries list
├── README.md                ← This file
├── data/
│   ├── dataset_original.csv ← Raw Kaggle dataset (YOU place this here)
│   └── dataset_cleaned.csv  ← Auto-generated after pipeline runs
└── outputs/
    ├── static_01_cop_histogram.png
    ├── static_02_correlation_heatmap.png
    ├── static_03_boxplot_by_hour.png
    ├── static_04_actual_vs_predicted.png
    ├── static_05_feature_importance.png
    ├── anim_01_cop_rolling_mean.gif
    └── anim_02_temp_vs_cop_scatter.gif
```

---

## Setup & Run Instructions

### Step 1 — Install dependencies
```bash
python install_packages.py
```

### Step 2 — Download the dataset
1. Go to: https://www.kaggle.com/datasets/chillerenergy/chiller-energy-data
2. Click **Download** (free Kaggle account required)
3. Extract the ZIP file
4. Rename the CSV to **`dataset_original.csv`**
5. Place it inside the **`data/`** folder

### Step 3 — Run the pipeline
```bash
python main.py
```

---

## Unique Filter (Anti-Duplication Rule)
- **Filter:** `Month == 7` (July — peak cooling season)
- This ensures a mathematically unique data slice per project rules

---

## Pipeline Modules

| Module | Class | Responsibility |
|--------|-------|----------------|
| 1 | `DataIngestion` | Load CSV, print shape & null counts |
| 2 | `DataCleaning` | Remove duplicates, fill nulls, encode, engineer COP, apply filter, remove outliers |
| 3 | `StatisticalAnalysis` | NumPy descriptive stats, skewness, correlation matrix, Welch T-test |
| 4 | `MachineLearning` | Train Linear Regression, Random Forest, Gradient Boosting; compare RMSE/MAE/R² |
| 5 | `Visualization` | 5 static PNG plots + 2 animated GIF files |

---

## GitHub Repository Name
```
EDS_[YourStudentNumber]_[YourSurname]
```
Example: `EDS_2021123456_Reyes`
