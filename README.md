# IoT Project 2: Weather Data Analysis

CSC 591/ECE 592 IoT Analytics - Data Acquisition and Analytics

## Overview

This project uses **Open-Meteo Historical Weather API** (benchmark data option) to analyze weather variables and their effect on apparent temperature. It does not use the WiFi or other hardware options.

**Variables:**
- **X1**: Temperature (°C)
- **X2**: Relative humidity (%)
- **X3**: Surface pressure (hPa)
- **X4**: Wind speed (km/h)
- **X5**: Cloud cover (%)
- **Y**: Apparent temperature (°C) – output

## Folder Structure

```
weather-data-analysis/
├── task1/                    # Task 1: Data Acquisition
│   ├── task1_data_acquisition.py
│   ├── weather_data.csv      (created when run)
│   └── weather_dataset.xlsx  (created when run)
├── task2/                    # Task 2: Basic Statistics
│   ├── task2_basic_statistics.py
│   ├── socs_analysis_report.md
│   ├── task2_descriptive_statistics.csv  (created when run)
│   ├── task2_pmf_plots.png   (created when run)
│   └── manual_20_samples.csv  (created when run)
├── task3/                    # Task 3: Data Visualization
│   ├── task3_data_visualization.py
│   ├── task3_visualization_analysis.md
│   ├── task3_boxplots.png    (created when run)
│   ├── task3_scatter_plots.png (created when run)
│   └── task3_density_curve.png (created when run)
├── run_all.py
├── requirements.txt
├── IoT_Project2_OpenMeteo_Template.xlsx
├── Pro2-Data (1).pdf
└── WiFi-configuration.pdf    (not used – course material for WiFi option)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires: Python 3.8+, pandas, numpy, matplotlib, scipy, requests, openpyxl.

## How to Run

**Option 1 – Run everything:**
```bash
python run_all.py
```

**Option 2 – Run tasks one by one:**
```bash
python task1/task1_data_acquisition.py
python task2/task2_basic_statistics.py
python task3/task3_data_visualization.py
```

Run from the project root. **Order matters:** task2 and task3 read `task1/weather_data.csv`, so task1 must run first.

## What Happens When You Run

**task1_data_acquisition.py**
- Fetches ~2200 days (~6 years) of noon weather for Raleigh, NC from Open-Meteo
- Writes `task1/weather_data.csv` and `task1/weather_dataset.xlsx`
- Needs internet

**task2_basic_statistics.py**
- Reads `task1/weather_data.csv` (first 100 rows)
- Writes `task2/task2_descriptive_statistics.csv`, `task2/task2_pmf_plots.png`, `task2/manual_20_samples.csv`

**task3_data_visualization.py**
- Reads `task1/weather_data.csv` (first 100 rows)
- Writes `task3/task3_boxplots.png`, `task3/task3_scatter_plots.png`, `task3/task3_density_curve.png`

## Where to Find Results

| Output | Location |
|--------|----------|
| Dataset (CSV) | `task1/weather_data.csv` |
| Dataset (Excel, for submission) | `task1/weather_dataset.xlsx` |
| Descriptive stats table | `task2/task2_descriptive_statistics.csv` |
| PMF plots | `task2/task2_pmf_plots.png` |
| 20 samples for manual calc | `task2/manual_20_samples.csv` |
| SOCS analysis | `task2/socs_analysis_report.md` |
| Boxplots | `task3/task3_boxplots.png` |
| Scatter plots | `task3/task3_scatter_plots.png` |
| Density curve | `task3/task3_density_curve.png` |
| Visualization analysis | `task3/task3_visualization_analysis.md` |

## Task Summary

1. **Task 1**: Data acquisition via Open-Meteo API (Raleigh, NC; noon readings; 2200+ samples)
2. **Task 2**: Descriptive statistics, PMF (50 intervals), SOCS analysis
3. **Task 3**: Boxplots, scatter plots, density curve, and interpretation
