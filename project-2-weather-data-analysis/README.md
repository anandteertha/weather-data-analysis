# Project 2: Weather Data Analysis

CSC 591 / ECE 592 IoT Analytics

## Overview

This folder contains the Project 2 work used to collect and explore the weather dataset that later feeds Project 3.

Variables:
- `X1`: Temperature (C)
- `X2`: Relative humidity (%)
- `X3`: Surface pressure (hPa)
- `X4`: Wind speed (km/h)
- `X5`: Cloud cover (%)
- `Y`: Apparent temperature (C)

## Structure

```text
project-2-weather-data-analysis/
├── task1/
├── task2/
├── task3/
├── run_all.py
├── requirements.txt
└── README.md
```

## How To Run

From this folder:

```bash
pip install -r requirements.txt
python run_all.py
```

You can also run the tasks individually:

```bash
python task1/task1_data_acquisition.py
python task2/task2_basic_statistics.py
python task3/task3_data_visualization.py
```

## Outputs

- `task1/weather_data.csv`: main dataset used in later work
- `task1/weather_dataset.xlsx`: Excel export of the dataset
- `task2/task2_descriptive_statistics.csv`: descriptive statistics
- `task2/task2_pmf_plots.png`: PMF plots
- `task2/manual_20_samples.csv`: sample subset for manual calculations
- `task2/socs_analysis_report.md`: Task 2 interpretation
- `task3/task3_boxplots.png`: boxplots
- `task3/task3_scatter_plots.png`: scatter plots
- `task3/task3_density_curve.png`: density curve
- `task3/task3_visualization_analysis.md`: Task 3 interpretation

## Notes

- `task1/weather_data.csv` is the dataset consumed by Project 3.
- Task 1 may require internet access because it fetches data from Open-Meteo.
