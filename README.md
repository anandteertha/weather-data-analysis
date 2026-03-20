# Weather Data Analysis Repository

This repository is organized as a two-project course repo for CSC 591 / ECE 592 IoT Analytics.

## Projects

### `project-2-weather-data-analysis`

Project 2 covers data acquisition and exploratory analysis:
- collects the weather dataset
- computes descriptive statistics
- produces initial visualizations

Key dataset artifact:
- [weather_data.csv](/c:/Users/anand/Development/weather-data-analysis/project-2-weather-data-analysis/task1/weather_data.csv)

Project entry points:
- [run_all.py](/c:/Users/anand/Development/weather-data-analysis/project-2-weather-data-analysis/run_all.py)
- [README.md](/c:/Users/anand/Development/weather-data-analysis/project-2-weather-data-analysis/README.md)

### `project-3-regression-forecasting`

Project 3 builds regression models on top of the Project 2 dataset:
- basic statistics with outlier filtering
- simple linear regression
- quadratic regression comparison
- multivariable linear regression
- residual diagnostics and final report

Project entry points:
- [run_all.py](/c:/Users/anand/Development/weather-data-analysis/project-3-regression-forecasting/run_all.py)
- [report.md](/c:/Users/anand/Development/weather-data-analysis/project-3-regression-forecasting/report.md)

## Structure

```text
weather-data-analysis/
├── project-2-weather-data-analysis/
└── project-3-regression-forecasting/
```

## Typical Workflow

1. Generate or verify the Project 2 dataset in `project-2-weather-data-analysis/task1/weather_data.csv`.
2. Run Project 3 from `project-3-regression-forecasting/`.
3. Submit the Project 3 report and generated plots/results as needed.

## Run Commands

Project 2:

```bash
cd project-2-weather-data-analysis
pip install -r requirements.txt
python run_all.py
```

Project 3:

```bash
cd project-3-regression-forecasting
pip install -r requirements.txt
python run_all.py
```

## Notes

- Project 3 expects the dataset created in Project 2.
- The folder names are intentionally descriptive so the repo reads clearly in a file browser and in git history.
