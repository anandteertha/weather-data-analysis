# Task 3.4: Analysis and Interpretation of Visualizations

## 3.1 Boxplot Findings

| Variable | Observations |
|----------|--------------|
| **X1 (Temperature)** | Median near center; whiskers show full range of summer temperatures. Few outliers possible at extremes. |
| **X2 (Humidity)** | Box spans wide range; IQR shows central 50% of humidity values. Potential outliers at very low or very high humidity. |
| **X3 (Pressure)** | Narrow box; pressure is stable. Outliers rare. |
| **X4 (Wind speed)** | Right-skewed; upper whisker longer. Some high-wind days extend the range. |
| **X5 (Cloud cover)** | Wide spread from 0–100%; box may not capture central tendency well if distribution is bimodal. |

**Key takeaway**: X2 and X5 have the largest spread; X3 has the smallest. This aligns with the descriptive statistics (variance and IQR).

---

## 3.2 Scatter Plot Findings

| Y vs Variable | Correlation Pattern | Interpretation |
|---------------|---------------------|----------------|
| **Y vs X1** | Strong positive linear | Apparent temperature tracks actual temperature closely. Primary driver of Y. |
| **Y vs X2** | Positive, some scatter | Higher humidity increases apparent temperature (heat index). Relationship is nonlinear at extremes. |
| **Y vs X3** | Weak or none | Pressure has minimal effect on perceived temperature. |
| **Y vs X4** | Slight negative | Wind provides cooling; higher wind tends to lower apparent temperature. |
| **Y vs X5** | Weak negative or mixed | Clouds reduce solar heating; effect may be secondary to X1 and X2. |

**Key takeaway**: X1 and X2 are the strongest predictors of Y. X3, X4, and X5 contribute less but add nuance (e.g., wind chill, cloud shading).

---

## 3.3 Density Curve of Y

- **Shape**: Likely unimodal, approximately symmetric or slightly skewed depending on the season.
- **Center**: Mean and median of Y should be close to each other if symmetric.
- **Spread**: Reflects day-to-day variation in "feels-like" temperature.
- **Interpretation**: The density curve summarizes how often different apparent temperatures occur. Peaks indicate the most common conditions.

---

## How Visualizations Enhance Understanding

1. **Boxplots** quickly show central tendency, spread, and outliers for each variable without needing to read numbers.
2. **Scatter plots** reveal relationships between Y and each Xi, highlighting which variables matter most for prediction.
3. **Density curve** summarizes the distribution of the output Y, supporting decisions about modeling (e.g., normality) and interpretation of typical conditions.

These visualizations complement the SOCS analysis and descriptive statistics, providing an intuitive view of the dataset and the factors influencing apparent temperature.
