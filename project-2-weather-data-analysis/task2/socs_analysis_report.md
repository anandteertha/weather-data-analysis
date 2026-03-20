# Task 2.3: SOCS Framework Analysis Report

## Dataset Overview
- **Output variable Y**: Apparent temperature (°C) – how hot/cold it feels
- **Input variables X1–X5**: Temperature, humidity, pressure, wind speed, cloud cover
- **Sample size**: 100 daily noon observations (Raleigh, NC)

---

## 1. Shape

| Variable | Shape Description |
|----------|-------------------|
| **X1 (Temperature)** | Approximately symmetric, unimodal. PMF shows a bell-like distribution centered near the mean. |
| **X2 (Humidity)** | Slightly right-skewed. Values cluster in mid-range (40–70%) with a tail toward higher humidity. |
| **X3 (Pressure)** | Approximately symmetric, narrow spread. Values cluster tightly around 1005–1015 hPa. |
| **X4 (Wind speed)** | Right-skewed. Most values are low (2–10 km/h) with fewer high-wind observations. |
| **X5 (Cloud cover)** | Bimodal or uniform tendency. Values spread across 0–100% with possible peaks at clear (0) and overcast (100). |

---

## 2. Outliers

| Variable | Outlier Assessment |
|----------|--------------------|
| **X1** | Few extreme high/low temperatures possible in summer; generally within expected range. |
| **X2** | Occasional very high humidity (>90%) or low (<30%) may appear; check for impact on Y. |
| **X3** | Pressure is stable; outliers rare. Any extreme values may indicate data errors. |
| **X4** | High wind speeds (>25 km/h) may be outliers; typically few in a 100-sample set. |
| **X5** | Cloud cover 0–100% is valid; no true outliers, but extremes affect apparent temperature. |

**Impact**: Outliers can skew mean and variance. The median and IQR are more robust. For Y (apparent temperature), extreme X1 (temperature) and X2 (humidity) have the strongest influence.

---

## 3. Center

| Variable | Mean vs Median | Interpretation |
|----------|----------------|----------------|
| **X1** | Mean ≈ median | Symmetric; center well represents typical temperature. |
| **X2** | Mean > median (if skewed right) | Higher humidity days pull mean up; median is more typical. |
| **X3** | Mean ≈ median | Stable pressure; both measures reflect central tendency. |
| **X4** | Mean > median | Right skew; a few windy days raise the mean. |
| **X5** | Depends on distribution | If bimodal, mean may fall between modes and be less representative. |

**Implication for Y**: Apparent temperature is most sensitive to X1 and X2. Their central values indicate the typical "feels-like" conditions.

---

## 4. Spread

| Variable | Range | IQR | Variance | Interpretation |
|----------|-------|-----|----------|----------------|
| **X1** | ~25–40°C (summer) | Moderate | Moderate | Temperature varies day-to-day; affects Y directly. |
| **X2** | ~30–95% | Moderate | High | Humidity has wide spread; strong effect on Y (heat index). |
| **X3** | ~1000–1015 hPa | Small | Low | Pressure is stable; weak direct effect on Y. |
| **X4** | ~0–25 km/h | Moderate | Moderate | Wind cools; higher spread means more variability in cooling effect. |
| **X5** | 0–100% | Large | High | Cloud cover varies greatly; affects solar heating and thus Y. |

**Implication for Y**: High variability in X1, X2, and X5 leads to high variability in Y. X3 and X4 contribute less to the spread of apparent temperature.

---

## Summary: Influence on Y (Apparent Temperature)

1. **X1 (Temperature)** – Strongest driver; Y tracks X1 closely.
2. **X2 (Humidity)** – Strong; high humidity increases apparent temperature (heat index effect).
3. **X5 (Cloud cover)** – Moderate; clouds reduce solar heating, lowering Y for given X1.
4. **X4 (Wind speed)** – Moderate; wind increases evaporative cooling, lowering Y.
5. **X3 (Pressure)** – Weak; pressure has minimal direct effect on perceived temperature.
