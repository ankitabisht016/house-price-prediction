# 🏠 House Price Prediction

**Python · Scikit-learn · Pandas · Linear Regression**

End-to-end regression pipeline on the California Housing Dataset to predict median house prices based on area, location, and demographic features.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **R² Score** | **0.8448** |
| MAE | $24,307 |
| RMSE | $30,902 |
| CV R² (5-fold) | 0.8495 ± 0.0054 |

---

## 🗂️ Project Structure

```
house_price_prediction/
├── house_price_prediction.py   # Main pipeline script
├── housing_data.csv            # California-style housing dataset (20,640 rows)
├── eda_dashboard.png           # EDA visualizations
├── model_results.png           # Model evaluation charts
└── README.md                   # This file
```

---

## 🔧 Pipeline Steps

### 1. Dataset
- **20,640 samples** · 8 original features
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `Households`, `Latitude`, `Longitude`
- Target: `HousePrice` (median house value in $100k units)

### 2. Exploratory Data Analysis (EDA)
- Statistical summary (mean, std, min/max, quartiles)
- Missing value detection
- Correlation heatmap across all features
- Price distribution, scatter plots, and boxplots by age group

### 3. Feature Engineering
Four new features created to improve model signal:
- `RoomsPerHousehold` — average rooms relative to house age
- `BedroomsPerRoom` — bedroom ratio
- `PopulationPerHousehold` — density metric
- `IncomePerPerson` — income normalized by population

### 4. Preprocessing
- **Missing values** handled via median imputation
- **Feature scaling** using `StandardScaler` (fit on train, transform on test)
- 80/20 train-test split with `random_state=42`

### 5. Model — Linear Regression
- `sklearn.linear_model.LinearRegression`
- 5-fold cross-validation on training set
- Evaluated on held-out test set

### 6. Evaluation Metrics
- **R² Score** — proportion of variance explained
- **MAE** — Mean Absolute Error (average dollar off)
- **RMSE** — Root Mean Squared Error (penalizes large errors)

---

## 🚀 How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Run the pipeline
python house_price_prediction.py
```

---

## 📈 Key Findings

- **Median Income** is by far the strongest predictor (correlation: 0.90)
- **House Age** and **Latitude** also positively influence price
- Engineered feature `IncomePerPerson` adds additional signal
- Model achieves **R² = 0.84** with consistent cross-validation scores

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML pipeline (scaling, model, metrics) |
| Matplotlib / Seaborn | Visualizations |
