# ============================================================
# House Price Prediction - California Housing Dataset
# Tools: Python | Scikit-learn | Pandas | Linear Regression
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ────────────────────────────────────────────────
# 1. LOAD DATASET
# ────────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION — CALIFORNIA HOUSING DATASET")
print("=" * 60)

df = pd.read_csv('housing_data.csv')

print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFeatures: {list(df.columns)}")

# ────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("─" * 60)

print("\n📊 First 5 rows:")
print(df.head().to_string())

print("\n📈 Statistical Summary:")
print(df.describe().round(2).to_string())

print("\n🔍 Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  No missing values found.")

print(f"\n🏠 House Price Range: ${df['HousePrice'].min()*100_000:,.0f} – ${df['HousePrice'].max()*100_000:,.0f}")
print(f"   Mean Price       : ${df['HousePrice'].mean()*100_000:,.0f}")
print(f"   Median Price     : ${df['HousePrice'].median()*100_000:,.0f}")

# ────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 2: FEATURE ENGINEERING")
print("─" * 60)

# New features
df['RoomsPerHousehold']    = df['AveRooms']    / df['HouseAge'].replace(0, 1)
df['BedroomsPerRoom']      = df['AveBedrms']   / df['AveRooms'].replace(0, 1)
df['PopulationPerHousehold'] = df['Population'] / df['Households'].replace(0, 1)
df['IncomePerPerson']      = df['MedInc']      / df['Population'].replace(0, 1) * 1000

# Clip extreme outliers
for col in ['RoomsPerHousehold', 'PopulationPerHousehold']:
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper)

print("  ✅ Created 4 new engineered features:")
print("     • RoomsPerHousehold")
print("     • BedroomsPerRoom")
print("     • PopulationPerHousehold")
print("     • IncomePerPerson")

# ────────────────────────────────────────────────
# 4. HANDLE MISSING VALUES (post-engineering)
# ────────────────────────────────────────────────
df.fillna(df.median(numeric_only=True), inplace=True)
print("\n  ✅ Missing values handled (median imputation).")

# ────────────────────────────────────────────────
# 5. CORRELATION ANALYSIS
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 3: CORRELATION WITH TARGET")
print("─" * 60)

corr = df.corr(numeric_only=True)['HousePrice'].sort_values(ascending=False)
print(f"\n{'Feature':<30} {'Correlation':>12}")
print("-" * 44)
for feat, val in corr.items():
    bar = "█" * int(abs(val) * 20)
    sign = "+" if val >= 0 else "-"
    print(f"  {feat:<28} {sign}{abs(val):.4f}  {bar}")

# ────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT & SCALING
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 4: TRAIN/TEST SPLIT & FEATURE SCALING")
print("─" * 60)

feature_cols = [c for c in df.columns if c != 'HousePrice']
X = df[feature_cols]
y = df['HousePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Training samples : {X_train.shape[0]:,}")
print(f"  Testing  samples : {X_test.shape[0]:,}")
print(f"  Features used    : {X_train.shape[1]}")
print("  ✅ StandardScaler applied.")

# ────────────────────────────────────────────────
# 7. TRAIN LINEAR REGRESSION MODEL
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 5: MODEL TRAINING — LINEAR REGRESSION")
print("─" * 60)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("  ✅ Model trained.")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n  5-Fold Cross-Validation R² Scores:")
for i, s in enumerate(cv_scores, 1):
    print(f"    Fold {i}: {s:.4f}")
print(f"  Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ────────────────────────────────────────────────
# 8. EVALUATION METRICS
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 6: MODEL EVALUATION")
print("─" * 60)

y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n  {'Metric':<30} {'Value':>10}")
print("  " + "─" * 42)
print(f"  {'R² Score':<30} {r2:>10.4f}")
print(f"  {'Mean Absolute Error (MAE)':<30} {mae:>10.4f}  (×$100k = ${mae*100_000:,.0f})")
print(f"  {'Root Mean Sq. Error (RMSE)':<30} {rmse:>10.4f}  (×$100k = ${rmse*100_000:,.0f})")

if r2 >= 0.85:
    print(f"\n  🎉 Target achieved! R² = {r2:.4f} ≥ 0.85")
else:
    print(f"\n  ℹ️  R² = {r2:.4f}")

# Feature coefficients
print("\n  📌 Top Feature Coefficients:")
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
print(coef_df.to_string(index=False))

# ────────────────────────────────────────────────
# 9. VISUALIZATIONS
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 7: GENERATING VISUALIZATIONS")
print("─" * 60)

plt.style.use('seaborn-v0_8-whitegrid')
BLUE  = '#2563EB'
GREEN = '#16A34A'
RED   = '#DC2626'
GRAY  = '#6B7280'

# ── Fig 1: EDA Dashboard ──────────────────────────────────
fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle('House Price Prediction — EDA Dashboard', fontsize=18, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.4, wspace=0.35)

# (a) Price distribution
ax1 = fig1.add_subplot(gs[0, 0])
ax1.hist(df['HousePrice'] * 100_000, bins=50, color=BLUE, edgecolor='white', alpha=0.85)
ax1.set_title('House Price Distribution', fontweight='bold')
ax1.set_xlabel('Price (USD)')
ax1.set_ylabel('Frequency')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e5:.1f}L'))

# (b) Correlation heatmap
ax2 = fig1.add_subplot(gs[0, 1:])
corr_matrix = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax2, annot_kws={'size': 7}, linewidths=0.5)
ax2.set_title('Feature Correlation Heatmap', fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.tick_params(axis='y', rotation=0, labelsize=8)

# (c) MedInc vs Price
ax3 = fig1.add_subplot(gs[1, 0])
ax3.scatter(df['MedInc'], df['HousePrice'], alpha=0.1, color=BLUE, s=5)
ax3.set_title('Median Income vs Price', fontweight='bold')
ax3.set_xlabel('Median Income (scaled)')
ax3.set_ylabel('House Price (×$100k)')

# (d) House Age vs Price
ax4 = fig1.add_subplot(gs[1, 1])
ax4.scatter(df['HouseAge'], df['HousePrice'], alpha=0.1, color=GREEN, s=5)
ax4.set_title('House Age vs Price', fontweight='bold')
ax4.set_xlabel('House Age (years)')
ax4.set_ylabel('House Price (×$100k)')

# (e) Boxplot by age bucket
ax5 = fig1.add_subplot(gs[1, 2])
df['AgeBucket'] = pd.cut(df['HouseAge'], bins=[0,10,20,30,40,55], labels=['0-10','10-20','20-30','30-40','40-55'])
df.boxplot(column='HousePrice', by='AgeBucket', ax=ax5, patch_artist=True,
           boxprops=dict(facecolor=BLUE, alpha=0.6))
ax5.set_title('Price by House Age Group', fontweight='bold')
ax5.set_xlabel('Age Bucket (years)')
ax5.set_ylabel('House Price (×$100k)')
plt.sca(ax5); plt.title('Price by House Age Group', fontweight='bold')
fig1.text(0.5, -0.01, 'Source: California Housing Dataset (scikit-learn)', ha='center', color=GRAY, fontsize=9)

plt.savefig('/home/claude/house_price_prediction/eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ EDA Dashboard saved.")

# ── Fig 2: Model Results Dashboard ───────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle('Linear Regression — Model Results', fontsize=16, fontweight='bold')

# (a) Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.3, color=BLUE, s=8, label='Predictions')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, '--', color=RED, linewidth=2, label='Perfect fit')
ax.set_xlabel('Actual Price (×$100k)')
ax.set_ylabel('Predicted Price (×$100k)')
ax.set_title('Actual vs Predicted Prices', fontweight='bold')
ax.legend()
ax.text(0.05, 0.92, f'R² = {r2:.4f}', transform=ax.transAxes,
        fontsize=11, color=GREEN, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREEN))

# (b) Residuals
residuals = y_test - y_pred
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.3, color=BLUE, s=8)
ax.axhline(0, color=RED, linewidth=2, linestyle='--')
ax.set_xlabel('Predicted Price (×$100k)')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot', fontweight='bold')

# (c) Residual distribution
ax = axes[1, 0]
ax.hist(residuals, bins=60, color=BLUE, edgecolor='white', alpha=0.85)
ax.axvline(0, color=RED, linewidth=2, linestyle='--')
ax.set_xlabel('Residual Value')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution', fontweight='bold')

# (d) Feature Importance (abs coefficients)
ax = axes[1, 1]
coef_plot = coef_df.head(10).copy()
colors = [GREEN if c > 0 else RED for c in coef_plot['Coefficient']]
ax.barh(coef_plot['Feature'], coef_plot['Coefficient'].abs(), color=colors, edgecolor='white')
ax.set_xlabel('|Coefficient| (standardized)')
ax.set_title('Top Feature Importances', fontweight='bold')
ax.invert_yaxis()

# Metrics box
metrics_text = f"R² Score : {r2:.4f}\nMAE      : {mae:.4f} (${mae*100_000:,.0f})\nRMSE     : {rmse:.4f} (${rmse*100_000:,.0f})"
fig2.text(0.5, -0.02, metrics_text, ha='center', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#EFF6FF', edgecolor=BLUE),
          family='monospace')

plt.tight_layout()
plt.savefig('/home/claude/house_price_prediction/model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Model Results Dashboard saved.")

# ────────────────────────────────────────────────
# 10. SAMPLE PREDICTIONS
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 8: SAMPLE PREDICTIONS")
print("─" * 60)

sample = X_test.iloc[:5]
sample_scaled = scaler.transform(sample)
preds = model.predict(sample_scaled)
actuals = y_test.iloc[:5].values

print(f"\n  {'#':<4} {'Actual ($)':>14} {'Predicted ($)':>14} {'Error ($)':>12}")
print("  " + "─" * 48)
for i, (a, p) in enumerate(zip(actuals, preds), 1):
    err = abs(a - p) * 100_000
    print(f"  {i:<4} ${a*100_000:>12,.0f} ${p*100_000:>12,.0f} ${err:>10,.0f}")

print("\n" + "=" * 60)
print("  ✅ PROJECT COMPLETE")
print(f"  R² Score : {r2:.4f}  |  MAE: ${mae*100_000:,.0f}  |  RMSE: ${rmse*100_000:,.0f}")
print("=" * 60)
