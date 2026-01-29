import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import GammaRegressor

warnings.filterwarnings('ignore')

# 1. Load Data
try:
    df = pd.read_excel('06 - CAS Predictive Modeling Case Competition- Dataset.xlsx')
except FileNotFoundError:
    print("ERROR: Could not find the Excel file. Make sure it is in the same folder!")
    exit()

# 2. Prepare Data 
amount_col = 'amount'
df_model = pd.get_dummies(df, columns=['greek', 'off_campus', 'sprinklered', 'coverage'], drop_first=True)

# Select numerical columns
feature_cols = [c for c in df_model.columns if c not in ['policy_id', 'amount', 'has_claim'] 
                and df_model[c].dtype in ['int64', 'float64', 'uint8']]

X = df_model[feature_cols].fillna(0)
y_sev = df_model[amount_col]

# Filter for Severity Analysis (Only look at claims > 0)
mask = y_sev > 0
X_sev = X[mask]
y_sev = y_sev[mask]

print(f"Data ready. Testing optimization on {len(X_sev)} claims...")

# 3. The Grid Search (finding the best Alpha)
# We test 3 values: 
# alpha=0 (Standard - what you have now)
# alpha=0.1 (Light regularization)
# alpha=1.0 (Heavy regularization)
param_grid = {'alpha': [0, 0.01, 0.1, 1.0]} 

print("\nRunning Grid Search (this might take 10 seconds)...")
grid = GridSearchCV(GammaRegressor(max_iter=1000), param_grid, cv=5, scoring='neg_mean_gamma_deviance')
grid.fit(X_sev, y_sev)

# 4. Results
print("\n" + "="*40)
print("OPTIMIZATION RESULTS")
print("="*40)
print(f"Best Alpha Parameter: {grid.best_params_['alpha']}")
print(f"Best Score (Deviance): {grid.best_score_:.4f}")

if grid.best_params_['alpha'] == 0:
    print("\nCONCLUSION: The standard model (Alpha=0) is optimal.")
    print("You do NOT need to change your code.")
else:
    print(f"\nCONCLUSION: The optimal alpha parameter={grid.best_params_['alpha']}")