from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load and split data
df = pd.read_csv('insurance.csv')
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Train model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

# Evaluate
preds = model.predict(X_valid)
print("Mean Absolute Error:", mean_absolute_error(y_valid, preds))

# Model Performance Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE:", mean_absolute_error(y_valid, preds))
print("RMSE:", mean_squared_error(y_valid, preds, squared=False))
print("R^2:", r2_score(y_valid, preds))
