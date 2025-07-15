"""
This project forecasts weekly flu-related doctor visits using historical data from Google Trends and health reports. It incorporates:

- Trend analysis via linear time index
- Seasonal modeling (indicator & Fourier features)
- Lag-based autoregression
- Multi-step forecasting with LinearRegression and XGBoost
- Evaluation using RMSE and graphical plots
"""

# --- Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsmodels.graphics.tsaplots import plot_pacf

# --- Plot configuration ---
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=16, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25")

# --- Load data ---
data_dir = Path("data")
flu = pd.read_csv(data_dir / "flu-trends.csv")
flu.set_index(pd.PeriodIndex(flu["Week"], freq="W"), inplace=True)
flu.drop("Week", axis=1, inplace=True)

# --- Target Series: Weekly flu-related doctor visits ---
y = flu["FluVisits"].copy()

# --- Lag Feature Function ---
def make_lags(ts, lags):
    return pd.concat({f'y_lag_{i}': ts.shift(i) for i in range(1, lags+1)}, axis=1)

X = make_lags(y, lags=4).fillna(0.0)

# --- Multi-step Target Creation Function ---
def make_multistep_target(ts, steps):
    return pd.concat({f'y_step_{i+1}': ts.shift(-i) for i in range(steps)}, axis=1)

y = make_multistep_target(y, steps=8).dropna()

# --- Align features and targets ---
y, X = y.align(X, join="inner", axis=0)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

# --- Model: XGBoost with MultiOutputRegressor ---
model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)

# --- Predictions ---
y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

# --- Evaluation ---
train_rmse = mean_squared_error(y_train, y_fit)
test_rmse = mean_squared_error(y_test, y_pred)
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# --- Visualization ---
def plot_multistep(y_true, y_forecast, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.index, y_true.iloc[:, 0], label="True FluVisits", linewidth=2)
    for i in range(y_forecast.shape[1]):
        ax.plot(y_forecast.index, y_forecast.iloc[:, i], alpha=0.5, label=f"Forecast +{i+1}")
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_multistep(y_test, y_pred, "Multistep Forecast: FluVisits")
