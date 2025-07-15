import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load and clean dataset
file_path = 'data/melb_data.csv'
data = pd.read_csv(file_path).dropna(axis=0)

# Target and features
y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = data[features]

# Train-validation split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Decision Tree model
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(train_X, train_y)
dt_preds = dt_model.predict(val_X)
print("Decision Tree MAE:", mean_absolute_error(val_y, dt_preds))

# Random Forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_preds = rf_model.predict(val_X)
print("Random Forest MAE:", mean_absolute_error(val_y, rf_preds))

# Try different tree sizes (leaf nodes)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

print("\n--- Tuning max_leaf_nodes ---")
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max leaf nodes: {max_leaf_nodes} \t MAE: {mae:.0f}")
