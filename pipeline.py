import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_csv('data/melb_data.csv')
y = data.Price
X = data.drop(['Price'], axis=1)

# Train-validation split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)

# Identify categorical and numerical columns
categorical_cols = [c for c in X_train_full.columns if X_train_full[c].nunique() < 10 and X_train_full[c].dtype == "object"]
numerical_cols = [c for c in X_train_full.columns if X_train_full[c].dtype in ['int64', 'float64']]
selected_cols = categorical_cols + numerical_cols

X_train = X_train_full[selected_cols].copy()
X_valid = X_valid_full[selected_cols].copy()

# Define preprocessing for numeric and categorical data
num_transformer = SimpleImputer(strategy='constant')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Train model
pipeline.fit(X_train, y_train)

# Make predictions
preds = pipeline.predict(X_valid)

# Evaluate
score = mean_absolute_error(y_valid, preds)
print('Pipeline MAE:', score)
