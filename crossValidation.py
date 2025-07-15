from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

# Load data
data = pd.read_csv('data/AER_credit_card_data.csv', true_values=['yes'], false_values=['no'])
X = data.drop('card', axis=1)
y = data['card']

# Set up pipeline
pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("Initial CV accuracy:", scores.mean())

# Remove known leaky predictors
leaky_features = ['expenditure', 'share', 'active', 'majorcards']
X_filtered = X.drop(leaky_features, axis=1)
scores_filtered = cross_val_score(pipeline, X_filtered, y, cv=5, scoring='accuracy')
print("CV accuracy after leakage fix:", scores_filtered.mean())
