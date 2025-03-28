# hyperparameter-tuning.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# 1️⃣ Load + preprocess exactly as in your app
df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv')
df['prev_gain'] = df.groupby('game')['gain/loss'].shift(1).fillna(0)
df['hash'] = df['hash'].fillna('middle')
for col in ['hash','off_form','personnel','concept']:
    df[col] = df[col].fillna('unknown')
    df[f'{col}_enc'] = LabelEncoder().fit_transform(df[col])
df['game_seconds_remaining'] = pd.to_numeric(df['game_seconds_remaining'], errors='coerce').fillna(0)
df['transformed_yard'] = df['yard_ln']

# 2️⃣ Define features & target
features = ["dn","dist","qtr","score_differential","transformed_yard","prev_gain",
            "hash_enc","off_form_enc","personnel_enc","concept_enc","game_seconds_remaining"]
target = 'play_type_encoded'
data = df.dropna(subset=features + [target])
X = data[features]
y = data[target].astype(int)

# 3️⃣ GridSearch over RandomForest hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)

print("✅ Best parameters:", grid.best_params_)
print(f"✅ Cross‑validated accuracy: {grid.best_score_ * 100:.2f}%")
