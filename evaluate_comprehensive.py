import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneGroupOut

# Load and sort your dataset by game and play number
df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv').sort_values(['game', 'play_#'])

# Preprocess: compute previous play gain and fill missing categorical values
df['prev_gain'] = df.groupby('game')['gain/loss'].shift(1).fillna(0)
df.fillna({'hash': 'middle', 'off_form': 'unknown', 'personnel': 'unknown', 'concept': 'unknown'}, inplace=True)

# Drop rows missing essential features or target
data = df.dropna(subset=['dn', 'dist', 'qtr', 'score_differential', 'yard_ln', 'prev_gain', 'play_type_encoded']).reset_index(drop=True)

# Define numeric and categorical feature columns
features_num = ['dn', 'dist', 'qtr', 'score_differential', 'yard_ln', 'prev_gain']
features_cat = ['hash', 'off_form', 'personnel', 'concept']

# One-hot encode categorical features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(data[features_cat])
X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(features_cat))

# Combine numeric and one-hot encoded features
X_num = data[features_num].reset_index(drop=True)
X = pd.concat([X_num, X_cat_df], axis=1)

# Target and groups
y = data['play_type_encoded'].astype(int)
groups = data['game']  # Use the 'game' column as the grouping variable

# Set up Leave-One-Group-Out cross-validation
logo = LeaveOneGroupOut()

total_correct = 0
total_samples = 0

# Loop through each group (game)
for train_idx, test_idx in logo.split(X, y, groups=groups):
    # Skip training if too few samples (adjust this threshold if needed)
    if len(train_idx) < 5:
        continue
    # Train LightGBM on the training set
    model = LGBMClassifier(n_estimators=200, max_depth=-1, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    # Predict on the test set
    preds = model.predict(X.iloc[test_idx])
    total_correct += (preds == y.iloc[test_idx]).sum()
    total_samples += len(test_idx)

# Calculate overall season-wide accuracy
accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
print(f"Season-wide evaluation accuracy: {total_correct}/{total_samples} = {accuracy:.1f}%")
