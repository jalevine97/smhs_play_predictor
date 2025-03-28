import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 🔧 — UPDATE THIS PATH to wherever your CSV actually lives
df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv')

# BASIC PREPROCESSING
df['prev_gain'] = df.groupby('game')['gain/loss'].shift(1).fillna(0)
df['hash'] = df['hash'].fillna('middle')
for col in ['hash','off_form','personnel','concept']:
    df[col] = df[col].fillna('unknown')
    df[f'{col}_enc'] = LabelEncoder().fit_transform(df[col])
df['game_seconds_remaining'] = pd.to_numeric(df['game_seconds_remaining'], errors='coerce').fillna(0)

# FEATURE SET — added formation, personnel, concept, game time
features = ["dn","dist","qtr","score_differential","yard_ln","prev_gain",
            "hash_enc","off_form_enc","personnel_enc","concept_enc","game_seconds_remaining"]
target = 'play_type_encoded'

total_correct = 0
total = 0

for _, game_df in df.groupby('game'):
    game_df = game_df.sort_values('play_#')
    for i in range(5, len(game_df)):
        train = game_df.iloc[:i].dropna(subset=features+[target])
        test = game_df.iloc[i:i+1]
        if train.empty: 
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(train[features], train[target].astype(int))
        pred = clf.predict(test[features])[0]
        total += 1
        if pred == test[target].iloc[0]:
            total_correct += 1

print(f"New Feature Accuracy → {total_correct}/{total} = {total_correct/total*100:.1f}%")
