import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneGroupOut

df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv').sort_values(['game','play_#'])
df['prev_gain'] = df.groupby('game_id')['gain/loss'].shift(1).fillna(0)
df.fillna({'hash':'middle','off_form':'unknown','personnel':'unknown','concept':'unknown'}, inplace=True)

# Drop any rows missing required fields BEFORE encoding
data = df.dropna(subset=['dn','dist','qtr','score_differential','yard_ln','prev_gain','play_type_encoded']).reset_index(drop=True)

# One‑hot encode categorical pre‑snap features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(data[['hash','off_form','personnel','concept']])

# Combine numeric + one‑hot into final feature matrix
X_num = data[['dn','dist','qtr','score_differential','yard_ln','prev_gain']]
X = pd.concat([X_num.reset_index(drop=True), pd.DataFrame(X_cat)], axis=1)

y = data['play_type_encoded'].astype(int)
groups = data['game']

logo = LeaveOneGroupOut()
correct = total = 0

for train_idx, test_idx in logo.split(X, y, groups=groups):
    if len(train_idx) < 5:
        continue
    model = LGBMClassifier(n_estimators=200, max_depth=-1, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])[0]
    total += 1
    if pred == y.iloc[test_idx].iat[0]:
        correct += 1

print(f"New eval accuracy: {correct}/{total} = {correct/total*100:.1f}%")
