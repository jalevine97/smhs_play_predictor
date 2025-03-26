import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut

TEAM = "SMHS"  # Our team

# -------------------------------
# Helper: Compute Sample Weight
# -------------------------------
def compute_sample_weight(row):
    weight = 1.0
    if row.get('td', 0) == 1:
        weight += 1.0
    if row.get('gain/loss', 0) > 0:
        weight += row['gain/loss'] / 10.0
    eff_value = str(row.get('eff', 'N')).strip().upper()
    if eff_value == 'Y':
        weight *= 1.2
    else:
        weight *= 0.8
    return weight

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv')
    df['transformed_yard'] = df['yard_ln']  # using raw yard line
    # Fill missing pre-snap categorical values
    df['hash'] = df['hash'].fillna("middle")
    df['off_form'] = df['off_form'].fillna("unknown")
    df['personnel'] = df['personnel'].fillna("unknown")
    df['concept'] = df['concept'].fillna("unknown")
    # Compute prev_gain per game
    df['prev_gain'] = df.groupby("game")["gain/loss"].shift(1).fillna(0)
    # Create binary label: 0 = RUN, 1 = PASS (based on play_type column)
    df['play_type_binary'] = df['play_type'].str.upper().apply(lambda x: 1 if x == "PASS" else 0)
    # --- New Interaction Features ---
    df['dn_dist'] = df['dn'] * df['dist']
    df['score_qtr'] = df['score_differential'] * df['qtr']
    return df

df = load_data()

# -------------------------------
# One-Hot Encoding for Pre-snap Categorical Features
# -------------------------------
features_cat = ['hash', 'off_form', 'personnel', 'concept']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df[features_cat])

def build_feature_matrix(df_subset):
    """
    Build the feature matrix by combining numeric features (and interactions)
    with one-hot encoded pre-snap categorical features.
    """
    features_num = ['dn', 'dist', 'qtr', 'score_differential', 'transformed_yard',
                    'prev_gain', 'dn_dist', 'score_qtr']
    num_df = df_subset[features_num].reset_index(drop=True)
    cat_df = df_subset[features_cat].reset_index(drop=True)
    cat_encoded = pd.DataFrame(ohe.transform(cat_df), columns=ohe.get_feature_names_out(features_cat))
    X = pd.concat([num_df, cat_encoded], axis=1)
    return X

# -------------------------------
# Helper: Stacking Ensemble for Run/Pass Prediction
# -------------------------------
def get_stack_clf():
    # Tuned parameters (from your offline tuning run)
    tuned_rf = RandomForestClassifier(n_estimators=300, min_samples_split=8, max_depth=None, random_state=42)
    tuned_lgb = LGBMClassifier(n_estimators=300, max_depth=20, learning_rate=0.03, random_state=42)
    return StackingClassifier(
        estimators=[('lgb', tuned_lgb), ('rf', tuned_rf)],
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1
    )

# -------------------------------
# Main App Interface
# -------------------------------
st.title("SMHS Play Predictor Prototype")
st.write(
    "Select a week and a play number. In this version, the model is trained on the entire season's data (all plays) "
    "to predict the play type (RUN vs. PASS) using a stacking ensemble. Pre‑snap cues—including interaction features "
    "and one‑hot encoded formations, personnel, and hash—are used."
)

schedule = {
    "Week 1": "Mission Viejo High School",
    "Week 2": "Centennial High School",
    "Week 3": "Liberty High School",
    "Week 4": "Oaks Christian School",
    "Week 5": "Leuzinger High School",
    "Week 6": "BYE",
    "Week 7": "Mater Dei High School",
    "Week 8": "St. John Bosco High School",
    "Week 9": "Orange Lutheran High School",
    "Week 10": "JSerra Catholic High School",
    "Week 11": "Servite",
    "Week 12": "Inglewood High School",
    "Week 13": "St. John Bosco High School"
}

selected_week = st.sidebar.selectbox("Select Week", list(schedule.keys()))
opponent = schedule[selected_week]

if opponent == "BYE":
    st.error("This is a bye week! No game scheduled.")
else:
    st.sidebar.write(f"**Opponent:** {opponent}")
    game_data = df[df['game'].str.contains(opponent, case=False, na=False)].copy()
    if game_data.empty:
        st.error(f"No data found for opponent: {opponent}")
    else:
        game_data["play_#"] = pd.to_numeric(game_data["play_#"], errors="coerce")
        game_data = game_data.dropna(subset=["play_#"])
        game_data["play_#"] = game_data["play_#"].astype(int)
        game_data = game_data.sort_values("play_#")
        
        max_play = int(game_data["play_#"].max())
        selected_play_num = st.sidebar.number_input("Select Play Number", min_value=1, max_value=max_play, value=1)
        
        selected_play = game_data[game_data["play_#"] == selected_play_num]
        if selected_play.empty:
            st.error("No play found for the selected play number!")
        else:
            raw_yard = selected_play['yard_ln'].iloc[0]
            if raw_yard < 0:
                yard_str = f"SMHS {abs(raw_yard)} Yard Line"
            elif raw_yard > 0:
                yard_str = f"Opponent {raw_yard} Yard Line"
            else:
                yard_str = "Midfield"
            st.subheader(f"Prediction Output from the {yard_str}")
            st.write(selected_play[["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain", "hash"]].to_dict(orient="records")[0])
            
            # -------------------------------
            # Entire-Season Training: Use All Plays
            # -------------------------------
            full_season_data = df.dropna(subset=["play_type_binary"])
            st.subheader("Building Prediction Model for Run vs. Pass (Entire Season)")
            X_train = build_feature_matrix(full_season_data)
            y_train = full_season_data["play_type_binary"].astype(int)
            X_test = build_feature_matrix(selected_play)
            
            stack_clf = get_stack_clf()
            stack_clf.fit(X_train, y_train)
            pred = stack_clf.predict(X_test)[0]
            play_type = "PASS" if pred == 1 else "RUN"
            st.write("**Predicted Play Type:**", play_type)
            pred_prob = stack_clf.predict_proba(X_test)[0]
            confidence = np.max(pred_prob) * 100
            st.write(f"**Model Confidence:** {confidence:.1f}%")
            
            # -------------------------------
            # Optional: Predict Play Direction
            # -------------------------------
            st.subheader("Predicting Play Direction")
            if play_type.upper() == "RUN":
                hist_dir = full_season_data[full_season_data["play_type"].str.upper() == "RUN"]
            else:
                hist_dir = full_season_data[full_season_data["play_type"].str.upper() == "PASS"]
            if len(hist_dir) < 5 or hist_dir["play_direction"].isnull().all():
                st.warning("Not enough data to predict play direction.")
            else:
                hist_dir = hist_dir.dropna(subset=["play_direction"])
                le_dir = LabelEncoder()
                hist_dir['direction_encoded'] = le_dir.fit_transform(hist_dir['play_direction'])
                features_dir = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain"]
                X_train_dir = hist_dir[features_dir]
                y_train_dir = hist_dir['direction_encoded']
                clf_dir = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42)
                clf_dir.fit(X_train_dir, y_train_dir)
                X_test_dir = selected_play[features_dir]
                pred_dir_encoded = clf_dir.predict(X_test_dir)[0]
                pred_direction = le_dir.inverse_transform([pred_dir_encoded])[0]
                st.write("**Predicted Play Direction:**", pred_direction)

# -------------------------------
# Season-Wide Evaluation & Final Model Retraining
# -------------------------------
st.markdown("---")
st.subheader("Evaluate & Retrain on Entire File (Season-Wide)")

if st.button("Run Historical Evaluation & Retrain Final Model"):
    total_correct = 0
    total_predictions = 0
    logo = LeaveOneGroupOut()
    # Use all season data with non-missing play_type_binary
    eval_data = df.dropna(subset=['dn', 'dist', 'qtr', 'score_differential',
                                    'transformed_yard', 'prev_gain', 'play_type_binary']).reset_index(drop=True)
    X_eval = build_feature_matrix(eval_data)
    y_eval = eval_data["play_type_binary"].astype(int)
    groups_eval = eval_data["game"]
    
    for train_idx, test_idx in logo.split(X_eval, y_eval, groups=groups_eval):
        if len(train_idx) < 5:
            continue
        stack_clf = get_stack_clf()
        stack_clf.fit(X_eval.iloc[train_idx], y_eval.iloc[train_idx])
        preds = stack_clf.predict(X_eval.iloc[test_idx])
        total_correct += (preds == y_eval.iloc[test_idx]).sum()
        total_predictions += len(test_idx)
    
    accuracy = total_correct / total_predictions * 100 if total_predictions > 0 else 0
    st.write(f"Historical Evaluation: {total_correct} correct out of {total_predictions} predictions. Accuracy: {accuracy:.1f}%")
    
    final_model = get_stack_clf()
    final_model.fit(X_eval, y_eval)
    st.write("Final ensemble model retrained on the entire dataset for future predictions.")
