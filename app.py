import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as st_stats
import optuna  # For deep hyperparameter tuning

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import SelectFromModel

# Try to import XGBoost and CatBoost; if not installed, we'll skip them.
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

try:
    from catboost import CatBoostClassifier
    cat_installed = True
except ImportError:
    cat_installed = False

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
    weight *= 1 + (0.5 - abs(row.get('trend_weight', 0.5) - 0.5))
    return weight

# -------------------------------
# Helper: Compute Estimated Time Left in Quarter
# -------------------------------
def compute_est_time_left_row(row, expected_plays=40):
    qtr = row["qtr"]
    play_num = row["play_#"]
    play_in_qtr = play_num - (qtr - 1) * expected_plays
    play_in_qtr = max(1, min(play_in_qtr, expected_plays))
    est_time_left = 720 * (1 - (play_in_qtr - 1) / (expected_plays - 1))
    return est_time_left

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv', index_col=False)
    df.reset_index(drop=True, inplace=True)
    df['transformed_yard'] = df['yard_ln']
    df['hash'] = df['hash'].fillna("middle")
    df['off_form'] = df['off_form'].fillna("unknown")
    df['personnel'] = df['personnel'].fillna("unknown")
    df['concept'] = df['concept'].fillna("unknown")

    if 'gain/loss' not in df.columns:
        if 'gain_loss' in df.columns:
            df.rename(columns={'gain_loss': 'gain/loss'}, inplace=True)
        else:
            st.error("The required column 'gain/loss' is missing from the data.")

    df['prev_gain'] = df.groupby("game")["gain/loss"].shift(1).fillna(0)
    df['roll_avg_gain'] = df.groupby("game")["gain/loss"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['play_type_binary'] = df['play_type'].str.upper().apply(lambda x: 1 if x == "PASS" else 0)

    def compute_trend_weight(series):
        return (series.shift(1).expanding().mean()).fillna(0.5)

    df["trend_weight"] = df.groupby("game")["play_type_binary"].transform(compute_trend_weight)

    df['dn_dist'] = df['dn'] * df['dist']
    df['score_qtr'] = df['score_differential'] * df['qtr']

    if "play_#" in df.columns:
        df["max_play_num"] = df.groupby("game")["play_#"].transform("max")
        df["game_progress"] = df["play_#"] / df["max_play_num"]
        df["est_time_left_sec"] = df.apply(lambda row: compute_est_time_left_row(row, expected_plays=40), axis=1)
    if "play_id" in df.columns:
        df["max_play_id"] = df.groupby("game")["play_id"].transform("max")
        df["offensive_play_progress"] = df["play_id"] / df["max_play_id"]

    return df

# -------------------------------
# Load Data and Fit Encoder
# -------------------------------
df = load_data()
features_cat = ['hash', 'off_form', 'personnel', 'concept']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df[features_cat])

# -------------------------------
# Feature Matrix Construction
# -------------------------------
def build_feature_matrix(df_subset):
    features_cat = ['hash', 'off_form', 'personnel', 'concept']
    features_num = ['dn', 'dist', 'qtr', 'score_differential', 'transformed_yard',
                    'prev_gain', 'dn_dist', 'score_qtr', 'roll_avg_gain', 'trend_weight']
    if "est_time_left_sec" in df_subset.columns:
        features_num.append("est_time_left_sec")
    if "offensive_play_progress" in df_subset.columns:
        features_num.append("offensive_play_progress")

    num_df = df_subset[features_num].reset_index(drop=True)
    cat_df = df_subset[features_cat].reset_index(drop=True)
    cat_encoded = pd.DataFrame(ohe.transform(cat_df), columns=ohe.get_feature_names_out(features_cat))
    X = pd.concat([num_df, cat_encoded], axis=1)
    return X

# -------------------------------
# Interactive Selector UI
# -------------------------------
st.sidebar.title("Play Selector")
unique_games = df["game"].unique().tolist()
selected_game = st.sidebar.selectbox("Select Game", unique_games)
game_data = df[df["game"] == selected_game]

play_numbers = game_data["play_#"].dropna().unique().astype(int)
selected_play = st.sidebar.selectbox("Select Play Number", sorted(play_numbers))

# -------------------------------
# Prediction and Evaluation
# -------------------------------
play_row = game_data[game_data["play_#"] == selected_play]
if play_row.empty:
    st.warning("No data for selected play.")
else:
    st.title("SMHS Play Predictor")
    st.subheader("Play Context")
    st.write(play_row.T)

    mask = (df["game"] != selected_game) | ((df["game"] == selected_game) & (df["play_#"] < selected_play))
    filtered_df = df[mask & df["play_type_binary"].notna()]

    X_train = build_feature_matrix(filtered_df)
    y_train = filtered_df["play_type_binary"].astype(int)
    X_test = build_feature_matrix(play_row)

    model = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)),
            ("lgb", LGBMClassifier(n_estimators=300, max_depth=20, learning_rate=0.03, random_state=42))
        ],
        final_estimator=LogisticRegression(solver='lbfgs'),
        cv=3,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    prob = model.predict_proba(X_test)[0][pred]
    play_type = "PASS" if pred == 1 else "RUN"

    st.subheader("Prediction")
    st.write("**Predicted Play Type:**", play_type)
    st.write(f"**Model Confidence:** {prob * 100:.1f}%")

    top_features = pd.Series(model.named_estimators_["rf"].feature_importances_, index=X_train.columns)
    top_features = top_features.sort_values(ascending=False).head(5)
    st.write("**Reasoning Behind Prediction:**")
    for feat, val in top_features.items():
        st.markdown(f"- **{feat}** contributed significantly (importance score: {val:.3f})")

    example_plays = filtered_df[filtered_df['play_type_binary'] == pred].copy()
    target_yard = play_row["transformed_yard"].values[0]
    yard_tolerance = 10
    same_dn = play_row["dn"].values[0]
    same_dist = play_row["dist"].values[0]

    example_plays = example_plays[
        (example_plays["dn"] == same_dn) &
        (abs(example_plays["dist"] - same_dist) <= 1) &
        (abs(example_plays["transformed_yard"] - target_yard) <= yard_tolerance)
    ]

    example_plays["yard_line"] = example_plays["transformed_yard"]
    example_plays["reason"] = example_plays.apply(
        lambda row: f"Similar down ({row['dn']}), distance (~{row['dist']}), yard (~{row['yard_line']})", axis=1
    )

    example_plays = example_plays.sort_values(by="gain/loss", ascending=False).head(3)
    st.write("**Example Similar Plays:**")
    st.dataframe(example_plays[["game", "play_#", "dn", "dist", "yard_line", "play_type", "gain/loss", "reason"]])

    if "play_direction" in df.columns and play_row["play_direction"].notna().any():
        dir_data = df[df["play_direction"].notna()]
        le_dir = LabelEncoder()
        dir_data["dir_encoded"] = le_dir.fit_transform(dir_data["play_direction"])
        dir_features = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain"]
        clf_dir = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_dir.fit(dir_data[dir_features], dir_data["dir_encoded"])
        pred_dir = clf_dir.predict(play_row[dir_features])[0]
        st.write("**Predicted Direction:**", le_dir.inverse_transform([pred_dir])[0])

    st.subheader("Actual Result")
    st.write("**Actual Play Type:**", play_row["play_type"].values[0])
    st.write("**Actual Direction:**", play_row["play_direction"].values[0] if "play_direction" in play_row.columns else "N/A")