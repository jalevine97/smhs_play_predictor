import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

TEAM = "SMHS"  # Our team

# In this version, we simply use the raw value from 'yard_ln'
def transform_yard_line_row(row):
    return row['yard_ln']

def compute_sample_weight(row):
    """
    Computes a sample weight for a historical play based on its success.
    - Base weight is 1.
    - If the play resulted in a touchdown (td == 1), add 1.
    - If the play gained positive yards, add (gain/loss) divided by 10.
    - Adjust weight based on efficiency: if eff == 'Y' (good), multiply by 1.2;
      if eff == 'N', multiply by 0.8.
    """
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

@st.cache_data
def load_data():
    df = pd.read_csv('data/sm_cleaned_with_correct_fields.csv')
    # Use the raw 'yard_ln' values directly
    df['transformed_yard'] = df['yard_ln']
    return df

df = load_data()

st.title("SMHS Play Predictor Prototype")
st.write(
    "Select a week and a play number. The system will use all past plays from that game "
    "(only those with play numbers lower than the selected play) to predict the play type "
    "and, if possible, the play direction. Previous play outcomes are now incorporated via "
    "the 'prev_gain' feature. Historical success (gain/loss, touchdowns, efficiency) is also considered."
)

# --- 2024 Opponent Schedule with Clarifications ---
schedule = {
    "Week 1": "Mission Viejo High School",       # MVHS
    "Week 2": "Centennial High School",            # CENTENNIAL
    "Week 3": "Liberty High School",               # LIBERTY
    "Week 4": "Oaks Christian School",             # OAKS
    "Week 5": "Leuzinger High School",             # LEUZINGER
    "Week 6": "BYE",                               # BYE (NO GAME)
    "Week 7": "Mater Dei High School",             # MDHS
    "Week 8": "St. John Bosch High School",        # SJB
    "Week 9": "Orange Lutheran High School",       # OLU
    "Week 10": "JSerra Catholic High School",       # JSERRA
    "Week 11": "Servite",                          # SERVITE
    "Week 12": "Inglewood High School",             # INGLEWOOD
    "Week 13": "St. John Bosch High School"         # SJB (second game)
}

# Sidebar: Week selection
selected_week = st.sidebar.selectbox("Select Week", list(schedule.keys()))
opponent = schedule[selected_week]

if opponent == "BYE":
    st.error("This is a bye week! No game scheduled.")
else:
    st.sidebar.write(f"**Opponent:** {opponent}")
    
    # --- Filter Data for the Selected Opponent ---
    game_data = df[df['game'].str.contains(opponent, case=False, na=False)].copy()
    
    if game_data.empty:
        st.error(f"No data found for opponent: {opponent}")
    else:
        # Ensure play numbers are numeric and sort by play number.
        game_data["play_#"] = pd.to_numeric(game_data["play_#"], errors="coerce")
        game_data = game_data.dropna(subset=["play_#"])
        game_data["play_#"] = game_data["play_#"].astype(int)
        game_data = game_data.sort_values("play_#")
        
        # Compute previous play outcome as the gain/loss from the previous play.
        # For the first play, fill with 0.
        game_data["prev_gain"] = game_data["gain/loss"].shift(1).fillna(0)
        
        # Sidebar: Play number selection
        max_play = int(game_data["play_#"].max())
        selected_play_num = st.sidebar.number_input("Select Play Number", min_value=1, max_value=max_play, value=1)
        
        # Retrieve the selected play's game state.
        selected_play = game_data[game_data["play_#"] == selected_play_num]
        if selected_play.empty:
            st.error("No play found for the selected play number!")
        else:
            # Retrieve the raw yard line for display
            raw_yard = selected_play['yard_ln'].iloc[0]
            if raw_yard < 0:
                yard_str = f"SMHS {abs(raw_yard)} Yard Line"
            elif raw_yard > 0:
                yard_str = f"Opponent {raw_yard} Yard Line"
            else:
                yard_str = "Midfield"
            
            st.subheader(f"Prediction Output from the {yard_str}")
            st.write(selected_play[["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain"]].to_dict(orient="records")[0])
            
            # --- Filter Historical Data for Play Type Prediction ---
            historical_data = game_data[game_data["play_#"] < selected_play_num]
            
            if len(historical_data) < 5:
                st.warning("Not enough historical plays in this game to build a predictive model.")
            else:
                st.subheader("Building Prediction Model for Play Type")
                # Include 'prev_gain' as an additional feature.
                features = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain"]
                target = "play_type_encoded"  # 0 = RUN, 1 = PASS
                
                historical_data = historical_data.dropna(subset=features + [target])
                X_train = historical_data[features]
                y_train = historical_data[target].astype(int)
                
                # Compute sample weights incorporating historical success and efficiency.
                sample_weights = historical_data.apply(compute_sample_weight, axis=1)
                
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train, sample_weight=sample_weights)
                
                X_test = selected_play[features]
                pred = clf.predict(X_test)[0]
                play_type = "RUN" if pred == 0 else "PASS"
                
                st.write("**Predicted Play Type:**", play_type)
                
                pred_prob = clf.predict_proba(X_test)[0]
                confidence = np.max(pred_prob) * 100
                st.write(f"**Play Type Model Confidence:** {confidence:.1f}%")
                
                # --- Build and Predict Play Direction ---
                st.subheader("Predicting Play Direction")
                if play_type == "RUN":
                    historical_direction_data = historical_data[historical_data["play_type"].str.lower() == "run"]
                else:
                    historical_direction_data = historical_data[historical_data["play_type"].str.lower() == "pass"]
                
                if len(historical_direction_data) < 5 or historical_direction_data["play_direction"].isnull().all():
                    st.warning("Not enough historical data to predict play direction.")
                else:
                    historical_direction_data = historical_direction_data.dropna(subset=["play_direction"])
                    
                    le = LabelEncoder()
                    historical_direction_data['direction_encoded'] = le.fit_transform(historical_direction_data['play_direction'])
                    
                    features_direction = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain"]
                    X_train_dir = historical_direction_data[features_direction]
                    y_train_dir = historical_direction_data['direction_encoded']
                    
                    clf_dir = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf_dir.fit(X_train_dir, y_train_dir)
                    
                    X_test_dir = selected_play[features_direction]
                    pred_dir_encoded = clf_dir.predict(X_test_dir)[0]
                    pred_direction = le.inverse_transform([pred_dir_encoded])[0]
                    
                    st.write("**Predicted Play Direction:**", pred_direction)
