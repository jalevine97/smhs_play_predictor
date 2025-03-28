import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

TEAM = "SMHS"  # Our team

# Use the raw 'yard_ln' values as our "transformed_yard"
def transform_yard_line_row(row):
    return row['yard_ln']

def compute_sample_weight(row):
    """
    Computes a sample weight for a historical play based on its success.
      - Base weight is 1.
      - If the play resulted in a touchdown (td == 1), add 1.
      - If the play gained positive yards, add (gain/loss) divided by 10.
      - Adjust weight based on efficiency: if eff == 'Y', multiply by 1.2; if eff == 'N', multiply by 0.8.
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
    # Use raw 'yard_ln' as our "transformed_yard"
    df['transformed_yard'] = df['yard_ln']
    # Ensure the hash column exists; fill missing values with "middle"
    df['hash'] = df['hash'].fillna("middle")
    # Encode the hash column ("left", "middle", "right") into numeric values
    hash_encoder = LabelEncoder()
    df['hash_encoded'] = hash_encoder.fit_transform(df['hash'])
    # Compute prev_gain for each game by grouping by "game" and shifting "gain/loss"
    df['prev_gain'] = df.groupby("game")["gain/loss"].shift(1).fillna(0)
    # Process play_category: fill missing values and encode
    df['play_category'] = df['play_category'].fillna("unknown")
    cat_encoder = LabelEncoder()
    df['play_category_encoded'] = cat_encoder.fit_transform(df['play_category'])
    return df, cat_encoder

df, cat_encoder = load_data()

st.title("SMHS Play Predictor Prototype")
st.write(
    "Select a week and a play number. The system uses all past plays from that game "
    "(only those with play numbers lower than the selected play) to predict the play type, "
    "play category, and, if possible, the play direction. Previous play outcomes (prev_gain), hash (snap location), "
    "and historical success (gain/loss, touchdowns, efficiency) are all considered."
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
    "Week 8": "St. John Bosco High School",        # Corrected name
    "Week 9": "Orange Lutheran High School",       # OLU
    "Week 10": "JSerra Catholic High School",       # JSERRA
    "Week 11": "Servite",                          # SERVITE
    "Week 12": "Inglewood High School",             # INGLEWOOD
    "Week 13": "St. John Bosco High School"         # Corrected name, second game
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
        
        # Sidebar: Play number selection
        max_play = int(game_data["play_#"].max())
        selected_play_num = st.sidebar.number_input("Select Play Number", min_value=1, max_value=max_play, value=1)
        
        # Retrieve the selected play's game state.
        selected_play = game_data[game_data["play_#"] == selected_play_num]
        if selected_play.empty:
            st.error("No play found for the selected play number!")
        else:
            # Display the raw yard_ln with a label.
            raw_yard = selected_play['yard_ln'].iloc[0]
            if raw_yard < 0:
                yard_str = f"SMHS {abs(raw_yard)} Yard Line"
            elif raw_yard > 0:
                yard_str = f"Opponent {raw_yard} Yard Line"
            else:
                yard_str = "Midfield"
            
            st.subheader(f"Prediction Output from the {yard_str}")
            st.write(selected_play[["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain", "hash"]].to_dict(orient="records")[0])
            
            # --- Filter Historical Data for Play Type Prediction ---
            historical_data = game_data[game_data["play_#"] < selected_play_num]
            
            if len(historical_data) < 5:
                st.warning("Not enough historical plays in this game to build a predictive model.")
            else:
                st.subheader("Building Prediction Model for Play Type")
                # Update feature list to include hash_encoded.
                features = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain", "hash_encoded"]
                target = "play_type_encoded"  # 0 = RUN, 1 = PASS
                
                historical_data = historical_data.dropna(subset=features + [target])
                X_train = historical_data[features]
                y_train = historical_data[target].astype(int)
                
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
                
                st.subheader("Building Prediction Model for Play Category")
                # Build a model to predict the play category.
                features_cat = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain", "hash_encoded"]
                target_cat = "play_category_encoded"
                historical_data_cat = historical_data.dropna(subset=features_cat + [target_cat])
                if len(historical_data_cat) < 5:
                    st.warning("Not enough historical plays to build a predictive model for play category.")
                else:
                    X_train_cat = historical_data_cat[features_cat]
                    y_train_cat = historical_data_cat[target_cat].astype(int)
                    clf_cat = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf_cat.fit(X_train_cat, y_train_cat)
                    X_test_cat = selected_play[features_cat]
                    pred_cat_encoded = clf_cat.predict(X_test_cat)[0]
                    pred_cat = cat_encoder.inverse_transform([pred_cat_encoded])[0]
                    st.write("**Predicted Play Category:**", pred_cat)
                
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
                    
                    features_direction = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain", "hash_encoded"]
                    X_train_dir = historical_direction_data[features_direction]
                    y_train_dir = historical_direction_data['direction_encoded']
                    
                    clf_dir = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf_dir.fit(X_train_dir, y_train_dir)
                    
                    X_test_dir = selected_play[features_direction]
                    pred_dir_encoded = clf_dir.predict(X_test_dir)[0]
                    pred_direction = le.inverse_transform([pred_dir_encoded])[0]
                    
                    st.write("**Predicted Play Direction:**", pred_direction)

# --- New Section: Evaluate & Retrain on Entire File ---
st.markdown("---")
st.subheader("Evaluate & Retrain on Entire File")

if st.button("Run Historical Evaluation & Retrain Final Model"):
    total_correct = 0
    total_predictions = 0

    # Group the data by game so that only past plays are used for prediction in each game.
    for game, game_df in df.groupby("game"):
        game_df = game_df.sort_values("play_#")
        # Compute prev_gain for each game (should already exist, but ensure it's computed)
        game_df["prev_gain"] = game_df.groupby("game")["gain/loss"].shift(1).fillna(0)
        features_eval = ["dn", "dist", "qtr", "score_differential", "transformed_yard", "prev_gain", "hash_encoded"]
        target_eval = "play_type_encoded"
        # For each play after the first few, predict using historical data only.
        for i in range(1, len(game_df)):
            train_data = game_df.iloc[:i]
            test_data = game_df.iloc[i:i+1]
            if len(train_data) < 5:
                continue  # Not enough training data for this game
            train_data = train_data.dropna(subset=features_eval + [target_eval])
            if train_data.empty:
                continue
            X_train_eval = train_data[features_eval]
            y_train_eval = train_data[target_eval].astype(int)
            sample_weights_eval = train_data.apply(compute_sample_weight, axis=1)
            clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_temp.fit(X_train_eval, y_train_eval, sample_weight=sample_weights_eval)
            X_test_eval = test_data[features_eval]
            pred_eval = clf_temp.predict(X_test_eval)[0]
            actual_eval = test_data[target_eval].iloc[0]
            total_predictions += 1
            if pred_eval == actual_eval:
                total_correct += 1

    accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
    st.write(f"Historical Evaluation: {total_correct} correct out of {total_predictions} predictions. Accuracy: {accuracy:.1f}%")

    # Retrain a final model on the entire dataset for future predictions.
    final_data = df.dropna(subset=features_eval + [target_eval])
    X_final = final_data[features_eval]
    y_final = final_data[target_eval].astype(int)
    sample_weights_final = final_data.apply(compute_sample_weight, axis=1)
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_final, y_final, sample_weight=sample_weights_final)
    st.write("Final model retrained on the entire dataset for future predictions.")
