import pandas as pd
import joblib
import os
import numpy as np
from typing import Dict, List, Tuple

MODEL_DIR = 'model_artifacts'

FEATURE_COLS = [
    'PTS_per_36', 'REB_per_36', 'AST_per_36', 'NET_RATING', 'W_PCT',
    'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'PIE',
    'OFF_RATING', 'DEF_RATING'
]

ALL_NBA_MAPPING = {
    1: '1st Team',
    2: '2nd Team',
    3: '3rd Team'
}


def load_model_and_scaler(target_name: str) -> Tuple[any, any]:
    """Loads a specific trained model and its associated scaler."""
    model_path = os.path.join(MODEL_DIR, f'{target_name}_model.joblib')
    scaler_path = os.path.join(MODEL_DIR, f'{target_name}_scaler.joblib')
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        print(f"Error: Model or Scaler not found for {target_name}. Paths checked:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print("Please ensure 'train_other_awards.py' was run successfully.")
        return None, None


def generate_predictions(df_current_season: pd.DataFrame) -> pd.DataFrame:
    """
    Generates MVP, DPOY, and All-NBA predictions (probabilities) for the current season data.
    
    Args:
        df_current_season: DataFrame of current season stats for all players.
        
    Returns:
        DataFrame with prediction probabilities appended.
    """
    if df_current_season.empty:
        return pd.DataFrame()

    X_predict = df_current_season[FEATURE_COLS].copy()
    
    results_df = df_current_season[['PLAYER_NAME', 'TEAM_ABBREVIATION']].copy()

    # --- 1. MVP Prediction (Binary Classification / Ranking) ---
    mvp_model, mvp_scaler = load_model_and_scaler('mvp')
    if mvp_model:
        X_scaled = mvp_scaler.transform(X_predict)
        
        mvp_probs = mvp_model.predict_proba(X_scaled)[:, 1]
        results_df['MVP_PROB'] = mvp_probs * 100 
        
        print("✅ MVP probabilities generated.")
    else:
        results_df['MVP_PROB'] = np.nan
        

    # --- 2. DPOY Prediction (Binary Classification / Ranking) ---
    dpoy_model, dpoy_scaler = load_model_and_scaler('DPOY_WINNER')
    if dpoy_model:
        X_scaled = dpoy_scaler.transform(X_predict)
        
        dpoy_probs = dpoy_model.predict_proba(X_scaled)[:, 1]
        results_df['DPOY_PROB'] = dpoy_probs * 100 
        
        print("✅ DPOY probabilities generated.")
    else:
        results_df['DPOY_PROB'] = np.nan
        

    # --- 3. All-NBA Prediction (Multi-Class Classification) ---
    all_nba_model, all_nba_scaler = load_model_and_scaler('ALL_NBA_TEAM')
    if all_nba_model:
        X_scaled = all_nba_scaler.transform(X_predict)
        
        predicted_class = all_nba_model.predict(X_scaled)
        
        all_nba_probs = all_nba_model.predict_proba(X_scaled)
        
        results_df['ALL_NBA_PREDICTION'] = [
            ALL_NBA_MAPPING.get(pc, 'No Team') for pc in predicted_class
        ]
        
        results_df['ALL_NBA_CONFIDENCE'] = np.max(all_nba_probs, axis=1) * 100
        
        print("✅ All-NBA predictions generated.")
    else:
        results_df['ALL_NBA_PREDICTION'] = 'Model Error'
        results_df['ALL_NBA_CONFIDENCE'] = np.nan

    return results_df.sort_values(by=['MVP_PROB', 'DPOY_PROB'], ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    try:
        # Load the full training data and filter for the latest season used in the training data
        df_full = pd.read_csv('../data/mvp_training_data.csv')
        current_season_str = df_full['SEASON'].max()
        df_current_season_data = df_full[df_full['SEASON'] == current_season_str].copy()
        print(f"Loaded {len(df_current_season_data)} players from season {current_season_str} for prediction demo.")
        
        # Run predictions
        predictions = generate_predictions(df_current_season_data)
        
        if not predictions.empty:
            print("\n--- Top MVP, DPOY, and All-NBA Predictions ---")
            # Display relevant prediction columns for the demo
            display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MVP_PROB', 'DPOY_PROB', 'ALL_NBA_PREDICTION', 'ALL_NBA_CONFIDENCE']
            print(predictions[display_cols].head(15).to_markdown(index=False))
            
    except FileNotFoundError:
        print("Cannot run demo. Ensure '../data/mvp_training_data.csv' exists and is correctly structured.")
    except Exception as e:
        print(f"An error occurred during prediction demo: {e}")