import pandas as pd
import os
import json
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the prediction logic
from predict_awards import generate_predictions

# --- CONFIGURATION ---
PLAYER_DATA_PATH = '../data/data/nba_player_data.csv'
TEAM_DATA_PATH = '../data/data/nba_team_data.csv'
OUTPUT_JSON_PATH = '../app/static/data/predictions.json'

TEAM_ABBREVIATION_MAP = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

def prepare_live_data():
    """
    Loads raw player/team data, filters for the current season, merges them,
    and performs feature engineering (Per 36 stats) required by the model.
    """
    print("--- Loading and Preprocessing Live Data ---")
    
    # Check for file existence first
    if not os.path.exists(PLAYER_DATA_PATH) or not os.path.exists(TEAM_DATA_PATH):
        print(f"Error: Raw data files not found at {PLAYER_DATA_PATH} or {TEAM_DATA_PATH}")
        return pd.DataFrame()

    # 1. Load Raw Data
    player_df = pd.read_csv(PLAYER_DATA_PATH)
    team_df = pd.read_csv(TEAM_DATA_PATH)
    
    # 2. Identify Current Season (The latest one in the data)
    current_season = player_df['SEASON'].max()
    print(f"Targeting Season: {current_season}")
    
    # 3. Filter for Current Season ONLY
    player_current = player_df[player_df['SEASON'] == current_season].copy()
    team_current = team_df[team_df['SEASON'] == current_season].copy()
    
    if player_current.empty:
        print("No player data found for the target season.")
        return pd.DataFrame()

    player_current['TEAM_NAME'] = player_current['TEAM_ABBREVIATION'].map(TEAM_ABBREVIATION_MAP)
    
    if team_current.empty:
        print(f"Warning: No team data found for season {current_season}. Aborting merge.")
        return pd.DataFrame() 
    
    # 4. Merge Player and Team Data
    merged_df = pd.merge(
        player_current, 
        team_current, 
        on=['SEASON', 'TEAM_NAME'],
        how='left'
    )
    
    merged_df = merged_df.rename(columns={
        'GP_x': 'GP',
        'MIN_x': 'MPG', 
        'PTS_x': 'PTS',
        'REB_x': 'REB',
        'AST_x': 'AST'
    })

    merged_df['MIN'] = merged_df['GP'] * merged_df['MPG']
    
    merged_df = merged_df.drop(columns=['GP_y', 'W_y', 'L_y', 'MIN_y', 'PTS_y'], errors='ignore')
    
    
    
    merged_df = merged_df[
        (merged_df['GP'] >= 5) & (merged_df['MIN'] >= 100)
    ].copy()
    
    merged_df['MIN_SAFE'] = merged_df['MIN'].replace(0, np.nan)
    
    merged_df['PTS_per_36'] = (merged_df['PTS'] / merged_df['MIN_SAFE']) * 36
    merged_df['REB_per_36'] = (merged_df['REB'] / merged_df['MIN_SAFE']) * 36
    merged_df['AST_per_36'] = (merged_df['AST'] / merged_df['MIN_SAFE']) * 36
    
    merged_df = merged_df.drop(columns=['MIN_SAFE'])
    
    team_cols = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'W_PCT']
    for col in team_cols:
        # Ensure column exists, as it might be missing if no team data was merged
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
        else:
            merged_df[col] = 0 

    
    print(f"Live data ready. Candidates after filtering (GP>=5, MIN>=100): {len(merged_df)}")
    return merged_df

def run_production_predictions():
    print("--- Starting Production Prediction Run ---")
    
    # 1. Get the prepared live data
    df_current = prepare_live_data()
    
    if df_current.empty:
        print("Aborting: No live data available.")
        return

    # 2. Generate Predictions
    results_df = generate_predictions(df_current)
    
    if results_df.empty:
        print("No predictions generated.")
        return

    # 3. Format for Web App (JSON)
    display_cols = [
        'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'PTS', 'REB', 'AST',
        'MVP_PROB', 'DPOY_PROB', 'ALL_NBA_PREDICTION', 'ALL_NBA_CONFIDENCE'
    ]
    
    # Ensure columns exist before selecting
    valid_cols = [c for c in display_cols if c in results_df.columns]
    final_data = results_df[valid_cols]

    # 4. Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    final_data.to_json(OUTPUT_JSON_PATH, orient='records', indent=4)
    print(f"\n✅ Predictions saved to: {OUTPUT_JSON_PATH}")
    
    if not final_data.empty:
        top_player = final_data.iloc[0]
        # Use .get() for safety in case prediction failed or column is missing
        mvp_prob = top_player.get('MVP_PROB', 0)
        print(f"   Top candidate: {top_player['PLAYER_NAME']} (MVP Prob: {mvp_prob:.1f}%)")

if __name__ == "__main__":
    run_production_predictions()