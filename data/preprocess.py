import pandas as pd 
import numpy as np 
from typing import List, Dict, Any
import os 

MVP_DICT = {
    '2024-25': 'Shai Gilgeous-Alexander',
    '2023-24': 'Nikola Jokic',
    '2022-23': 'Joel Embiid',
    '2021-22': 'Nikola Jokic',
    '2020-21': 'Nikola Jokic',
    '2019-20': 'Giannis Antetokounmpo',
    '2018-19': 'Giannis Antetokounmpo',
    '2017-18': 'James Harden',
    '2016-17': 'Russell Westbrook',
    '2015-16': 'Stephen Curry',
    '2014-15': 'Stephen Curry',
    '2013-14': 'Kevin Durant',
    '2012-13': 'LeBron James',
    '2011-12': 'LeBron James',
    '2010-11': 'Derrick Rose',
    '2009-10': 'LeBron James',
    '2008-09': 'LeBron James',
    '2007-08': 'Kobe Bryant',
    '2006-07': 'Dirk Nowitzki',
    '2005-06': 'Steve Nash',
    '2004-05': 'Steve Nash',
    '2003-04': 'Kevin Garnett',
    '2002-03': 'Tim Duncan',
    '2001-02': 'Tim Duncan',
    '2000-01': 'Allen Iverson',
}

TEAM_ABBREVIATION_MAP = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

def load_and_clean_data(player_path: str, team_path: str) -> pd.DataFrame:
    if not os.path.exists(player_path) or not os.path.exists(team_path):
        print("ERROR: One or both data files do not exist.")
        return pd.DataFrame()
    
    print("Loading and merging raw data...")
    player_df = pd.read_csv(player_path)
    team_df = pd.read_csv(team_path)

    print("Adding TEAM_ABBREVIATION to team data...")
    team_df['TEAM_ABBREVIATION'] = team_df['TEAM_NAME'].map(TEAM_ABBREVIATION_MAP)
    
    # Check if any teams couldn't be mapped
    unmapped_teams = team_df[team_df['TEAM_ABBREVIATION'].isna()]['TEAM_NAME'].unique()
    if len(unmapped_teams) > 0:
        print(f"WARNING: Could not map abbreviations for teams: {unmapped_teams}")

    team_columns_to_keep = ['TEAM_ABBREVIATION', 'SEASON', 'W', 'L', 'W_PCT', 
                             'OFF_RATING', 'DEF_RATING', 'NET_RATING']
    
    missing_cols = [col for col in team_columns_to_keep if col not in team_df.columns]
    if missing_cols:
        print(f"ERROR: Team data is missing columns: {missing_cols}")
        return pd.DataFrame()
    
    team_df = team_df[team_columns_to_keep]
    team_df = team_df.rename(columns={'NET_RATING': 'TEAM_NET_RATING'})

    merged_df = pd.merge(player_df, team_df, 
                         left_on=['SEASON', 'TEAM_ABBREVIATION'], 
                         right_on=['SEASON', 'TEAM_ABBREVIATION'], 
                         how='left')
    
    team_stat_cols = ['W', 'L', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'TEAM_NET_RATING']
    for col in team_stat_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
    
    merged_df = merged_df.dropna(subset=['MIN', 'PTS', 'REB', 'AST'])
    print(f"Data merged successfully. Total records: {len(merged_df)}")
    return merged_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the target variable (MVP_WINNER) and new predictive features.
    
    Args:
        df: The merged player and team DataFrame.

    Returns:
        DataFrame with new engineered features and the target variable.
    """
    if df.empty:
        return df
        
    print("--- Engineering Features and Target Variable ---")

    # Convert MVP_WINNERS dict to a DataFrame for joining
    mvp_df = pd.DataFrame(MVP_DICT.items(), columns=['SEASON', 'MVP_WINNER_NAME'])
    
    # Join with the main dataset
    df = pd.merge(df, mvp_df, on='SEASON', how='left')
    
    # Create the binary target column 
    df['MVP_WINNER'] = np.where(df['PLAYER_NAME'] == df['MVP_WINNER_NAME'], 1, 0)
    
    # Drop the temporary MVP_WINNER_NAME column
    df = df.drop(columns=['MVP_WINNER_NAME'], errors='ignore')

    # A. Player Efficiency/Impact Score: A combination of key stats per 36 minutes
    df['PTS_per_36'] = (df['PTS'] / df['MIN']) * 36
    df['REB_per_36'] = (df['REB'] / df['MIN']) * 36
    df['AST_per_36'] = (df['AST'] / df['MIN']) * 36

    df_filtered = df[df['GP'] >= 50].copy()

    print(f"Feature engineering complete. Players filtered by GP>=50: {len(df_filtered)}")
    return df_filtered

def create_training_data():
    """
    Main function to execute the preprocessing pipeline and save the final dataset.
    """
    PLAYER_DATA_PATH = 'data/nba_player_data.csv'
    TEAM_DATA_PATH = 'data/nba_team_data.csv'
    OUTPUT_PATH = 'data/mvp_training_data.csv'
    
    print("\n==============================================")
    print("      Starting MVP Data Preprocessing")
    print("==============================================\n")
    
    merged_data = load_and_clean_data(PLAYER_DATA_PATH, TEAM_DATA_PATH)

    if merged_data.empty:
        print("Preprocessing failed due to missing input data.")
        return

    final_data = engineer_features(merged_data)
    
    if not final_data.empty:
        final_data.to_csv(OUTPUT_PATH, index=False)
        
        mvp_count = final_data['MVP_WINNER'].sum()
        print(f"\n✅ Final training data saved to '{OUTPUT_PATH}'.")
        print(f"   Total rows: {len(final_data)} | Total MVP Winners Labeled: {mvp_count}")
        print("\nFeatures in the final dataset (Head):")
        print(final_data[['PLAYER_NAME', 'SEASON', 'GP', 'MIN', 'PTS_per_36', 'TEAM_NET_RATING', 'W_PCT', 'MVP_WINNER']].head())
    else:
        print("Final DataFrame is empty. Check data loading and filtering steps.")

if __name__ == '__main__':
    create_training_data()
