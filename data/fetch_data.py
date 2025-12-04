import time 
import pandas as pd
import datetime as dt
from nba_api.stats.endpoints import (
    leaguedashplayerstats, # Player stats (Base and Advanced)
    leaguedashteamstats, # Team stats (Base and Advanced)
)
from nba_api.stats.library.parameters import SeasonType
from typing import List, Dict, Any

COMMON_PARAMS = {}

def get_current_season_str() -> str: 
    """
    Get the current NBA season in 'YYYY-YY' format.
    """
    today = dt.date.today()
    year = today.year
    month = today.month
    if month >= 10:  # NBA season starts in October
        season = f"{year}-{str(year + 1)[-2:]}"
    else:
        season = f"{year - 1}-{str(year)[-2:]}"
    return season


def fetch_player_data(season: str) -> pd.DataFrame:
    """
    Fetches comprehensive player data for a given season str

    Arg:
        season (str): Season in 'YYYY-YY' format.

    Returns:
        A pd.DataFrame containing combined player stats or an empty DataFrame on failure.
    """
    print(f"Fetching player data for season {season}...")

    try: 
        print("Fetching base player stats (Per Game PTS, AST, REB, etc.)...")
        base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",  # Use 'per_mode_detailed' as per the documentation
            season_type_all_star="Regular Season",  # Use 'season_type_all_star' for season type
            measure_type_detailed_defense="Base",  # Use 'measure_type_detailed_defense' for base stats
        ).get_data_frames()[0]

        required_columns = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
        if not all(col in base_stats.columns for col in required_columns):
            print(f"Missing required columns in base player stats for season {season}. Skipping...")
            return pd.DataFrame()

        base_stats = base_stats[required_columns]

        time.sleep(3)

        print("Fetching advanced player stats (Net rating, Usage %, etc.)...")
        advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",  # Use 'per_mode_detailed' as per the documentation
            season_type_all_star="Regular Season",  # Use 'season_type_all_star' for season type
            measure_type_detailed_defense="Advanced",  # Use 'measure_type_detailed_defense' for advanced stats
        ).get_data_frames()[0]

        required_columns_adv = ['PLAYER_ID', 'NET_RATING', 'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'PIE']
        if not all(col in advanced_stats.columns for col in required_columns_adv):
            print(f"Missing required columns in advanced player stats for season {season}. Skipping...")
            return pd.DataFrame()

        advanced_stats = advanced_stats[required_columns_adv]

        time.sleep(3)

        combined_data = pd.merge(base_stats, advanced_stats, on='PLAYER_ID', how='inner')

        print(f"Merged {len(combined_data)} player records with base and advanced stats.")
        return combined_data
    
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error occurred while fetching player data for season {season}: {e}")
        return pd.DataFrame()

def fetch_team_data(season: str) -> pd.DataFrame:
    """
    Fetches team efficiency and win/loss records for a given season string.

    Arg:
        season (str): Season in 'YYYY-YY' format.
    Returns:
        A pd.DataFrame containing team stats or an empty DataFrame on failure.
    """
    print(f"Fetching team data for season {season}...")

    try: 
        print("Fetching base team stats (Win/loss records)...")

        base_team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals",
            measure_type_detailed_defense="Base",
        ).get_data_frames()[0]

        required_columns = ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS']
        if not all(col in base_team_stats.columns for col in required_columns):
            print(f"Missing required columns in base team stats for season {season}. Skipping...")
            return pd.DataFrame()

        base_team_stats = base_team_stats[required_columns]

        time.sleep(3)

        print("Fetching advanced team stats (Net Rating, Off/Def ratings)...")

        efficiency_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]

        required_columns_eff = ['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']
        if not all(col in efficiency_stats.columns for col in required_columns_eff):
            print(f"Missing required columns in advanced team stats for season {season}. Skipping...")
            return pd.DataFrame()

        efficiency_stats = efficiency_stats[required_columns_eff]

        time.sleep(3)

        combined_data = pd.merge(base_team_stats, efficiency_stats, on='TEAM_ID', how='inner')
        combined_data['SEASON'] = season 

        print(f"Merged {len(combined_data)} team records with base and efficiency stats.")
        return combined_data
    
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error occurred while fetching team data for season {season}: {e}")
        return pd.DataFrame()

def run_data_pipeline():
    """
    Executes the data fetching for all defined seasons and saves the results to CSV files for ML training.
    """

    START_YEAR = 2023
    current_season_str = get_current_season_str()

    try:
        current_year_end = int(current_season_str.split('-')[0]) + 1
    except ValueError: 
        print("Error determining current season year. Defaulting to 2025 end.")
        current_year_end = 2000

    seasons_to_fetch = []
    for start_year in range(START_YEAR, current_year_end):
        end_year_short = str(start_year + 1)[-2:]
        season_str = f"{start_year}-{end_year_short}"
        seasons_to_fetch.append(season_str)
    
    print(f"\n--- STARTING FULL DATA PIPELINE ({len(seasons_to_fetch)} Seasons) ---")
    print(f"Fetching seasons: {seasons_to_fetch[0]} through {seasons_to_fetch[-1]}")

    all_player_data = []
    all_team_data = []
    
    for season in seasons_to_fetch:
        player_df = fetch_player_data(season)
        if not player_df.empty:
            all_player_data.append(player_df)
        
        team_df = fetch_team_data(season)
        if not team_df.empty:
            all_team_data.append(team_df)
        
        time.sleep(5)
    
    if all_player_data:
        final_player_df = pd.concat(all_player_data, ignore_index=True)
        final_player_df.to_csv('data/nba_player_data.csv', index=False)
        print(f"\n✅ All Player data saved to 'data/nba_player_data.csv'. Total rows: {len(final_player_df)}")
        print(final_player_df.head())

    if all_team_data:
        final_team_df = pd.concat(all_team_data, ignore_index=True)
        final_team_df.to_csv('data/nba_team_data.csv', index=False)
        print(f"✅ All Team data saved to 'data/nba_team_data.csv'. Total rows: {len(final_team_df)}")
        print(final_team_df.head())

if __name__ == "__main__":
    run_data_pipeline()

