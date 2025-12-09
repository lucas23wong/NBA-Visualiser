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

DPOY_DICT = {
    '2024-25': 'Evan Mobley',
    '2023-24': 'Rudy Gobert', 
    '2022-23': 'Jaren Jackson Jr.',
    '2021-22': 'Marcus Smart',
    '2020-21': 'Rudy Gobert',
    '2019-20': 'Giannis Antetokounmpo',
    '2018-19': 'Rudy Gobert',
    '2017-18': 'Rudy Gobert',
    '2016-17': 'Draymond Green',
    '2015-16': 'Kawhi Leonard',
    '2014-15': 'Kawhi Leonard',
    '2013-14': 'Joakim Noah',
    '2012-13': 'Marc Gasol',
    '2011-12': 'Tyson Chandler',
    '2010-11': 'Dwight Howard',
    '2009-10': 'Dwight Howard',
    '2008-09': 'Dwight Howard',
    '2007-08': 'Kevin Garnett',
    '2006-07': 'Marcus Camby',
    '2005-06': 'Ben Wallace',
    '2004-05': 'Ben Wallace',
    '2003-04': 'Meta World Peace',
    '2002-03': 'Ben Wallace',
    '2001-02': 'Ben Wallace',
    '2000-01': 'Dikembe Mutombo',
}

ALL_NBA_DICT = {
    '2024-25': {
        'Nikola Jokic': 1, 'Shai Gilgeous-Alexander': 1, 'Jayson Tatum': 1, 'Donovan Mitchell': 1, 'Giannis Antetokounmpo': 1,
        'Jalen Brunson': 2, 'Stephen Curry': 2, 'Anthony Edwards': 2, 'LeBron James': 2, 'Evan Mobley': 2,
        'James Harden': 3, 'Cade Cunningham': 3, 'Tyrese Haliburton': 3, 'Jalen Williams': 3, 'Karl-Anthony Towns': 3,
    },
    '2023-24': {
        'Nikola Jokic': 1, 'Shai Gilgeous-Alexander': 1, 'Luka Doncic': 1, 'Giannis Antetokounmpo': 1, 'Jayson Tatum': 1,
        'Jalen Brunson': 2, 'Anthony Edwards': 2, 'Kevin Durant': 2, 'Kawhi Leonard': 2, 'Anthony Davis': 2,
        'Tyrese Haliburton': 3, 'Devin Booker': 3, 'Stephen Curry': 3, 'LeBron James': 3, 'Domantas Sabonis': 3,
    },
    '2022-23': {
        'Nikola Jokic': 1, 'Shai Gilgeous-Alexander': 1, 'Luka Doncic': 1, 'Giannis Antetokounmpo': 1, 'Joel Embiid': 1,
        'Jayson Tatum': 2, 'Donovan Mitchell': 2, 'Stephen Curry': 2, 'Jimmy Butler': 2, 'Jaylen Brown': 2,
        'LeBron James': 3, 'Damian Lillard': 3, 'De\'Aaron Fox': 3, 'Julius Randle': 3, 'Nikola Vucevic': 3,
    },
    '2021-22': {
        'Nikola Jokic': 1, 'Luka Doncic': 1, 'Devin Booker': 1, 'Jayson Tatum': 1, 'Giannis Antetokounmpo': 1,
        'Ja Morant': 2, 'Stephen Curry': 2, 'Kevin Durant': 2, 'Joel Embiid': 2, 'DeMar DeRozan': 2,
        'Karl-Anthony Towns': 3, 'LeBron James': 3, 'Chris Paul': 3, 'Trae Young': 3, 'Pascal Siakam': 3,
    },
    '2020-21': {
        'Nikola Jokic': 1, 'Stephen Curry': 1, 'Luka Doncic': 1, 'Kawhi Leonard': 1, 'Giannis Antetokounmpo': 1,
        'Damian Lillard': 2, 'Chris Paul': 2, 'Julius Randle': 2, 'Joel Embiid': 2, 'LeBron James': 2,
        'Kyrie Irving': 3, 'Bradley Beal': 3, 'Rudy Gobert': 3, 'Jimmy Butler': 3, 'Paul George': 3,
    },
    '2019-20': {
        'Giannis Antetokounmpo': 1, 'LeBron James': 1, 'James Harden': 1, 'Luka Doncic': 1, 'Anthony Davis': 1,
        'Kawhi Leonard': 2, 'Damian Lillard': 2, 'Chris Paul': 2, 'Pascal Siakam': 2, 'Nikola Jokic': 2,
        'Jayson Tatum': 3, 'Jimmy Butler': 3, 'Rudy Gobert': 3, 'Ben Simmons': 3, 'Russell Westbrook': 3,
    },
    '2018-19': {
        'Giannis Antetokounmpo': 1, 'LeBron James': 1, 'James Harden': 1, 'Paul George': 1, 'Nikola Jokic': 1,
        'Kawhi Leonard': 2, 'Kevin Durant': 2, 'Stephen Curry': 2, 'Damian Lillard': 2, 'Joel Embiid': 2,
        'Russell Westbrook': 3, 'Kemba Walker': 3, 'Rudy Gobert': 3, 'Blake Griffin': 3, 'LeBron James': 3, # LeBron dropped to 3rd team
    },
    '2017-18': {
        'LeBron James': 1, 'James Harden': 1, 'Anthony Davis': 1, 'Damian Lillard': 1, 'Kevin Durant': 1,
        'Giannis Antetokounmpo': 2, 'Russell Westbrook': 2, 'Joel Embiid': 2, 'LaMarcus Aldridge': 2, 'DeMar DeRozan': 2,
        'Stephen Curry': 3, 'Victor Oladipo': 3, 'Karl-Anthony Towns': 3, 'Paul George': 3, 'Jimmy Butler': 3,
    },
    '2016-17': {
        'LeBron James': 1, 'James Harden': 1, 'Russell Westbrook': 1, 'Kawhi Leonard': 1, 'Anthony Davis': 1,
        'Kevin Durant': 2, 'Stephen Curry': 2, 'Giannis Antetokounmpo': 2, 'Rudy Gobert': 2, 'Isaiah Thomas': 2,
        'DeMar DeRozan': 3, 'John Wall': 3, 'DeAndre Jordan': 3, 'Jimmy Butler': 3, 'Draymond Green': 3,
    },
    '2015-16': {
        'Stephen Curry': 1, 'LeBron James': 1, 'Kawhi Leonard': 1, 'Russell Westbrook': 1, 'DeAndre Jordan': 1,
        'Kevin Durant': 2, 'Damian Lillard': 2, 'Draymond Green': 2, 'DeMarcus Cousins': 2, 'Chris Paul': 2,
        'Paul George': 3, 'Andre Drummond': 3, 'Klay Thompson': 3, 'LaMarcus Aldridge': 3, 'Al Horford': 3,
    },
    '2014-15': {
        'Stephen Curry': 1, 'LeBron James': 1, 'James Harden': 1, 'Anthony Davis': 1, 'Marc Gasol': 1,
        'Russell Westbrook': 2, 'Chris Paul': 2, 'LaMarcus Aldridge': 2, 'Blake Griffin': 2, 'Pau Gasol': 2,
        'Kyrie Irving': 3, 'Klay Thompson': 3, 'DeMarcus Cousins': 3, 'Tim Duncan': 3, 'Kawhi Leonard': 3,
    },
    '2013-14': {
        'LeBron James': 1, 'Kevin Durant': 1, 'Chris Paul': 1, 'James Harden': 1, 'Joakim Noah': 1,
        'Stephen Curry': 2, 'Tony Parker': 2, 'Blake Griffin': 2, 'Kevin Love': 2, 'Dwight Howard': 2,
        'Damian Lillard': 3, 'Goran Dragic': 3, 'LaMarcus Aldridge': 3, 'Paul George': 3, 'Al Jefferson': 3,
    },
    '2012-13': {
        'LeBron James': 1, 'Kevin Durant': 1, 'Chris Paul': 1, 'Kobe Bryant': 1, 'Tim Duncan': 1,
        'Carmelo Anthony': 2, 'Tony Parker': 2, 'Russell Westbrook': 2, 'Blake Griffin': 2, 'Marc Gasol': 2,
        'James Harden': 3, 'Dwyane Wade': 3, 'David Lee': 3, 'Paul George': 3, 'Dwight Howard': 3,
    },
    '2011-12': {
        'LeBron James': 1, 'Kevin Durant': 1, 'Chris Paul': 1, 'Kobe Bryant': 1, 'Dwight Howard': 1,
        'Tony Parker': 2, 'Russell Westbrook': 2, 'Kevin Love': 2, 'Blake Griffin': 2, 'Andrew Bynum': 2,
        'Dwyane Wade': 3, 'Rajon Rondo': 3, 'Carmelo Anthony': 3, 'Dirk Nowitzki': 3, 'Tyson Chandler': 3,
    },
    '2010-11': {
        'LeBron James': 1, 'Dirk Nowitzki': 1, 'Derrick Rose': 1, 'Kobe Bryant': 1, 'Dwight Howard': 1,
        'Dwyane Wade': 2, 'Russell Westbrook': 2, 'Pau Gasol': 2, 'Amar\'e Stoudemire': 2, 'Carmelo Anthony': 2,
        'Tony Parker': 3, 'Rajon Rondo': 3, 'Al Horford': 3, 'Manu Ginobili': 3, 'Kevin Love': 3,
    },
    '2009-10': {
        'LeBron James': 1, 'Carmelo Anthony': 1, 'Kobe Bryant': 1, 'Dwight Howard': 1, 'Dwyane Wade': 1,
        'Dirk Nowitzki': 2, 'Kevin Durant': 2, 'Steve Nash': 2, 'Amar\'e Stoudemire': 2, 'Deron Williams': 2,
        'Paul Pierce': 3, 'Joe Johnson': 3, 'Tim Duncan': 3, 'Gerald Wallace': 3, 'Pau Gasol': 3,
    },
    '2008-09': {
        'LeBron James': 1, 'Kobe Bryant': 1, 'Dwight Howard': 1, 'Dwyane Wade': 1, 'Dirk Nowitzki': 1,
        'Chris Paul': 2, 'Tony Parker': 2, 'Paul Pierce': 2, 'Brandon Roy': 2, 'Tim Duncan': 2,
        'Chauncey Billups': 3, 'Danny Granger': 3, 'Pau Gasol': 3, 'Shaquille O\'Neal': 3, 'Joe Johnson': 3,
    },
    '2007-08': {
        'LeBron James': 1, 'Kobe Bryant': 1, 'Kevin Garnett': 1, 'Dwight Howard': 1, 'Chris Paul': 1,
        'Tim Duncan': 2, 'Dirk Nowitzki': 2, 'Tracy McGrady': 2, 'Amar\'e Stoudemire': 2, 'Joe Johnson': 2,
        'Paul Pierce': 3, 'Chauncey Billups': 3, 'Yao Ming': 3, 'Gilbert Arenas': 3, 'Carlos Boozer': 3,
    },
    '2006-07': {
        'Dirk Nowitzki': 1, 'LeBron James': 1, 'Kobe Bryant': 1, 'Tim Duncan': 1, 'Steve Nash': 1,
        'Tracy McGrady': 2, 'Dwyane Wade': 2, 'Chauncey Billups': 2, 'Kevin Garnett': 2, 'Chris Bosh': 2,
        'Gilbert Arenas': 3, 'Joe Johnson': 3, 'Dwight Howard': 3, 'Yao Ming': 3, 'Ben Wallace': 3,
    },
    '2005-06': {
        'LeBron James': 1, 'Dirk Nowitzki': 1, 'Kobe Bryant': 1, 'Steve Nash': 1, 'Shaquille O\'Neal': 1,
        'Dwyane Wade': 2, 'Chauncey Billups': 2, 'Elton Brand': 2, 'Tim Duncan': 2, 'Ben Wallace': 2,
        'Gilbert Arenas': 3, 'Carmelo Anthony': 3, 'Shawn Marion': 3, 'Yao Ming': 3, 'Allen Iverson': 3,
    },
    '2004-05': {
        'LeBron James': 1, 'Dirk Nowitzki': 1, 'Shaquille O\'Neal': 1, 'Allen Iverson': 1, 'Steve Nash': 1,
        'Dwyane Wade': 2, 'Dwyane Wade': 2, 'Ray Allen': 2, 'Tim Duncan': 2, 'Kevin Garnett': 2,
        'Tracy McGrady': 3, 'Kobe Bryant': 3, 'Ben Wallace': 3, 'Gilbert Arenas': 3, 'Jermaine O\'Neal': 3,
    },
    '2003-04': {
        'Kevin Garnett': 1, 'Tim Duncan': 1, 'Shaquille O\'Neal': 1, 'Kobe Bryant': 1, 'Jason Kidd': 1,
        'Dirk Nowitzki': 2, 'Jermaine O\'Neal': 2, 'Paul Pierce': 2, 'Baron Davis': 2, 'Michael Redd': 2,
        'Tracy McGrady': 3, 'Ron Artest': 3, 'Ben Wallace': 3, 'Chauncey Billups': 3, 'Andrei Kirilenko': 3,
    },
    '2002-03': {
        'Kevin Garnett': 1, 'Tim Duncan': 1, 'Shaquille O\'Neal': 1, 'Kobe Bryant': 1, 'Tracy McGrady': 1,
        'Dirk Nowitzki': 2, 'Jermaine O\'Neal': 2, 'Paul Pierce': 2, 'Allen Iverson': 2, 'Jason Kidd': 2,
        'Ben Wallace': 3, 'Stephon Marbury': 3, 'Gary Payton': 3, 'Jamal Mashburn': 3, 'Baron Davis': 3,
    },
    '2001-02': {
        'Tim Duncan': 1, 'Kevin Garnett': 1, 'Shaquille O\'Neal': 1, 'Kobe Bryant': 1, 'Jason Kidd': 1,
        'Tracy McGrady': 2, 'Dirk Nowitzki': 2, 'Paul Pierce': 2, 'Allen Iverson': 2, 'Chris Webber': 2,
        'Peja Stojakovic': 3, 'Antoine Walker': 3, 'Ben Wallace': 3, 'Baron Davis': 3, 'Gary Payton': 3,
    },
    '2000-01': {
        'Allen Iverson': 1, 'Shaquille O\'Neal': 1, 'Tim Duncan': 1, 'Vince Carter': 1, 'Tracy McGrady': 1,
        'Chris Webber': 2, 'Dirk Nowitzki': 2, 'Ray Allen': 2, 'Gary Payton': 2, 'Karl Malone': 2,
        'Kevin Garnett': 3, 'Jason Kidd': 3, 'Antonio McDyess': 3, 'Ben Wallace': 3, 'Michael Finley': 3,
    },
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

def season_to_standard_format(season_val):
    """
    Converts season format (e.g., 2000 or 2000.0) to 'YYYY-YY' (e.g., '2000-01').
    """
    try:
        year = int(float(season_val))
        next_year = str(year + 1)[-2:]
        return f"{year}-{next_year}"
    except Exception:
        return str(season_val)


def load_and_clean_data(player_path: str, team_path: str) -> pd.DataFrame:
    if not os.path.exists(player_path) or not os.path.exists(team_path):
        print("ERROR: One or both data files do not exist.")
        return pd.DataFrame()
    
    print("Loading and merging raw data...")
    player_df = pd.read_csv(player_path)
    team_df = pd.read_csv(team_path)

    print("Standardizing SEASON format for consistent merging...")
    player_df['SEASON'] = player_df['SEASON'].astype(str)
    team_df['SEASON'] = team_df['SEASON'].apply(season_to_standard_format)

    print("Adding TEAM_ABBREVIATION to team data...")
    team_df['TEAM_ABBREVIATION'] = team_df['TEAM_NAME'].map(TEAM_ABBREVIATION_MAP)
    
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
    df = pd.merge(df, mvp_df, on='SEASON', how='left')
    df['MVP_WINNER'] = np.where(df['PLAYER_NAME'] == df['MVP_WINNER_NAME'], 1, 0)
    df = df.drop(columns=['MVP_WINNER_NAME'], errors='ignore')

    dpoy_df = pd.DataFrame(DPOY_DICT.items(), columns=['SEASON', 'DPOY_WINNER_NAME'])
    df = pd.merge(df, dpoy_df, on='SEASON', how='left')
    df['DPOY_WINNER'] = np.where(df['PLAYER_NAME'] == df['DPOY_WINNER_NAME'], 1, 0)
    df = df.drop(columns=['DPOY_WINNER_NAME'], errors='ignore')

    # --- 3. All-NBA Target ---
    df['ALL_NBA_TEAM'] = 0 # Default: Not on any team (Class 0)
    
    for season, winners in ALL_NBA_DICT.items():
        if season in df['SEASON'].values:
            for player, team_number in winners.items():
                # Identify the specific row by Season and Player
                condition = (df['SEASON'] == season) & (df['PLAYER_NAME'] == player)
                
                # Check if the player exists in the DataFrame for that season
                if condition.any():
                    # Set the ALL_NBA_TEAM value (1, 2, or 3) for that player
                    df.loc[condition, 'ALL_NBA_TEAM'] = team_number

    df['MIN_SAFE'] = df['MIN'].replace(0, np.nan)

    # A. Player Efficiency/Impact Score: A combination of key stats per 36 minutes
    df['PTS_per_36'] = (df['PTS'] / df['MIN']) * 36
    df['REB_per_36'] = (df['REB'] / df['MIN']) * 36
    df['AST_per_36'] = (df['AST'] / df['MIN']) * 36

    df = df.drop(columns=['MIN_SAFE'], errors='ignore')
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
