import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import joblib
import os
import numpy as np
from typing import List, Dict, Any

INPUT_DATA_PATH = '../data/mvp_training_data.csv'
OUTPUT_DIR = 'model_artifacts'

FEATURE_COLS = [
    'PTS_per_36', 'REB_per_36', 'AST_per_36', 'NET_RATING', 'W_PCT',
    'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'PIE',
    'OFF_RATING', 'DEF_RATING'
]

TARGET_COLS = ['DPOY_WINNER', 'ALL_NBA_TEAM'] 

def load_data():
    """
    Loads the preprocessed MVP training data and ensures chronological sort.
    """
    print("--- 1. Loading Preprocessed Data ---")
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"FATAL ERROR: Input training data file '{INPUT_DATA_PATH}' not found.")
        print("Please ensure 'data/preprocess.py' has been run successfully.")
        return None
        
    df = pd.read_csv(INPUT_DATA_PATH)
    
    missing_targets = [col for col in TARGET_COLS if col not in df.columns]
    if missing_targets:
        print(f"CRITICAL: Missing target columns in data: {missing_targets}")
        print("You must update 'data/preprocess.py' to label these awards before training.")
        return None
        
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS + ['SEASON'])
    
    df = df.sort_values('SEASON').reset_index(drop=True)
    

    df_train = df[df['SEASON'] != '2025-26'].copy()

    print(f"Training instances loaded: {len(df_train)}")
    print(f"Features selected: {FEATURE_COLS}")
    return df_train


def train_single_award_model(X: pd.DataFrame, y: pd.Series, target_name: str):
    """
    Trains, tunes, and evaluates a single Random Forest model using TimeSeriesSplit.
    
    Args:
        X: Feature DataFrame.
        y: Single target Series.
        target_name: The name of the target variable.
    
    Returns:
        tuple: (Trained model, Fitted scaler)
    """
    print(f"\n--- 2. Starting Model Training for: {target_name} ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    
    scoring_metric = 'roc_auc' if y.nunique() == 2 else 'f1_weighted'
    
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring=scoring_metric, 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters for {target_name}: {grid_search.best_params_}")
    print(f"Best score ({scoring_metric}): {grid_search.best_score_:.4f}")

    final_test_index = list(tscv.split(X_scaled))[4][1]
    X_test_scaled = X_scaled[final_test_index]
    y_test = y.iloc[final_test_index]
    
    y_pred = best_model.predict(X_test_scaled)
    
    print(f"\n--- Evaluation for Target: {target_name} ---")
    if y.nunique() == 2:
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # Note: ROC AUC is better than accuracy for imbalanced data
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    else:
        print(f"F1 Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return best_model, scaler


def save_model_artifacts(model, scaler, target_name: str, output_dir: str):
    """
    Saves the trained model and scaler with target-specific names.

    Args: 
        model: Trained machine learning model.
        scaler: Fitted scaler object.
        target_name (str): Name of the target variable for naming files.
        output_dir(str): Directory to save the artifacts.
    
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Use target name in the file path
    model_path = os.path.join(output_dir, f'{target_name}_model.joblib')
    scaler_path = os.path.join(output_dir, f'{target_name}_scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✅ Model and Scaler for '{target_name}' saved to '{output_dir}/'")


def run_training_pipeline():
    """
    Main function to run the entire other awards training process.
    """
    print("\n==============================================")
    print("   Starting Other Player Awards Training")
    print("==============================================")
    
    df_train = load_data()
    
    if df_train is None:
        return

    X = df_train[FEATURE_COLS]
    
    # -------------------------------------
    # 2. Train DPOY Model
    # -------------------------------------
    y_dpoy = df_train['DPOY_WINNER']
    dpoy_model, dpoy_scaler = train_single_award_model(X, y_dpoy, 'DPOY_WINNER')
    save_model_artifacts(dpoy_model, dpoy_scaler, 'DPOY_WINNER', OUTPUT_DIR)
    
    # -------------------------------------
    # 3. Train All-NBA Model
    # -------------------------------------
    y_all_nba = df_train['ALL_NBA_TEAM']
    
    X_all_nba = X[y_all_nba != 0].copy()
    y_all_nba = y_all_nba[y_all_nba != 0].copy()

    all_nba_model, all_nba_scaler = train_single_award_model(X_all_nba, y_all_nba, 'ALL_NBA_TEAM')
    save_model_artifacts(all_nba_model, all_nba_scaler, 'ALL_NBA_TEAM', OUTPUT_DIR)


if __name__ == '__main__':
    run_training_pipeline()