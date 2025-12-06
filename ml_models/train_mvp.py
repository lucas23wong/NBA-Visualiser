import pandas as pd 
import numpy as np
import os 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

INPUT_PLAYER_DATA_PATH = '../data/mvp_training_data.csv'
OUTPUT_DIR = 'model_artifacts'

def load_and_split_data(path: str):
    """
    Loads the preprocessed data, defines features + target, and splits it into training and testing sets.

    Args: 
        path(str): filepath to preprocessed data CSV
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and evaluation or none if no file is found.
    """
    if not os.path.exists(path):
        print(f"Input data file '{path}' not found.")
        return None, None, None, None
    
    print("--- Loading Preprocessed Data ---")
    df = pd.read_csv(path)

    feature_cols = [
        'PTS_per_36', 'REB_per_36', 'AST_per_36', 'NET_RATING', 'W_PCT',
        'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'PIE', 
        'OFF_RATING', 'DEF_RATING',                        
    ]

    X = df[[col for col in feature_cols if col in df.columns]]
    y = df['MVP_WINNER']

    if X.shape[0] == 0: 
        print("Feature matrix is empty. Cannot proceed with training.")
        return None, None, None, None
    if X.shape[1] == 0: 
        print("No feature columns found in the dataset. Cannot proceed with training.")
        return None, None, None, None
    
    print(f"Features selected: {list(X.columns)}")
    print(f"Total instances: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Trains a Logistic Regression model and evaluates its performance.

    Args: 
        X_train, X_test, y_train, y_test: Datasets for training and evaluation.

    Returns:
        tuple: (trained_model, scaler) or (None, None) if training fails.
    """
    print("\n--- Training Logistic Regression Model ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed.")

    model = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    print("Logisitcic Regression model training completed.")

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler

def save_model_artifacts(model, scaler, output_dir: str):
    """
    Saves the trained model and scaler using joblib.

    Args: 
        model: Trained machine learning model.
        scaler: Fitted scaler object.
        output_dir(str): Directory to save the artifacts.
    
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, 'mvp_model.joblib')
    scaler_path = os.path.join(output_dir, 'mvp_scaler.joblib')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n✅ Model saved to '{model_path}'")
    print(f"✅ Scaler saved to '{scaler_path}'")

def run_training_pipeline():
    """
    Executes full model training and saving process.
    """
    X_train, X_test, y_train, y_test = load_and_split_data(INPUT_PLAYER_DATA_PATH)
    
    if X_train is None:
        return

    model, scaler = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    save_model_artifacts(model, scaler, OUTPUT_DIR)


if __name__ == '__main__':
    run_training_pipeline()