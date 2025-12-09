### NBA PlayerAI: MVP, DPOY, ALL-NBA Prediction and Player Statistics Visualization Tool

## Overview

NBA PlayerAI is a full-stack web application built with Flask and Python, with the purpose of analyzing and visualizing NBA player performance and predicting the 2025-26 NBA season awards using machine learning models. The project encompasses a complete data pipeline, from scraping live statistics using API to generating and displaying predictions and historical player trends.

----------------------------------------------------------------

## Key Features

Real-Time Leaderboards: Displays current top-performing players across major statistical categories such as Points, Rebounds, Assists per game.

Award Predictions: Uses trained classification models (e.g., Logistic Regression or Random Forest) to predict MVP, Defensive Player of the Year (DPOY), and All-NBA Teams with corresponding probabilities.

Player History Visualization: Uses D3.js to render interactive line charts showing a player's seasonal performance trend for any chosen statistic.

Responsive Web Interface: Built with Flask and Tailwind CSS for a clean, modern, and adaptive user experience.

----------------------------------------------------------------

## Project Demonstration Video Link: https://youtu.be/O8H2sELXEE4

----------------------------------------------------------------

## How to Run Locally (Installation Guide)

Prerequisites

You will need the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- git (for cloning the repository)

1. Installation

Clone the repository:

git clone <repository-url>
cd visual_nba

Create and Activate a Virtual Environment:

# Create the environment
python3 -m venv .venv

# Activate the environment (macOS/Linux)
source .venv/bin/activate

# Activate the environment (Windows)
.venv\Scripts\activate


Install Dependencies:
You must install all required Python packages. Your project uses libraries like Flask, pandas, scikit-learn, nba_api, and joblib. Ensure you have a requirements.txt file listing these.

pip install -r requirements.txt


2. Data Pipeline Execution (Crucial Step)

The application cannot run without the pre-generated data files in the /data and /app/static/data directories. You must execute the following scripts in order:

Fetch Raw Data (fetch_data.py):
This script scrapes comprehensive player and team statistics from the NBA API for historical and current seasons. The historical data dates back to the 2000-01 NBA season.

python3 fetch_data.py

-> Output files: data/data/nba_player_data.csv and data/data/nba_team_data.csv.

Preprocess Training Data (preprocess.py):
This script cleans, merges, applies feature engineering (e.g., PER-36 minutes metrics, relative team performance), and applies the historical MVP winner labels.

python3 preprocess.py

-> Output file: data/mvp_training_data.csv.

Generate Predictions (run_predictions.py):
This script loads the pre-trained ML models from model_artifacts/ (assuming they are trained and saved) and uses the latest data to output predictions for the current season.

python3 run_predictions.py

Output file: app/static/data/predictions.json 

3. Start the Web Server

After the data pipeline is complete, you can start the Flask server:

# Ensure your virtual environment is active
python3 app.py

The application will be accessible at: http://127.0.0.1:5000/

----------------------------------------------------------------

## Application Usage

The navigation bar allows access to the three main pages:

1. Home Page (/)

Purpose: Displays basic leaderboards (e.g., Top 5 in PTS, AST, REB).

Functionality: Data is loaded from the most recent season data set (nba_player_data.csv) and presented in structured lists ranking the TOP 5 players in each statistical category. 

2. Predictions Page (/predictions)

Purpose: View machine learning predictions for major awards.

Functionality: Reads predictions.json and displays the top 5 candidates for MVP, DPOY, and ALL-NBA awards. Each ranked player's associated probability is also displayed to the user. 

3. Player Visualization Page (/player/<player_id>)

Purpose: Visualize a player's career trend for a specific statistic.

Steps:
- Enter the exact Player Name (e.g., "Nikola Jokic") in the input box.
- Select the desired Statistic (e.g., PTS, REB, AST, MIN) from the dropdown.
- Click the Load History button.

Technical Note: This triggers an internal API call (/api/history/<player_name>/<stat_name>) defined in app.py, which fetches data from the historical nba_player_data.csv and renders a D3.js line chart on the page.