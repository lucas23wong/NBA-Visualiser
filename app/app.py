from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
from datetime import datetime
import os
import csv
import json  

custom_template_path = os.path.join(os.path.dirname(__file__), 'pantheon')
app = Flask(__name__, template_folder=custom_template_path)
app.secret_key = os.environ.get("SECRET_KEY", "devsecretkey")


app.started = False
@app.before_request
def clear_session_on_start():
    if not app.started:
        session.clear()
        app.started = True
        

    
@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.context_processor
def inject_now():
    """Inject the current year into all templates."""
    return {'now': lambda: datetime.now().year}


@app.route("/")
def index():
    """Page to display league leaders in points, rebounds, and blocks for the 2025-26 season."""
    data_file = os.path.join(os.path.dirname(__file__), "../data/data/nba_player_data.csv")
    top_players = {"points": [], "rebounds": [], "blocks": []}

    try:
        with open(data_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            players = [row for row in reader if row["SEASON"] == "2025-26"]  # Filter for 2025-26 season

            # Sort and get top 5 players for each stat
            top_players["points"] = sorted(players, key=lambda x: float(x["PTS"]), reverse=True)[:5]
            top_players["rebounds"] = sorted(players, key=lambda x: float(x["REB"]), reverse=True)[:5]
            top_players["blocks"] = sorted(players, key=lambda x: float(x["BLK"]), reverse=True)[:5]
            top_players["turnovers"] = sorted(players, key=lambda x: float(x["TOV"]), reverse=True)[:5]
            top_players["assists"] = sorted(players, key=lambda x: float(x["AST"]), reverse=True)[:5]
            top_players["steals"] = sorted(players, key=lambda x: float(x["STL"]), reverse=True)[:5]
    except FileNotFoundError:
        flash("Data file not found.", "error")
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")

    return render_template("index.html", top_points=top_players["points"], top_rebounds=top_players["rebounds"], top_blocks=top_players["blocks"], top_turnovers=top_players["turnovers"], top_assists=top_players["assists"], top_steals=top_players["steals"])


@app.route("/player_visualization", methods=["GET", "POST"])
def player_visualization():
    """Page to display a searched player's stats from the latest season."""
    if request.method == "POST":
        player_name = request.form.get("player_name")
        if not player_name:
            flash("Please enter a player's name.", "error")
            return redirect(url_for("player_visualization"))

        # Load data from nba_player_data.csv
        data_file = os.path.join(os.path.dirname(__file__), "../data/data/nba_player_data.csv")
        latest_season_data = None

        try:
            with open(data_file, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row["PLAYER_NAME"].lower() == player_name.lower():
                        if not latest_season_data or row["SEASON"] > latest_season_data["SEASON"]:
                            latest_season_data = row
        except FileNotFoundError:
            flash("Data file not found.", "error")
            return redirect(url_for("player_visualization"))

        if not latest_season_data:
            flash(f"No data found for player: {player_name}", "error")
            return redirect(url_for("player_visualization"))

        return render_template("player_viz.html", player_data=latest_season_data)

    return render_template("player_viz.html", player_data=None)


@app.route("/award_predictions")
def award_predictions():
    """Page to display league leaders in points, rebounds, and blocks for the 2025-26 season."""
    data_file = "/Users/lucaswong/Developer/python-projects/visual_nba/app/data/data/predictions.csv"
    top_players = {"MPV": [], "DPOY": [], "ALL_NBA_CONFIDENCE": []}
    try:
        with open(data_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            players = [row for row in reader]
            # Sort and get top 5 players for each stat
            top_players["MVP"] = sorted(players, key=lambda x: float(x["MVP_PROB"]), reverse=True)[:5]
            top_players["DPOY"] = sorted(players, key=lambda x: float(x["DPOY_PROB"]), reverse=True)[:5]

            players2 = [row for row in players if str(row["ALL_NBA_PREDICTION"]) == "1st Team"]  # Filter for 1st all nba team
            print(players2)
            top_players["ALL_NBA_CONFIDENCE"] = sorted(players2, key=lambda x: float(x["ALL_NBA_CONFIDENCE"]), reverse=True)[:5]
    except FileNotFoundError:
        flash("Data file not found.", "error")
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")

    return render_template("predictions.html", top_MVP=top_players["MVP"], top_DPOY=top_players["DPOY"], top_ALL_NBA_CONFIDENCE=top_players["ALL_NBA_CONFIDENCE"])


@app.route("/api/history/<player_name>/<stat_name>", methods=["GET"])
def api_history(player_name, stat_name):
    """
    API endpoint to get historical stats for a player for visualization.
    """
    data_file = os.path.join(os.path.dirname(__file__), "../data/data/nba_player_data.csv")
    historical_data = []

    try:
        with open(data_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["PLAYER_NAME"].lower() == player_name.lower():
                    historical_data.append({
                        "season": row["SEASON"],
                        "value": float(row[stat_name]) if stat_name in row and row[stat_name] else None
                    })

        if not historical_data:
            return {"error": f"No data found for player: {player_name}"}, 404

        return {"data": historical_data}, 200
    except FileNotFoundError:
        return {"error": "Data file not found."}, 500
    except KeyError:
        return {"error": f"Invalid stat name: {stat_name}"}, 400
    except Exception as e:
        return {"error": str(e)}, 500