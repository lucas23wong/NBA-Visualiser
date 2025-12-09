### Technical Design Document: NBA PlayerAI

1. Project Structure and Components

The NBA PlayerAI project can be split into 3 components: Data, Backend, and Frontend. The application is designed for simplicity, utilizing Flask as a lightweight micro-framework and client-side rendering for visualizations.

The core components of the system are:

Data Scraping (fetch_data.py): Built with Python and the nba_api library. Its role is to fetch raw player and team statistics from the NBA's official endpoints for multiple historical seasons and the current 2025-26 NBA season. 

Data Pipeline (preprocess.py): Uses Python and pandas for data transformation. This is where data is cleaned, feature engineering is performed (e.g., per-36 min calculations, team win percentage inclusion), and historical MVP winners are labeled for model training.

Machine Learning (/ml_models): Uses Python, scikit-learn, and joblib. This component trains classification models, logistic Regression or Random Forest, on the processed historical data to predict awards. Trained model artifacts (.joblib files) are persisted to disk to use in the web application. 

Backend (Flask - app.py): This is the application server which isimplemented with Flask. It handles all routing, template rendering using Jinja2, and serves the critical API endpoint for fetching historical data (/api/history).

Frontend (HTML/D3.js): This is the user interface for data visualization. It combines static HTML templates, dynamic rendering via Jinja2, Tailwind CSS styling, and uses D3.js to draw the interactive historical trend charts in the user's browser.

----------------------------------------------------------------

2. Backend Design Decisions (app.py)

The entire backend logic is housed in a single app.py file. This design choice simplifies dependency management and project deployment while maintaining clear separation of concerns within the code.

A. Routing and Templating

All routes (e.g., /, /predictions, /player/<id>) are handled by Flask decorators.

Data Pre-loading: For the static pages (Home and Predictions), necessary data (leaderboards from CSV, predictions from JSON) is read once or on-demand and passed directly to the Jinja templates during render_template.

Template Structure: The application follows standard Flask practices using a base template (base.html) for consistent navigation, and page-specific templates (index.html, predictions.html, player_viz.html) extend it.

----------------------------------------------------------------

3. Data Pipeline and ML Structure

The data pipeline is designed to be executed before the web application starts as I wanted to  separate the computationally intensive training and prediction processes from the web serving process.

A. Data Persistence

Source Data: Raw and processed data are stored as static CSV files in the project's data directory.

Prediction Output: The final prediction results are saved as a static JSON file (app/static/data/predictions.json). The Flask application reads this static file directly. This design decision ensures prediction results are consistent and fast to load, as the web app doesn't have to wait for the ML models to re-run time and time again.

B. Feature Engineering (in preprocess.py)

Feature engineering is crucial in improving the predictive accuracy of the ML models. Key features I chose to derive:

- Rate Statistics: Converting absolute numbers (PTS, REB, AST) to per-36 minutes statistics to normalize for playing time.

- Team Context: Incorporating winning percentage (W_PCT) and Net Rating (NET_RATING) of the player's team, as team success is a known to be correlated with awards like MVP.

- Advanced Metrics: Utilizing key league-wide advanced metrics like True Shooting Percentage (TS_PCT), Usage Percentage (USG_PCT), and Player Impact Estimate (PIE) provides new lenses to view a player's effectiveness on the court. 

----------------------------------------------------------------

4. Frontend Visualization Decisions

The visualization uses client-side rendering for responsiveness and interactivity:

A. D3.js for Player Trends

Choice Rationale: D3.js was chosen for the player history chart because it offers maximum control over data binding, scales, and axes, allowing for the precise rendering of the time-series data (seasons vs. statistic value).

Interactivity: Using D3.js enables smooth transitions and interactive elements to the web application's graph, such as mouseover events that display tooltips with exact season values, enhancing the user's data exploration experience.

B. Styling

Tailwind CSS: All styling is managed via Tailwind CSS utility classes. This gives the web application a modern, NBA colour theme, and fully responsive design without the need for custom, extensive CSS files allowing me to focus on the data scraping and ML aspect of the project. 