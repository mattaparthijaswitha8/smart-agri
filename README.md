# Smart Agriculture Crop Recommendation (Machine Learning / Data Mining)

Recommends the best crop from **rainfall, average temperature and soil type**, with live
Decision Tree, Regression and K-Means data mining.

## Run (from INSIDE this folder - the one containing app.py)
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000   (Ctrl + C in the terminal to stop)

## Pages
- `/`          LIVE crop recommendation + decision tree + K-Means + regression + explorer, all on ONE page
- `/live`      same page (alias); works offline - d3 is bundled
    1. Live suitability + expected-yield bars (move rainfall / temperature / soil)
    2. Live Decision Tree - change depth / leaf size, the tree re-trains instantly,
       the red path shows how YOUR field is classified, click nodes to expand/collapse
    3. Live K-Means - animated centroid movement, change k and axes, inertia curve,
       cluster profiles, star = your field
    4. Live Regression - yield curves vs rainfall and temperature, red dot = predicted yield
    5. Live data explorer - hover, zoom, pan, click crops in the legend to filter
- `/analysis`  Data-mining report (heat-map, tree, confusion matrix, actual vs predicted ...)

## Files
- `data/crop_data.csv`   30,000-row dataset (rainfall, temperature, soil, pH, humidity, state, year, season, crop, yield)
- `generate_dataset.py`  creates the dataset (delete the CSV and run it to regenerate)
- `app.py`               Flask server + all ML models + APIs
- `static/live.js`       live dashboard code; `static/d3.min.js` is the bundled chart library
