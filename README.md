# House Valuation System

Predicting residential property prices on the **Melbourne housing dataset** and serving the trained
model through a small **Flask** web app for live, interactive predictions.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## Overview

This project covers a complete regression workflow — from cleaning the raw Melbourne listings to
training and comparing several models, then exposing the best one behind a web form where a user enters
a few property attributes and gets an estimated price.

- **Dataset:** Melbourne housing data (`data.csv.csv`) — suburb, rooms, type, distance, land size,
  bathrooms, and more.
- **Task:** Regression — predict `Price` from property features.
- **Models explored** (see the notebook): Linear Regression, Decision Tree, Random Forest and XGBoost,
  with missing-value imputation and preprocessing pipelines via scikit-learn.
- **App:** A Flask app (`main.py`) loads the trained model (`house.joblib`) and predicts price from
  four inputs — number of rooms, bathrooms, land size, and distance from the city centre.

## Project structure

```
├── Price Prediction Machine using Machine learning.ipynb   # EDA, preprocessing, model training & comparison
├── mode_usage.ipynb                                         # Loading and using the saved model
├── main.py                                                  # Flask app serving predictions
├── templates/index.html                                     # Prediction form UI
├── static/                                                  # CSS + JS for the UI
├── data.csv.csv                                             # Melbourne housing dataset
└── house.joblib                                             # Trained, serialized model
```

## How to run

```bash
# 1. Install dependencies
pip install flask scikit-learn xgboost numpy pandas joblib

# 2. Start the web app (loads house.joblib)
python main.py

# 3. Open http://127.0.0.1:5000 and enter the property details
```

To retrain from scratch, run **`Price Prediction Machine using Machine learning.ipynb`**, which loads
`data.csv.csv`, trains the models, and re-exports `house.joblib`.

## Notes

- The trained model `house.joblib` is committed for convenience so the app runs out of the box. It is a
  large binary; if you retrain, you can regenerate it from the notebook rather than versioning new copies.
- This was one of my earlier end-to-end ML projects and is kept here as a learning/portfolio piece.
