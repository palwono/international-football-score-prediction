# International Football Score Prediction: Advanced Ensemble & Dixon-Coles Correction ⚽

This repository contains the predictive modeling pipeline developed for the Data Science Competition GAMMAFEST 2026 organized by IPB University. The challenge focuses on forecasting international football match scores by tackling complex non-linear dynamics and low-scoring match probability shifts.

## 📌 Problem Statement
Predicting football match outcomes requires capturing both the offensive and defensive capabilities of teams, factoring in historical trends, tournament importance, and geographical advantages. The primary objective is to minimize a highly customized evaluation metric: the **Augmented Weighted Mean Absolute Error (AW-MAE)**, which strictly penalizes exact score misses, outcome errors, and goal difference discrepancies.

## 📊 Dataset
The dataset encompasses international matches from 1872 to 2026, enriched with multiple dimensions:
* **Match Context:** Tournament type (e.g., FIFA World Cup, Friendly), neutral venue indicators, and host country details.
* **Performance Metrics:** Elo ratings, FIFA rankings, historical head-to-head records, and recent form (win rates, points).
* **Socio-Economic & Geographical Factors:** Population sizes, GDP per capita, venue altitude, and estimated travel distances.

## 🛠️ Methodology & Architecture
* **Advanced Feature Engineering:** Extracted 68+ robust features, leveraging smooth and time-decayed recency target encoding to capture team dynamics and high-stakes tournament strengths.
* **Diverse ML Ensemble:** Trained LightGBM, XGBoost, and CatBoost models, optimizing simultaneously for Poisson (count) and L1/MAE (regression) objectives.
* **Systematic Bias Calibration:** Applied Isotonic Calibration to correct underlying model prediction biases.
* **Dixon-Coles Correction:** Utilized Maximum Likelihood Estimation (MLE) to mathematically adjust probabilities for low-scoring matches (e.g., 0-0, 1-0), addressing the limitations of independent Poisson distributions.
* **Joint Match-Level Optimization:** Calculated the expected AW-MAE loss matrix across all possible scorelines to select the optimal discrete prediction for both teams jointly.

## 🚀 Results
Evaluated against the ground truth dataset, the optimized ensemble pipeline successfully reduced goal prediction errors and achieved a final AW-MAE score of **3.0941**.
