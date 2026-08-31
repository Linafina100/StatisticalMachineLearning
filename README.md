# StatisticalMachineLearning
# Bike Demand Prediction Model

This repository contains Python scripts for exploring and predicting bike sharing demand using environmental and temporal data.

## Project Structure

* **`data_exploration.py`**: Analyzes the dataset (`training_data_vt2025.csv`) to uncover patterns in bike demand. It maps categorical variables, handles missing values, and generates visualizations (time-based, holiday vs. weekday, and weather distributions).
* **`model_training.py`**: Focuses on feature engineering, feature selection, and evaluating various machine learning models (Logistic Regression, Random Forest, LDA, QDA, KNN, and a Dummy Classifier) to predict if stock needs increasing. 

## Features & Methodology

* **Feature Engineering**: Creates custom features including `is_winter`, `rush_hour` (15:00-19:00), `night_time` (20:00-07:00), and `is_there_snow` based on snow depth.
* **Feature Selection**: Compares features using Recursive Feature Elimination (RFE) across Logistic Regression, Random Forest, and LDA, as well as `SelectKBest` with ANOVA F-value (`f_classif`).
* **Model Evaluation**: Implements `GridSearchCV` (10-fold cross-validation) to find optimal hyperparameters[cite: 6]. Models are evaluated on Test Accuracy, Precision, Recall, and F1-score.

## Dependencies
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
