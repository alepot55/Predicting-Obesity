# Obesity Risk Classification and Analysis

This repository contains the code and documentation for a project developed as part of the Fundamentals of Artificial Intelligence course in the Bachelor of Science in Computer Engineering program at Sapienza University of Rome (Academic Year 2023/2024).

The project focuses on analyzing and classifying obesity risk based on behavioral and demographic data using various Machine Learning models.

## Dataset

The project utilizes the "Obesity or CVD Risk" dataset, which is available on Kaggle: [https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cardiovascular-risk-classify/data](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cardiovascular-risk-classify/data).

The dataset includes 17 attributes and 2111 instances related to individuals from Mexico, Peru, and Colombia.

## Project Overview

The goal of this project is to develop and evaluate different Machine Learning models to predict obesity risk. The project involves the following steps:

1. **Data Preprocessing:**
    *   Handling categorical data by converting text labels into numerical values.
    *   Implementing one-hot encoding for variables with multiple categories.
    *   (Optional) Standardizing features to ensure proportional contribution to the learning process.
    *   (Optional) Discretizing continuous numerical variables.
2. **Model Development:**
    *   Linear Regression (with analytical solution and gradient descent, including stochastic gradient descent).
    *   Decision Trees.
    *   Logistic Regression (with gradient descent, including stochastic gradient descent).
    *   K-Nearest Neighbors (KNN).
    *   Neural Networks.
3. **Model Evaluation:**
    *   Using K-Fold Cross Validation for hyperparameter selection.
    *   Evaluating regression models with metrics like MSE, MAE, and RMSE.
    *   Evaluating classification models with metrics like Accuracy, Precision, Recall, F1 Score, and AUC.
    *   Visualizing the confusion matrix, ROC curve, and Precision-Recall curve.
4. **Hyperparameter Optimization:**
    *   Implementing Grid Search to find the optimal hyperparameter combinations for each model.

## Repository Structure

*   `notebook.ipynb`: Jupyter Notebook containing the code for data preprocessing, model development, training, evaluation, and analysis.
*   `README.md`: This file, providing an overview of the project.

## Dependencies

The project requires the following Python libraries:

*   NumPy
*   Pandas
*   scikit-learn
*   Matplotlib
*   Seaborn
*   printree

These can all be installed with pip using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn printree
