# Multiclass Prediction of Thalassemia Disorders Using Machine Learning Models on HPLC Data

## Overview

This repository contains code and models for classifying thalassemia and related hemoglobinopathies using machine learning (ML) methods on High-Performance Liquid Chromatography (HPLC) data. The primary goal is to automate the detection of thalassemia carriers, non-carriers, and individuals with other hemoglobin disorders, providing an accessible solution in resource-limited settings.

## Methodology

The approach developed in this project revolves around a series of essential steps:

1. **Data Preprocessing**: 
   - **Handling Missing Values**: The dataset contains some missing values for features like MCV, MCH, HbA2, and others. These are imputed using the median for numerical features and mode for categorical features (like gender or weakness).
   - **Outlier Detection**: Outliers are identified using the Z-score method. If values deviate more than 3 standard deviations from the mean, they are replaced with the median value of the feature.
   - **Data Normalization**: To standardize the range of numerical features, they are scaled between 0 and 1 to facilitate the convergence of certain algorithms, particularly scale-sensitive ones like KNN and SVM.

2. **Feature Selection**:
   - **Random Forest Feature Importance**: Key features such as HbA2, HbA0, MCV, and MCH are selected based on their importance for the classification task.
   - **ANOVA F-value**: Features are evaluated based on their ability to distinguish between different classes of thalassemia, using statistical significance.
   - **Recursive Feature Elimination (RFE)**: This technique is used to recursively remove the least significant features, focusing on the most informative ones.
   - **SHAP (SHapley Additive exPlanations)**: Used to explain model predictions by showing how each feature contributes to the final prediction.
   - **Information Gain**: This method assesses how much information each feature provides about the class, contributing to the classification task.

3. **Data Balancing**: Due to class imbalance (especially for rare thalassemia subtypes), Random Oversampling is employed to duplicate instances from minority classes. This prevents the model from being biased towards the majority class.

4. **Modeling**:
   - Various machine learning algorithms are used for thalassemia classification, including:
     - **Random Forest (RF)**
     - **Gradient Boosting Classifier (GBC)**
     - **Support Vector Machine (SVM)**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree (DT)**
     - **LightGBM**
     - **Voting Classifier**
     - **Multi-layer Perceptron (MLP)**

   - **Hyperparameter Tuning**: Hyperparameters are tuned using `GridSearchCV` to identify the optimal values for each model and improve their performance.

5. **Model Evaluation**:
   - **Metrics**: The models are evaluated using multiple performance metrics including:
     - Accuracy
     - F1-score (weighted)
     - Precision (weighted)
     - Recall (weighted)
     - Cross-validation accuracy (5-fold)
   - **Cross-validation** is used to ensure the model's generalizability and prevent overfitting.

![Overview of Methodology](./Pictures/Methodology.png)

## Code Structure

### Data Preprocessing

The preprocessing steps are implemented in the `Data Preprocessing All.ipynb` notebook. It handles tasks such as:

- Imputation of missing values using median or mode.
- Outlier detection and handling.
- Label standardization for target variable (Diagnosis).
- Data normalization for scale-sensitive models.

### Model Evaluation

The `Model Evaluation (Only LightGBM) without class balancing.ipynb` notebook focuses on training and evaluating the **LightGBM** model without class balancing, while the `model-evaluation.ipynb` evaluates multiple classifiers (including RF, SVM, GBC, KNN, etc.) using cross-validation and hyperparameter tuning.

### Result Analysis and Visualization

The `Plots for Result Analysis.ipynb` notebook contains visualizations to analyze and compare the results of different models. The performance of each model is visualized using charts that display accuracy, F1-scores, and cross-validation results.