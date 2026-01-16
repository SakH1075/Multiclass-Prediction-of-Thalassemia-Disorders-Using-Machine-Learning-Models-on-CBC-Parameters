# Multiclass Prediction of Thalassemia Disorders Using Machine Learning Models on CBC Parameters

## Overview
This repository contains code and trained machine learning (ML) models for **multiclass classification of thalassemia and related hemoglobinopathies** using **Complete Blood Count (CBC)** parameters derived from **High-Performance Liquid Chromatography (HPLC)**-linked datasets.  

The primary goal is to support **automated screening** of:
- **Thalassemia carriers**
- **Non-carriers (normal)**
- **Other hemoglobin disorders**

This work is designed to be **low-cost and accessible**, especially for **resource-limited settings**.

---

## Project Workflow (Methodology)

### 1) Data Preprocessing
Implemented in: `Data_Preprocessing_All.ipynb`

Steps include:

**Handling Missing Values**
- Numerical features (e.g., `MCV`, `MCH`, `RBC`) → **median imputation**
- Categorical features (e.g., `Gender`, `Weakness`) → **mode imputation**

**Outlier Detection & Handling**
- Outliers detected via **Z-score**
- If `|z| > 3`, the value is treated as an outlier and replaced with the **feature median**

**Target Label Standardization**
- Fixes inconsistencies such as extra spaces and mixed capitalization in the `Diagnosis` column
- Standardizes labels into a consistent naming convention (e.g., `"Normal"`)

**Data Normalization**
- Scales numerical features to **[0, 1]**
- Helpful for scale-sensitive models like **KNN** and **SVM**

---

### 2) Feature Selection
Method:
- **Random Forest Feature Importance**

Selected key features based on an importance threshold of **0.05**, including:
- `MCV`, `MCH`, `RDWcv`, `RBC`, `HB`, `MCHC`, `Age`, `Present District`

---

### 3) Data Balancing
Because of significant class imbalance (especially for rare subtypes):

- **Random Oversampling** is used to duplicate minority class samples  
- **SMOTE was considered** but not used due to extremely low sample counts in some classes

---

### 4) Modeling
Implemented in: `model-evaluation.ipynb`

Models trained and evaluated:
- Random Forest (RF)
- Gradient Boosting Classifier (GBC)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree (DT)
- LightGBM
- Voting Classifier
- Multi-layer Perceptron (MLP)

---

### 5) Model Evaluation
Metrics used:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- 5-Fold Cross-validation accuracy

Cross-validation is applied to improve generalizability and reduce overfitting risk.

---

### 6) Result Analysis & Visualization
Implemented in: `Plots_for_Result_Analysis.ipynb`

Includes plots and comparisons across models, typically showing:
- Accuracy per model
- F1-score per model
- Comparative performance visualization

---

## Repository Structure
```text
.
├── Datasets/                         # Dataset files (raw/processed)
├── Pictures/                         # Saved figures/plots (optional)
├── Data Preprocessing All.ipynb      # Cleaning + preprocessing + balancing
├── model-evaluation.ipynb            # Train + evaluate ML models (CV + metrics)
├── Plots for Result Analysis.ipynb   # Result visualization
└── README.md
