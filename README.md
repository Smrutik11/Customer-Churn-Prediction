# Customer Churn Prediction using Machine Learning

## Overview

This project presents an end-to-end Machine Learning pipeline to predict customer churn using demographic information, service usage, and billing details. The objective is to identify customers who are likely to discontinue a service, enabling businesses to implement proactive customer retention strategies.

---

## Objectives

- Perform Exploratory Data Analysis (EDA) to understand customer behavior
- Preprocess data by handling missing values and encoding categorical features
- Address class imbalance using SMOTE
- Train and compare multiple Machine Learning models
- Select the best-performing model
- Predict churn for new customer records
- Save the trained model for future inference

---

## Dataset

- **Dataset:** Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Features:** 20
- **Target Variable:** Churn (Yes/No)

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Pickle
- Jupyter Notebook / Google Colab

---

## Project Workflow

1. Data Cleaning and Preprocessing
   - Removed irrelevant features
   - Handled missing values
   - Converted data types

2. Exploratory Data Analysis
   - Distribution analysis
   - Count plots
   - Box plots
   - Correlation heatmap

3. Feature Engineering
   - Label encoding of categorical variables
   - Preserved encoders for inference

4. Handling Class Imbalance
   - Applied SMOTE on the training dataset

5. Model Development
   - Decision Tree
   - Random Forest
   - XGBoost
   - 5-Fold Cross Validation

6. Model Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1-Score

7. Model Persistence
   - Saved the trained model and preprocessing objects using Pickle

8. Prediction
   - Generated churn prediction and prediction probability for unseen customer data

---

## Results

The Random Forest classifier achieved the best overall performance among the evaluated models.

| Metric | Value |
|---------|-------|
| Best Model | Random Forest |
| Accuracy | ~78% |
| Validation | 5-Fold Cross Validation |

The evaluation focused on Recall and F1-Score for the churn class due to its business significance.

---

## Sample Prediction

```text
Prediction: No Churn

Prediction Probability:
[[0.78 0.22]]
```

---


