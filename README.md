# Customer-Churn-Prediction
# Customer Churn Prediction using Machine Learning

##  Project Overview
Customer churn refers to customers who stop using a company’s service.  
This project builds an **end-to-end Machine Learning pipeline** to predict whether a customer is likely to churn based on demographic details, service usage, and billing information.

The model helps businesses **identify high-risk customers early** and take preventive actions to improve retention.

## Objectives
- Analyze customer behavior using Exploratory Data Analysis (EDA)
- Handle missing values and categorical features
- Address class imbalance using SMOTE
- Train and compare multiple ML models
- Select the best-performing model
- Predict churn for new/unseen customers
- Save and load the trained model for deployment

## Dataset
- **Source:** Telco Customer Churn Dataset
- **Rows:** 7,043 customers
- **Features:** 20 (Demographics, services, billing)
- **Target Variable:** `Churn` (Yes / No)

##  Tech Stack & Tools
- **Programming Language:** Python  
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Imbalanced-learn (SMOTE)
  - XGBoost
  - Pickle
- **Platform:** Google Colab / Jupyter Notebook

## Project Workflow
1. **Data Loading & Cleaning**
   - Removed irrelevant columns (CustomerID)
   - Handled missing values in `TotalCharges`
   - Converted data types

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Box plots for numerical features
   - Count plots for categorical features
   - Correlation heatmap

3. **Feature Engineering**
   - Label Encoding for categorical variables
   - Stored encoders for future inference

4. **Handling Class Imbalance**
   - Applied **SMOTE** on training data

5. **Model Training & Evaluation**
   - Decision Tree
   - Random Forest
   - XGBoost
   - 5-fold Cross Validation
   - Metrics: Accuracy, Precision, Recall, F1-score

6. **Model Selection**
   - Random Forest achieved the best performance with default parameters

7. **Model Saving & Loading**
   - Saved trained model with feature names using `pickle`
   - Saved encoders for consistent preprocessing

8. **Prediction on New Data**
   - Encoded new customer input
   - Preserved feature order
   - Generated churn prediction and probability

## Model Performance (Random Forest)
- **Accuracy:** ~78%
- Focused on **Recall & F1-score** for churn class due to business importance
- Output includes both:
  - Churn Prediction (`Churn` / `No Churn`)
  - Prediction Probability (`[[P(No Churn), P(Churn)]]`)

## Sample Output
Prediction: No Churn; 
Prediction Probability: [[0.78 0.22]]
