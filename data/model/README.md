# Saved Model

This directory contains the trained machine learning model and preprocessing artifacts used for customer churn prediction.

## Files

- `customer_churn_model.pkl` — Trained Random Forest classifier along with the feature names used during prediction.
- `encoders.pkl` — Saved LabelEncoder objects used to transform categorical features during inference.

These files allow the trained model to be loaded and used for predictions without retraining.
