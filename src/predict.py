import pickle
import pandas as pd


# Load trained model
with open("model/customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["feature_names"]


# Example customer data
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}


# Convert input to DataFrame
input_data_df = pd.DataFrame([input_data])


# Load saved encoders
with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


# Encode categorical features
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])


# Ensure correct feature order
input_data_df = input_data_df[feature_names]


# Generate prediction
prediction = loaded_model.predict(input_data_df)
prediction_probability = loaded_model.predict_proba(input_data_df)


# Display result
result = "Churn" if prediction[0] == 1 else "No Churn"

print(f"Prediction: {result}")
print(f"Prediction Probability: {prediction_probability[0]}")
