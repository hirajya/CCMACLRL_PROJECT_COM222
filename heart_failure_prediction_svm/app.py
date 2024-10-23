import streamlit as st
import pandas as pd
import joblib  

# # Relative paths to the model and scaler from app.py
# MODEL_PATH = '../models/model/10-23-2024_1227AM_SVMmodel.pkl'
# SCALER_PATH = '../models/scaler/10-23-2024_1227AM_scaler.pkl'  # Update this path as needed

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/model/10-23-2024_1227AM_SVMmodel.pkl"))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/scaler/10-23-2024_1227AM_scaler.pkl"))
model = joblib.load(MODEL_PATH)

print(BASE_DIR)
print(f"Model path: {MODEL_PATH}")
print(f"Scaler path: {SCALER_PATH}")

# Load the model
def load_model():
    model = joblib.load(MODEL_PATH)  # Use joblib's load function
    return model

# Load the scaler
def load_scaler():
    scaler = joblib.load(SCALER_PATH)  # Use joblib's load function
    return scaler

# Main function for the Streamlit app
def main():
    st.title("Heart Failure Prediction App")
    
    st.write("""
    This app predicts whether a patient is likely to experience heart failure based on clinical data.
    Please input the necessary information to get a prediction.
    """)
    
    # User input section
    age = st.slider("Age", 40, 95, 60)
    anaemia = st.selectbox("Anaemia", ["Yes", "No"])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", 23, 7861, 250)
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    ejection_fraction = st.slider("Ejection Fraction (%)", 14, 80, 38)
    high_blood_pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
    platelets = st.number_input("Platelets (kiloplatelets/mL)", 25.0, 850.0, 250.0)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.0)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 113, 150, 137)
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    time = st.number_input("Follow-up Time (days)", 1, 300, 150)

    # Convert user inputs to model input format
    user_input = {
        'age': age,
        'anaemia': 1 if anaemia == "Yes" else 0,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': 1 if diabetes == "Yes" else 0,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': 1 if high_blood_pressure == "Yes" else 0,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': 1 if sex == "Male" else 0,
        'smoking': 1 if smoking == "Yes" else 0,
        'time': time
    }

    def predict_new_patient(scaler, svm_model, new_patient_data):
        
        # Scale the new data using the same scaler
        new_data_scaled = scaler.transform(new_patient_data)

        # Make predictions
        predictions = svm_model.predict(new_data_scaled)

        return predictions

    # Convert dictionary to DataFrame for prediction
    input_df = pd.DataFrame([user_input])

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()
    
    if st.button("Predict"):
        print(user_input)
        prediction = predict_new_patient(scaler, model, input_df)  # Use the scaled input for prediction
        print(prediction)
        result = "Heart Failure" if prediction[0] == 1 else "No Heart Failure"
        
        st.subheader("Prediction Result:")
        st.write(result)

if __name__ == "__main__":
    main()
