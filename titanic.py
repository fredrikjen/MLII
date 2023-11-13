import streamlit as st
import pandas as pd
import joblib

# Load your trained model and the scaler
model = joblib.load("Titanic.pkl")
scaler = joblib.load("scaler_titanic.pkl")

# Function to predict Titanic survival
def predict_survival(features):
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform([features])
    # Predict using the SVM classifier
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit application layout
st.title('Titanic Survival Prediction')

# Input fields for the features your model requires
pclass = st.number_input("Pclass", min_value=1, max_value=3, value=1, step=1)
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
siblings_spouses_aboard = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
parents_children_aboard = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=10.0, step=0.5)

# Button to make prediction
if st.button('Predict'):
    # Convert sex to the format expected by the model
    sex_encoded = 1 if sex == "male" else 0

    features = [pclass, sex_encoded, age, siblings_spouses_aboard, parents_children_aboard, fare]
    prediction = predict_survival(features)
    
    if prediction == 1:
        st.success("The passenger is predicted to have Survived")
    else:
        st.success("The passenger is predicted to have Not Survived")
