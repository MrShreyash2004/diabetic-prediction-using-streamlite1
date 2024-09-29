#!/usr/bin/env python  
# coding: utf-8  

import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score   

# Load Dataset  
@st.cache_data  
def load_data():  
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"  
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',  
                    'DiabetesPedigreeFunction', 'Age', 'Outcome']  
    data = pd.read_csv(url, names=column_names)  
    return data  

# Build Model  
def build_model(data):  
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values  

    # Split the data into training and testing sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

    # Standardize features  
    scaler = StandardScaler()  
    X_train = scaler.fit_transform(X_train)  
    X_test = scaler.transform(X_test)  

    # Train the model  
    model = LogisticRegression(random_state=0, max_iter=1000)  # Increased max_iter for convergence  
    model.fit(X_train, y_train)  

    # Model accuracy  
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  

    return model, scaler, accuracy  

# Input Page  
def input_page():  
    st.title("🩺 Diabetes Prediction App")  
    st.write("Please enter patient details below to predict diabetes:")  

    with st.form(key='input_form', clear_on_submit=True):  
        st.header("Patient Information")  

        # Input fields without columns for better visibility  
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1, help="Number of times the patient has been pregnant.")  
        glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test.")  
        blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70, help="Diastolic blood pressure (mm Hg).")  
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80, help="2-Hour serum insulin (mu U/ml).")  
        age = st.number_input('Age', min_value=15, max_value=100, value=30, help="Age of the patient in years.")  
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, help="Triceps skin fold thickness measured in mm.")  
        bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f", help="Weight in kg/(height in m)^2.")  
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, format="%.2f", help="Diabetes pedigree function.")  

        submit_button = st.form_submit_button(label='🔍 Predict')  

    if submit_button:  
        # Store inputs in session state for use on the output page  
        st.session_state.inputs = {  
            'Pregnancies': pregnancies,  
            'Glucose': glucose,  
            'BloodPressure': blood_pressure,  
            'SkinThickness': skin_thickness,  
            'Insulin': insulin,  
            'BMI': bmi,  
            'DiabetesPedigreeFunction': diabetes_pedigree,  
            'Age': age  
        }  
        st.session_state.page = "output"  # Navigate to output page  

        st.success("Thank you! Your data has been submitted. Click on the 'Predict' button for the results.")  

# Output Page  
def output_page(model, scaler, accuracy):  
    st.title("📊 Prediction Result")  

    input_data = pd.DataFrame(st.session_state.inputs, index=[0])  
    
    # Display user input vertically using markdown
    st.subheader('Patient Input')
    for col in input_data.columns:
        st.markdown(f"**{col}**: {input_data[col].values[0]}")

    # Predict diabetes  
    input_scaled = scaler.transform(input_data)  
    prediction = model.predict(input_scaled)  
    prediction_proba = model.predict_proba(input_scaled)  

    # Display prediction results  
    st.subheader('Prediction Result')  
    if prediction[0] == 1:  
        st.error('😔 The model predicts that the patient is **Positive for Diabetes**.')  
        st.write("But don’t worry! With the right diet and exercise, diabetes is manageable. Stay strong and consult with your doctor for further steps.")  
    else:  
        st.success('🎉 The model predicts that the patient is **Negative for Diabetes**!')  
        st.write("Keep up the healthy lifestyle! You’re doing great, but regular check-ups are always a good idea to stay on top of your health.")  

    # Correctly display the confidence score
    confidence = prediction_proba[0][prediction[0]] * 100
    st.write(f"**Confidence of Prediction:** {confidence:.2f}%")  

    # Button to predict another patient
    if st.button('🔄 Predict Another Patient'):
        st.session_state.page = "input"  # Navigate back to input page

# Main Function  
def main():  
    # Load data and build model  
    data = load_data()  
    model, scaler, accuracy = build_model(data)  

    # Page navigation  
    if 'page' not in st.session_state:  
        st.session_state.page = "input"  # Default to input page  

    if st.session_state.page == "input":  
        input_page()  
    else:  
        output_page(model, scaler, accuracy)  

if __name__ == '__main__':  
    main()
