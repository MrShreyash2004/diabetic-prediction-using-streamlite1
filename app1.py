#!/usr/bin/env python  
# coding: utf-8  

import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  

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
    model = LogisticRegression(random_state=0)  
    model.fit(X_train, y_train)  

    # Model accuracy  
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  

    return model, scaler, accuracy  

# Main Page  
def main_page(model, scaler):  
    st.title("🩺 Diabetes Prediction App")  
    st.write("Please enter patient details below to predict diabetes:")  

    with st.form(key='input_form', clear_on_submit=True):  
        # Input fields  
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)  
        glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)  
        blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)  
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)  
        age = st.number_input('Age', min_value=15, max_value=100, value=30)  
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)  
        bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")  
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, format="%.2f")  

        predict_button = st.form_submit_button(label='🔍 Predict')  

    if predict_button:  
        # Store inputs in session state for prediction page  
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

        # Redirect to prediction page  
        st.session_state.predicted = True  
        st.experimental_rerun()  # Reload the app to show prediction page  

# Prediction Page  
def prediction_page(model, scaler):  
    st.title("🩺 Prediction Result")  

    # Retrieve inputs from session state  
    input_data = pd.DataFrame(st.session_state.inputs, index=[0])  
    input_scaled = scaler.transform(input_data)  
    prediction = model.predict(input_scaled)  
    prediction_proba = model.predict_proba(input_scaled)  

    # Display prediction results    
    if prediction[0] == 1:  
        st.error('The model predicts that the patient is **Positive for Diabetes**')  
        confidence = prediction_proba[0][1]  
        color = 'red'  # Color for positive prediction  
    else:  
        st.success('The model predicts that the patient is **Negative for Diabetes**')  
        confidence = prediction_proba[0][0]  
        color = 'green'  # Color for negative prediction  

    # Visualize prediction as a pie chart  
    labels = ['Diabetes Positive', 'Diabetes Negative']  
    sizes = [confidence * 100, (1 - confidence) * 100]  
    colors = [color, 'lightgray']  
    explode = (0.1, 0)  # explode the 1st slice (diabetes positive)  

    plt.figure(figsize=(6, 4))  
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)  
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.  
    st.pyplot(plt)  

# Main app structure  
def main():  
    # Load data and build model  
    data = load_data()  
    model, scaler, accuracy = build_model(data)  
    
    # Ensure session state variables exist  
    if 'predicted' not in st.session_state:  
        st.session_state.predicted = False  
    if 'inputs' not in st.session_state:  
        st.session_state.inputs = {}  

    # Display the appropriate page  
    if not st.session_state.predicted:  
        main_page(model, scaler)  
    else:  
        prediction_page(model, scaler)  

if __name__ == '__main__':  
    main()
