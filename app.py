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
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    # Model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy

# Input Page
def input_page():
    st.title("Diabetes Prediction App")
    st.write("Please enter patient details below:")

    with st.form(key='input_form'):
        st.header("Patient Information")

        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
        glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
        bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
        age = st.number_input('Age', min_value=15, max_value=100, value=30)

        submit_button = st.form_submit_button(label='Predict')

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

# Output Page
def output_page(model, scaler, accuracy):
    st.title("Prediction Result")

    input_data = pd.DataFrame(st.session_state.inputs, index=[0])
    
    # Display user input
    st.subheader('Patient Input')
    st.write(input_data)

    # Predict diabetes
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display prediction results
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('The model predicts that the patient is **Positive for Diabetes**')
    else:
        st.success('The model predicts that the patient is **Negative for Diabetes**')

    st.write(f"Prediction Confidence: {prediction_proba[0][prediction][0]*100:.2f}%")

# Main app structure
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
