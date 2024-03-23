import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart disease Prediction App

This app predicts if a patient has a heart disease

Data obtained from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset.
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.selectbox('Chest pain type', (0, 1, 2, 3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar', (0, 1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina: ', (0, 1))
    old = st.sidebar.number_input('Oldpeak ')
    slope = st.sidebar.number_input('The slope of the peak exercise ST segment: ')
    ca = st.sidebar.selectbox('Number of major vessels', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Thal', (0, 1, 2))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': tres,
        'chol': chol,
        'fbs': fbs,
        'restecg': res,
        'thalach': tha,
        'exang': exa,
        'oldpeak': old,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load the heart disease dataset for encoding purposes
heart_dataset = pd.read_csv("C:\sai\major project\heart_disease_data.csv")
heart_dataset = heart_dataset.drop(columns=['target'])

# Combine user input features with the entire dataset
df = pd.concat([input_df, heart_dataset], axis=0)

# Ensure consistent feature columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df[categorical_columns] = df[categorical_columns].astype(int)

# Select only the first row (user input data)
df = df.iloc[0:1]

st.write(input_df)

# Reads in the saved classification model
model_file = "C:\sai\major project\svm_model.pkl"
with open(model_file, 'rb') as model_pkl:
    model = pickle.load(model_pkl)

# Apply the model to get decision function values
decision_values = model.decision_function(df)

# Use decision function values to compute probabilities
prediction_proba = 1 / (1 + np.exp(-decision_values))

# Convert probabilities to class labels (0 or 1)
prediction = (prediction_proba > 0.5).astype(int)

st.subheader('Prediction')
if prediction[0] == 1:
    st.write("The model predicts that the user has heart disease.")
else:
    st.write("The model predicts that the user does not have heart disease.")

st.subheader('Prediction Probability')
st.write(prediction_proba)
