import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved scaler and imputer
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('imputer.pkl', 'rb') as file:
    imputer = pickle.load(file)

# Define numerical features used in training
numerical_features = ['Rank', '$Worldwide', '$Domestic', 'Domestic %', '$Foreign',
                      'Foreign %', 'Year', 'Vote_Count']

# Streamlit app
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: Arial, sans-serif;
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #3b5998;
            text-align: center;
            margin-bottom: 10px;
        }
        .subheader {
            font-size: 24px;
            color: #ffffff;
            text-align: center;
            margin-top: 20px;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #dddddd;
            margin-bottom: 30px;
        }
        .stSidebar {
            background-color: #1e1e2f;
            padding: 10px;
        }
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            color: #ffffff; /* Darker text color for better contrast */
            margin-bottom: 10px;
        }
        .predict-button {
            background-color: #4caf50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px;
        }
        .stSlider > div[role='slider'] {
            background-color: #555 !important; /* Darker slider background */
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🎥 Movie Hit/Flop Prediction 🍿</div>', unsafe_allow_html=True)
st.markdown('<p class="description">Use this app to predict whether a movie will be a <b>Hit</b> or <b>Flop</b> based on its box office and other features.</p>', unsafe_allow_html=True)

# Input fields for the numerical features
st.sidebar.markdown('<div class="sidebar-header">🔢 Enter Movie Features</div>', unsafe_allow_html=True)

rank = st.sidebar.number_input("🏆 Rank", min_value=1, step=1, value=50)
worldwide = st.sidebar.number_input("🌎 Worldwide Box Office ($)", min_value=0.0, step=1e6, value=1e8)
domestic = st.sidebar.number_input("🏠 Domestic Box Office ($)", min_value=0.0, step=1e6, value=5e7)
domestic_pct = st.sidebar.slider("📊 Domestic %", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
foreign = st.sidebar.number_input("🌍 Foreign Box Office ($)", min_value=0.0, step=1e6, value=5e7)
foreign_pct = st.sidebar.slider("📈 Foreign %", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
year = st.sidebar.number_input("📅 Release Year", min_value=2000, max_value=2024, step=1, value=2022)
vote_count = st.sidebar.number_input("🗳️ Vote Count", min_value=0, step=1, value=5000)

# Collect inputs into a numpy array
input_data = np.array([[rank, worldwide, domestic, domestic_pct, foreign, foreign_pct, year, vote_count]])

# Preprocess the input data using the saved scaler and imputer
input_data_imputed = imputer.transform(input_data)  # Handle missing values
input_data_scaled = scaler.transform(input_data_imputed)  # Scale features

# Predict the result
if st.sidebar.button("🎯 Predict", key="predict-button"):
    prediction = model.predict(input_data_scaled)[0]
    result = "Hit 🍾" if prediction == 1 else "Flop 💔"
    st.markdown(f'<div class="subheader">Prediction: <b>{result}</b></div>', unsafe_allow_html=True)

# Footer with emoji
st.markdown("<footer style='text-align: center; margin-top: 50px;'>Made with ❤️ and Python 🐍</footer>", unsafe_allow_html=True)