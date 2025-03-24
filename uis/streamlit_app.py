##################################
# Loading Python libraries
##################################
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

##################################
# Defining FastAPI endpoint URL
##################################
SP_FASTAPI_BASE_URL = "http://127.0.0.1:8001/plot_coxph_survival_profile/"

##################################
# Setting the page layout to wide
##################################
st.set_page_config(layout="wide")

##################################
# Listing the variables
##################################
variables = ['AGE',
             'EJECTION_FRACTION',
             'SERUM_CREATININE',
             'SERUM_SODIUM',
             'ANAEMIA',
             'HIGH_BLOOD_PRESSURE']

##################################
# Initializing lists to store user responses
##################################
test_case_request = {}

##################################
# Creating a title for the application
##################################
st.markdown("""---""")
st.markdown("<h1 style='text-align: center;'>Heart Failure Survival Probability Estimator</h1>", unsafe_allow_html=True)

##################################
# Providing a description for the application
##################################
st.markdown("""---""")
st.markdown("<h5 style='font-size: 20px;'>This model evaluates the heart failure survival risk of a test case based on certain cardiovascular, hematologic and metabolic markers. Pass the appropriate details below to visually assess your characteristics against the study population, plot your survival probability profile, estimate your heart failure survival probabilities at different time points, and determine your risk category. For more information on the complete model development process, you may refer to this <a href='https://johnpaulinepineda.github.io/Portfolio_Project_55/' style='font-weight: bold;'>Jupyter Notebook</a>. Additionally, all associated datasets and code files can be accessed from this <a href='https://github.com/JohnPaulinePineda/Portfolio_Project_55' style='font-weight: bold;'>GitHub Project Repository</a>.</h5>", unsafe_allow_html=True)

##################################
# Creating a section for 
# selecting the options
# for the test case characteristics
##################################
st.markdown("""---""")
st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Cardiovascular, Hematologic and Metabolic Markers</h4>", unsafe_allow_html=True)
st.markdown("""---""")

##################################
# Creating sliders for numeric features
# and radio buttons for categorical features
# and storing the user inputs
##################################
input_column_1, input_column_2, input_column_3, input_column_4, input_column_5, input_column_6, input_column_7, input_column_8 = st.columns(8)
with input_column_2:
    age_numeric_input = st.slider(variables[0], min_value=20, max_value=100, value=20)
with input_column_3:
    ejection_fraction_numeric_input = st.slider(variables[1], min_value=10, max_value=80, value=10)
with input_column_4:
    serum_creatinine_numeric_input = st.slider(variables[2], min_value=0.5, max_value=10.0, value=0.5)
with input_column_5:
    serum_sodium_numeric_input = st.slider(variables[3], min_value=110, max_value=150, value=50)
with input_column_6:
    anaemia_categorical_input = st.radio(variables[4], ('Present', 'Absent'), horizontal=True)
    anaemia_numeric_input = 1 if anaemia_categorical_input == 'Present' else 0
with input_column_7:
    high_blood_pressure_categorical_input = st.radio(variables[5], ('Present', 'Absent'), horizontal=True)
    high_blood_pressure_numeric_input = 1 if high_blood_pressure_categorical_input == 'Present' else 0

st.markdown("""---""")

##################################
# Consolidating the user inputs
# in the defined format for the API endpoint
##################################  
test_case_request = {
        "AGE": age_numeric_input,
        "EJECTION_FRACTION": ejection_fraction_numeric_input,
        "SERUM_CREATININE": serum_creatinine_numeric_input,
        "SERUM_SODIUM": serum_sodium_numeric_input,
        "ANAEMIA": anaemia_numeric_input,
        "HIGH_BLOOD_PRESSURE": high_blood_pressure_numeric_input
    }

st.markdown("""
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

entered = st.button("Assess Characteristics Against Study Population + Plot Survival Probability Profile + Estimate Heart Failure Survival Probability + Predict Risk Category")

##################################
# Defining the code logic
# for the button action
##################################    
if entered:
    ##################################
    # Defining a section title
    # for the test case characteristics
    ##################################    
    st.markdown("""---""")      
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Characteristics</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 

    ##################################
    # Defining a section title
    # for the test case heart failure survival probability estimation
    ################################## 
    st.markdown("""---""")    
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Heart Failure Survival Probability Estimation</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 

    ##################################
    # Sending a request to FastAPI
    ##################################
    response = requests.post(SP_FASTAPI_BASE_URL, json=test_case_request)
    
    if response.status_code == 200:
        plot_data = response.json()["plot"]
        img = base64.b64decode(plot_data)
        image = Image.open(BytesIO(img))
        st.image(image, use_column_width=True)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

    ##################################
    # Defining a section title
    # for the test case model prediction summary
    ##################################      
    st.markdown("""---""")   
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Model Prediction Summary</h4>", unsafe_allow_html=True)    
    st.markdown("""---""")

    
    st.markdown("""---""")



