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
SP_FASTAPI_BASE_URL = "http://127.0.0.1:8001"

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
##################################
# First row input
##################################
row1_col1, _, _ = st.columns(3)
with row1_col1:
    age_numeric_input = st.slider("AGE (Years)", min_value=20, max_value=100, value=20)
##################################
# Second row input
##################################
row2_col1, _, _ = st.columns(3)
with row2_col1:
    ejection_fraction_numeric_input = st.slider("EJECTION FRACTION (%)", min_value=10, max_value=80, value=10)
##################################
# Third row input
##################################
row3_col1, _, _ = st.columns(3)
with row3_col1:
    serum_creatinine_numeric_input = st.slider("SERUM CREATININE (mg/dL)", min_value=0.5, max_value=10.0, value=0.5)
##################################
# Fourth row input
##################################
row4_col1, _, _ = st.columns(3)
with row4_col1:
    serum_sodium_numeric_input = st.slider("SERUM SODIUM (mEq/L)", min_value=110, max_value=150, value=50)
##################################
# Fifth row input
##################################
row5_col1, _, _ = st.columns(3)
with row5_col1:
    anaemia_categorical_input = st.radio("ANAEMIA INDICATION", ('Present', 'Absent'), horizontal=True)
    anaemia_numeric_input = 1 if anaemia_categorical_input == 'Present' else 0
##################################
# Sixth row input
##################################
row6_col1, _, _ = st.columns(3)
with row6_col1:
    high_blood_pressure_categorical_input = st.radio("HIGH BLOOD PRESSURE INDICATION", ('Present', 'Absent'), horizontal=True)
    high_blood_pressure_numeric_input = 1 if high_blood_pressure_categorical_input == 'Present' else 0

st.markdown("""---""")

##################################
# Consolidating the user inputs
# in the defined format for the API endpoint
##################################  
test_case_request = {
        variables[0]: age_numeric_input,
        variables[1]: ejection_fraction_numeric_input,
        variables[2]: serum_creatinine_numeric_input,
        variables[3]: serum_sodium_numeric_input,
        variables[4]: anaemia_numeric_input,
        variables[5]: high_blood_pressure_numeric_input
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
    # Sending a plot-coxph-survival-profile request to FastAPI
    ##################################
    response = requests.post(f"{SP_FASTAPI_BASE_URL}/plot-kaplan-meier-grid/", json=test_case_request)
    
    if response.status_code == 200:
        plot_data = response.json()["plot"]
        img = base64.b64decode(plot_data)
        image = Image.open(BytesIO(img))
        st.image(image, use_column_width=True)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

    ##################################
    # Defining a section title
    # for the test case heart failure survival probability estimation
    ################################## 
    st.markdown("""---""")    
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Case Heart Failure Survival Probability Estimation</h4>", unsafe_allow_html=True)    
    st.markdown("""---""") 

    ##################################
    # Sending a plot-coxph-survival-profile request to FastAPI
    ##################################
    response = requests.post(f"{SP_FASTAPI_BASE_URL}/plot-coxph-survival-profile/", json=test_case_request)
    
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

    ##################################
    # Sending a compute-test-coxph-survival-probability-class request to FastAPI
    ##################################
    response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-test-coxph-survival-probability-class/", json=test_case_request)
    
    if response.status_code == 200:
        result = response.json()
        survival_probabilities = result["survival_probabilities"]
        risk_category = result["risk_category"]
        
        color = "blue" if risk_category == "Low-Risk" else "red"
        
        time_points = [50, 100, 150, 200, 250]
        for i, time in enumerate(time_points):
            st.markdown(
                f"<h4 style='font-size: 20px;'>Estimated Heart Failure Survival Probability ({time} Days): <span style='color:{color};'>{survival_probabilities[i]:.5f}%</span></h4>",
                unsafe_allow_html=True
            )
        
        st.markdown(
            f"<h4 style='font-size: 20px;'>Predicted Risk Category: <span style='color:{color};'>{risk_category}</span></h4>",
            unsafe_allow_html=True
        )
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

    st.markdown("""---""")



