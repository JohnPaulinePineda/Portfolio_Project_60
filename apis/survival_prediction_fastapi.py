##################################
# Loading Python libraries
##################################
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import joblib
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')

##################################
# Defining file paths
##################################
MODELS_PATH = r"models"
PARAMETERS_PATH = r"parameters"
DATASETS_PATH = r"datasets"
PIPELINES_PATH = r"pipelines"

##################################
# Loading the original training predictor data
##################################
try:
    X_train = pd.read_csv(os.path.join("..", DATASETS_PATH, "X_train.csv"), index_col=0)
except Exception as e:
    raise RuntimeError(f"Error loading original training predictor data: {str(e)}")

##################################
# Loading the original training response data
##################################
try:
    y_train = pd.read_csv(os.path.join("..", DATASETS_PATH, "y_train.csv"), index_col=0)
except Exception as e:
    raise RuntimeError(f"Error loading original response training data: {str(e)}")

##################################
# Loading the original preprocessed training data
##################################
try:
    x_original_EDA = pd.read_csv(os.path.join("..", DATASETS_PATH, "heart_failure_EDA.csv"), index_col=0)
except Exception as e:
    raise RuntimeError(f"Error loading original preprocessed training data: {str(e)}")

##################################
# Loading the model preprocessing pipeline
##################################
try:
    coxph_pipeline = joblib.load(os.path.join("..", PIPELINES_PATH, "coxph_pipeline.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model processing pipeline: {str(e)}")

##################################
# Loading the model
##################################
try:
    final_survival_prediction_model = joblib.load(os.path.join("..", MODELS_PATH, "coxph_best_model.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

##################################
# Loading the median parameter
##################################
try:
    numeric_feature_median = joblib.load(os.path.join("..", PARAMETERS_PATH, "numeric_feature_median_list.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model feature median: {str(e)}")

##################################
# Loading the threshold parameter
##################################
try:
    final_survival_prediction_model_risk_group_threshold = joblib.load(os.path.join("..", PARAMETERS_PATH, "coxph_best_model_risk_group_threshold.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model risk group threshold: {str(e)}")

##################################
# Defining the input schema for the function that
# preprocesses an individual test case
# and plots the estimated survival profiles
# using a Kaplan-Meier Plot grid
##################################
class TestCaseRequest(BaseModel):
    AGE: float
    EJECTION_FRACTION: float
    SERUM_CREATININE: float
    SERUM_SODIUM: float
    ANAEMIA: int
    HIGH_BLOOD_PRESSURE: int    

##################################
# Defining the input schema for the function that
# generates the heart failure survival profile,
# estimates the heart failure survival probabilities,
# and predicts the risk category
# of an individual test case
##################################
class TestSample(BaseModel):
    features_individual: List[float]

##################################
# Defining the input schema for the function that
# generates the heart failure survival profile and
# estimates the heart failure survival probabilities
# of a list of train cases
##################################
class TrainList(BaseModel):
    features_list: List[List[float]]

##################################
# Defining the input schema for the function that
# creates dichotomous bins for the numeric features
# of a list of train cases
##################################
class BinningRequest(BaseModel):
    X_original_list: List[dict]
    numeric_feature: str

##################################
# Defining the input schema for the function that
# plots the estimated survival profiles
# using Kaplan-Meier Plots
##################################
class KaplanMeierRequest(BaseModel):
    df: List[dict]
    cat_var: str
    new_case_value: Optional[str] = None

##################################
# Formulating the API endpoints
##################################

##################################
# Initializing the FastAPI app
##################################
app = FastAPI()

##################################
# Defining a GET endpoint for
# for validating API service connection
##################################
@app.get("/")
def root():
    return {"message": "Welcome to the Survival Prediction API!"}

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
@app.post("/compute-individual-coxph-survival-probability-class/")
def compute_individual_coxph_survival_probability_class(sample: TestSample):
    try:
        # Defining expected column names for the input data
        column_names = ["AGE", "ANAEMIA", "EJECTION_FRACTION", "HIGH_BLOOD_PRESSURE", "SERUM_CREATININE", "SERUM_SODIUM"]

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_test_sample = pd.DataFrame([sample.features_individual], columns=column_names)

        # Obtaining the survival function for an individual test case
        survival_function = final_survival_prediction_model.predict_survival_function(X_test_sample)

        # Predicting the risk category based on survival model output
        risk_category = (
            "High-Risk"
            if (final_survival_prediction_model.predict(X_test_sample)[0] > final_survival_prediction_model_risk_group_threshold)
            else "Low-Risk"
        )

        # Defining the specific survival time points for probability estimation
        survival_time = np.array([50, 100, 150, 200, 250])

        # Estimating survival probabilities at defined time points using interpolation
        survival_probability = np.interp(
            survival_time, survival_function[0].x, survival_function[0].y
        )
        survival_probabilities = survival_probability * 100

        # Returning computed survival function, probabilities, and risk category
        return {
            "survival_function": survival_function[0].y.tolist(),
            "survival_time": survival_time.tolist(),
            "survival_probabilities": survival_probabilities.tolist(),
            "risk_category": risk_category,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
@app.post("/compute-test-coxph-survival-probability-class/")
def compute_test_coxph_survival_probability_class(test_case: TestCaseRequest):
    try:
        # Converting the data input to a pandas DataFrame with appropriate column names
        X_test_sample = pd.DataFrame([test_case.dict()])

        # Computing the survival function for the given test case
        survival_function = final_survival_prediction_model.predict_survival_function(X_test_sample)

        # Predicting the risk category based on survival model output
        risk_category = (
            "High-Risk"
            if (final_survival_prediction_model.predict(X_test_sample)[0] > final_survival_prediction_model_risk_group_threshold)
            else "Low-Risk"
        )

        # Defining the specific survival time points for probability estimation
        survival_time = np.array([50, 100, 150, 200, 250])

        # Estimating survival probabilities at defined time points using interpolation
        survival_probability = np.interp(
            survival_time, survival_function[0].x, survival_function[0].y
        )
        survival_probabilities = survival_probability * 100

        # Returning computed survival function, probabilities, and risk category
        return {
            "survival_function": survival_function[0].y.tolist(),
            "survival_time": survival_time.tolist(),
            "survival_probabilities": survival_probabilities.tolist(),
            "risk_category": risk_category,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
@app.post("/compute-list-coxph-survival-profile/")
def compute_list_coxph_survival_profile(train_list: TrainList):
    try:
        # Defining expected column names for the input data
        column_names = ["AGE", "ANAEMIA", "EJECTION_FRACTION", "HIGH_BLOOD_PRESSURE", "SERUM_CREATININE", "SERUM_SODIUM"]

        # Converting input data to a pandas DataFrame with specified column names
        X_train_list = pd.DataFrame(train_list.features_list, columns=column_names)

        # Computing the survival function for all cases in the batch
        survival_function = final_survival_prediction_model.predict_survival_function(X_train_list)

        # Extracting survival probability profiles from the survival function output
        survival_profiles = [sf.y.tolist() for sf in survival_function]

        # Returning the computed survival profiles in JSON format
        return {"survival_profiles": survival_profiles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
@app.post("/bin-numeric-model-feature/")
def bin_numeric_model_feature(request: BinningRequest):
    try:
        # Converting the data input to a pandas DataFrame with appropriate column names
        X_original_list = pd.DataFrame(request.X_original_list)

        # Retrieving the precomputed median value for the selected numeric feature
        median = numeric_feature_median.loc[request.numeric_feature]

        # Dichotomizing the data input to categories based on the median
        X_original_list[request.numeric_feature] = np.where(
            X_original_list[request.numeric_feature] <= median, "Low", "High"
        )

        # Return the updated data as a list of dictionaries (JSON format)
        return X_original_list.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# plotting the estimated survival profiles
# using Kaplan-Meier Plots
##################################
@app.post("/plot-kaplan-meier/")
def plot_kaplan_meier(request: KaplanMeierRequest):
    try:
        # Converting input data to a pandas DataFrame
        df = pd.DataFrame(request.df)
        cat_var = request.cat_var
        new_case_value = request.new_case_value

        # Initializing a Kaplan-Meier estimator
        kmf = KaplanMeierFitter()

        # Creating the Kaplan-Meier plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Defining category labels and colors based on variable type
        if cat_var in ['AGE', 'EJECTION_FRACTION', 'SERUM_CREATININE', 'SERUM_SODIUM']:
            categories = ['Low', 'High']
            colors = {'Low': 'blue', 'High': 'red'}
        else:
            categories = ['Absent', 'Present']
            colors = {'Absent': 'blue', 'Present': 'red'}

        # Plotting Kaplan-Meier survival curves for each category    
        for value in categories:
            mask = df[cat_var] == value
            kmf.fit(df['TIME'][mask], event_observed=df['DEATH_EVENT'][mask], label=f'{cat_var}={value} (Baseline Distribution)')
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[value], linestyle='-', linewidth=6.0, alpha=0.30)

        # Overlaying the survival curve for the test case (if provided)
        if new_case_value is not None:
            mask_new_case = df[cat_var] == new_case_value
            kmf.fit(df['TIME'][mask_new_case], event_observed=df['DEATH_EVENT'][mask_new_case], label=f'{cat_var}={new_case_value} (Test Case)')
            kmf.plot_survival_function(ax=ax, ci_show=False, color='black', linestyle=':', linewidth=3.0)

        # Customizing plot labels and legend
        ax.set_title(f'DEATH_EVENT Survival Probabilities by {cat_var} Categories')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('TIME')
        ax.set_ylabel('DEATH_EVENT Survival Probability')
        ax.legend(loc='lower left')

        # Saving the plot to a buffer for encoding
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Closing the plot to release resources
        plt.close(fig)

        # Returning the base64-encoded image
        return {"plot": base64_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a utility function for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
def bin_numeric_feature(df, numeric_feature):
    # Retrieving the precomputed median value for the given feature
    median = numeric_feature_median.loc[numeric_feature]

    # Assigning category labels based on whether the value is below or above the median
    df[numeric_feature] = np.where(df[numeric_feature] <= median, "Low", "High")
    return df

##################################
# Defining a POST endpoint for
# preprocessing an individual test case
##################################
@app.post("/preprocess-test-case/")
def preprocess_test_case(test_case: TestCaseRequest):
    try:
        # Converting input test case into a pandas DataFrame
        X_test_sample = pd.DataFrame([test_case.dict()])

        # Applying Yeo-Johnson transformation to normalize numerical features
        X_test_sample_transformed = coxph_pipeline.named_steps['yeo_johnson'].transform(X_test_sample)

        # Converting transformed array back to a DataFrame with appropriate column names
        X_test_sample_converted = pd.DataFrame([X_test_sample_transformed[0]],
                                               columns=["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM", "ANAEMIA", "HIGH_BLOOD_PRESSURE"])

        # Binning continuous numerical variables into categorical groups (e.g., 'Low' vs 'High')
        for col in ["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM"]:
            X_test_sample_converted = bin_numeric_feature(X_test_sample_converted, col)

        # Converting binary categorical variables into 'Absent' or 'Present' labels    
        for col in ["ANAEMIA", "HIGH_BLOOD_PRESSURE"]:
            X_test_sample_converted[col] = X_test_sample_converted[col].apply(lambda x: 'Absent' if x < 1 else 'Present')

        # Returning the preprocessed test case as a dictionary
        return X_test_sample_converted.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a utility function for
# plotting the Kaplan-Meier Plots
# of a list of train cases
##################################
def plot_kaplan_meier_profile(df, cat_var, ax, new_case_value=None):

    # Initializing a Kaplan-Meier estimator
    kmf = KaplanMeierFitter()

    # Defining categories and corresponding colors
    if cat_var in ['AGE', 'EJECTION_FRACTION', 'SERUM_CREATININE', 'SERUM_SODIUM']:
        categories = ['Low', 'High']
        colors = {'Low': 'blue', 'High': 'red'}
    else:
        categories = ['Absent', 'Present']
        colors = {'Absent': 'blue', 'Present': 'red'}
    
    # Plotting Kaplan-Meier survival curves for baseline categories
    for value in categories:
        mask = df[cat_var] == value
        if mask.sum() > 0:  # Ensure there are data points to plot
            kmf.fit(df['TIME'][mask], event_observed=df['DEATH_EVENT'][mask], label=f'{cat_var}={value} (Baseline)')
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[value], linestyle='-', linewidth=6.0, alpha=0.30)
    
    # Overlaying test case survival function if it matches an existing category
    if new_case_value is not None and (df[cat_var] == new_case_value).any():
        mask_new_case = df[cat_var] == new_case_value
        kmf.fit(df['TIME'][mask_new_case], event_observed=df['DEATH_EVENT'][mask_new_case], label=f'{cat_var}={new_case_value} (Test Case)')
        kmf.plot_survival_function(ax=ax, ci_show=False, color='black', linestyle=':', linewidth=3.0)

##################################
# Defining a POST endpoint for
# plotting the estimated survival profiles
# using a Kaplan-Meier Plot grid
# for an individual test case
# against the training data as baseline
##################################
@app.post("/plot-kaplan-meier-grid/")
def plot_kaplan_meier_grid(test_case: TestCaseRequest):
    try:
        # Converting input test case into a pandas DataFrame
        X_test_sample = pd.DataFrame([test_case.dict()])

        # Applying Yeo-Johnson transformation to normalize numerical features
        X_test_sample_transformed = coxph_pipeline.named_steps['yeo_johnson'].transform(X_test_sample)

        # Converting transformed array back to a DataFrame with appropriate column names
        X_test_sample_converted = pd.DataFrame([X_test_sample_transformed[0]],
                                               columns=["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM", "ANAEMIA", "HIGH_BLOOD_PRESSURE"])

        # Binning continuous numerical variables into categorical groups (e.g., 'Low' vs 'High')
        for col in ["AGE", "EJECTION_FRACTION", "SERUM_CREATININE", "SERUM_SODIUM"]:
            X_test_sample_converted = bin_numeric_feature(X_test_sample_converted, col)

        # Converting binary categorical variables into 'Absent' or 'Present' labels      
        for col in ["ANAEMIA", "HIGH_BLOOD_PRESSURE"]:
            X_test_sample_converted[col] = X_test_sample_converted[col].apply(lambda x: 'Absent' if x < 1 else 'Present')

        # Creating a 2x3 plot grid for Kaplan-Meier survival analysis
        fig, axes = plt.subplots(3, 2, figsize=(17, 13))
        heart_failure_predictors = ['AGE', 'EJECTION_FRACTION', 'SERUM_CREATININE', 'SERUM_SODIUM', 'ANAEMIA', 'HIGH_BLOOD_PRESSURE']

        # Extracting training data subset excluding unused predictors
        X_train_indices = X_train.index.tolist()
        x_original_MI = x_original_EDA.copy()
        x_original_MI = x_original_MI.drop(['DIABETES','SEX', 'SMOKING','CREATININE_PHOSPHOKINASE','PLATELETS'], axis=1)
        x_train_MI = x_original_MI.loc[X_train_indices]

        # Binning continuous numeric features in training data for visualization
        for numeric_column in ["AGE","EJECTION_FRACTION","SERUM_CREATININE","SERUM_SODIUM"]:
            x_train_MI_EDA = bin_numeric_feature(x_train_MI, numeric_column)

        # Generating Kaplan-Meier plots for each predictor
        for i, predictor in enumerate(heart_failure_predictors):
            ax = axes[i // 2, i % 2]
            plot_kaplan_meier_profile(x_train_MI_EDA, predictor, ax, new_case_value=X_test_sample_converted[predictor][0])
            ax.set_title(f'Baseline Survival Probability by {predictor} Categories')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('TIME')
            ax.set_ylabel('Estimated Survival Probability')
            ax.legend(loc='lower left')

        # Adjusting layout and save the plot as a base64-encoded image
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Closing the plot to release resources
        plt.close(fig)

        # Returning the base64-encoded image
        return {"plot": base64_image}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# plotting the estimated survival profiles
# using a Kaplan-Meier Plot grid
# for an individual test case
# against the training data as baseline
##################################
@app.post("/plot-coxph-survival-profile/")
def plot_coxph_survival_profile(test_case: TestCaseRequest):
    try:
        # Getting the survival functions for training data cases
        X_train_survival_function = final_survival_prediction_model.predict_survival_function(X_train) 

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_test_sample = pd.DataFrame([test_case.dict()])

        # Computing the survival function for the input test case
        X_test_survival_function = final_survival_prediction_model.predict_survival_function(X_test_sample)

        # Determining the risk category based on model prediction and predefined risk threshold
        X_test_risk_category = (
            "High-Risk"
            if (final_survival_prediction_model.predict(X_test_sample)[0] > final_survival_prediction_model_risk_group_threshold)
            else "Low-Risk"
        )

        # Defining time points for survival probability estimation
        X_test_survival_time = np.array([50, 100, 150, 200, 250])

        # Estimating survival probabilities for the test case at defined time points
        X_test_survival_probability = np.interp(
            X_test_survival_time, X_test_survival_function[0].x, X_test_survival_function[0].y
        )
        X_test_survival_probabilities = X_test_survival_probability * 100

        # Computing the estimated survival probabilities for the test case at five defined time points and determining the risk category 
        X_test_sample_survival_function = X_test_survival_function
        X_test_sample_prediction_50 = X_test_survival_probabilities[0]
        X_test_sample_prediction_100 = X_test_survival_probabilities[1]
        X_test_sample_prediction_150 = X_test_survival_probabilities[2]
        X_test_sample_prediction_200 = X_test_survival_probabilities[3]
        X_test_sample_prediction_250 = X_test_survival_probabilities[4]
        X_test_sample_risk_category = X_test_risk_category
        X_test_sample_survival_time = X_test_survival_time
        X_test_sample_survival_probability = X_test_survival_probability

        # Resetting the index for plotting survival functions for the training data
        y_train_reset_index = y_train.reset_index()

        # Creating a 1x1 plot for survival function visualization
        fig, ax = plt.subplots(figsize=(17, 8))

        # Plotting survival curves for training data (color-coded by death event status)
        for i, surv_func in enumerate(X_train_survival_function):
            ax.step(surv_func.x, 
                    surv_func.y, 
                    where="post", 
                    color='red' if y_train_reset_index['DEATH_EVENT'][i] == 1 else 'blue', 
                    linewidth=6.0,
                    alpha=0.05)

        # Defining color and style for low-risk category    
        if X_test_sample_risk_category == "Low-Risk":
            ax.step(X_test_sample_survival_function[0].x, 
                    X_test_sample_survival_function[0].y, 
                    where="post", 
                    color='blue',
                    linewidth=6.0,
                    linestyle='-',
                    alpha=0.30,
                    label='Test Case (Low-Risk)')
            ax.step(X_test_sample_survival_function[0].x, 
                    X_test_sample_survival_function[0].y, 
                    where="post", 
                    color='black',
                    linewidth=3.0,
                    linestyle=':',
                    label='Test Case (Low-Risk)')

            # Adding vertical lines at defined time points to show survival probability estimates
            for survival_time, survival_probability in zip(X_test_sample_survival_time, X_test_sample_survival_probability):
                ax.vlines(x=survival_time, ymin=0, ymax=survival_probability, color='blue', linestyle='-', linewidth=2.0, alpha=0.30)

            # Defining legend elements for training data and test case    
            red_patch = plt.Line2D([0], [0], color='red', lw=6, alpha=0.30,  label='Death Event Status = True')
            blue_patch = plt.Line2D([0], [0], color='blue', lw=6, alpha=0.30, label='Death Event Status = False')
            black_patch = plt.Line2D([0], [0], color='black', lw=3, linestyle=":", label='Test Case (Low-Risk)')
        
        # Defining color and style for high-risk category 
        if X_test_sample_risk_category == "High-Risk":
            ax.step(X_test_sample_survival_function[0].x, 
                    X_test_sample_survival_function[0].y, 
                    where="post", 
                    color='red',
                    linewidth=6.0,
                    linestyle='-',
                    alpha=0.30,
                    label='Test Case (High-Risk)')
            ax.step(X_test_sample_survival_function[0].x, 
                    X_test_sample_survival_function[0].y, 
                    where="post", 
                    color='black',
                    linewidth=3.0,
                    linestyle=':',
                    label='Test Case (High-Risk)')

            # Adding vertical lines at defined time points to show survival probability estimates
            for survival_time, survival_probability in zip(X_test_sample_survival_time, X_test_sample_survival_probability):
                ax.vlines(x=survival_time, ymin=0, ymax=survival_probability, color='red', linestyle='-', linewidth=2.0, alpha=0.30)
            
            # Defining legend elements for training data and test case
            red_patch = plt.Line2D([0], [0], color='red', lw=6, alpha=0.30,  label='Death Event Status = True')
            blue_patch = plt.Line2D([0], [0], color='blue', lw=6, alpha=0.30, label='Death Event Status = False')
            black_patch = plt.Line2D([0], [0], color='black', lw=3, linestyle=":", label='Test Case (High-Risk)')
        
        # Setting plot title and labels
        ax.legend(handles=[red_patch, blue_patch, black_patch], facecolor='white', framealpha=1, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
        ax.set_title('Final Survival Prediction Model: Cox Proportional Hazards Regression')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Estimated Survival Probability')

        # Adjusting layout and save the plot as a base64-encoded image
        plt.tight_layout(rect=[0, 0, 1.00, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Closing the plot to release resources
        plt.close(fig)

        # Returning the base64-encoded image
        return {"plot": base64_image}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Running the FastAPI app
##################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)   
    