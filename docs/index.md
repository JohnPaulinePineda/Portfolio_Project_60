***
# Model Deployment : Containerizing and Deploying Machine Learning API Endpoints on Open-Source Platforms

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *April 5, 2025*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Model Development](#1.1)
        * [1.1.1 Project Background](#1.1.1)
        * [1.1.2 Data Background](#1.1.1)
        * [1.1.3 Model Building](#1.1.2)
        * [1.1.4 Model Inference](#1.1.3)
    * [1.2 Application Programming Interface (API) Development](#1.2)
        * [1.2.1 API Building](#1.2.1)
        * [1.2.2 API Testing](#1.2.2)
    * [1.3 Application Programming Interface (API) Containerization](#1.3)
        * [1.3.1 Docker File Creation](#1.3.1)
        * [1.3.2 Docker Image Building](#1.3.2)
        * [1.3.3 Docker Image Testing](#1.3.3)
        * [1.3.4 Docker Image Storage](#1.3.4)
    * [1.4 Application Programming Interface (API) Deployment](#1.4)
        * [1.4.1 Docker Image Execution and Hosting](#1.4.1)
    * [1.5 User Interface (UI) Development](#1.5)
        * [1.5.1 UI Building With API Calls](#1.5.1)
    * [1.6 Web Application Deployment](#1.6)
        * [1.6.1 UI Hosting](#1.6.1)
        * [1.6.1 Applicaton Testing](#1.6.2)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores **open-source solutions for containerizing and deploying machine learning API endpoints**, focusing on the implementation of a heart failure survival prediction model as a web application in <mark style="background-color: #CCECFF"><b>Python</b></mark>. The objective was to operationalize a **Cox Proportional Hazards Regression** survival model by deploying an interactive UI that enables users to input cardiovascular, hematologic, and metabolic markers and receive survival probability estimates at different time points. The project workflow involved multiple stages: first, a RESTful API was developed using the **FastAPI** framework to serve the survival prediction model. The API was tested locally to ensure correct response formatting and model inference behavior. Next, the application was containerized using **Docker**, enabling a reproducible and portable environment. The Docker image was built, tested, and pushed to **DockerHub** for persistent storage before being deployed on **Render**, an open-source cloud platform for hosting containerized applications. To enable user interaction, a web-based interface was developed using **Streamlit**. The UI facilitated data input via range sliders and radio buttons, processed user entries, and sent requests to the **FastAPI** backend for prediction and visualization. The **Streamlit** app was then deployed on **Render** to ensure full integration with the containerized API. End-to-end testing verified the functionality of the deployed application, confirming that API endpoints, model predictions, and UI elements worked seamlessly together. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document.

[Machine Learning Model Deployment](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/), also known as model operationalization, is the process of integrating a trained model into a production environment where it can generate real-world predictions. This involves packaging the model, along with its dependencies, into a scalable and reliable system that can handle user requests and return predictions in real time. Deployment strategies can range from embedding models into web applications via RESTful APIs to deploying them in containerized environments using Docker and Kubernetes for scalability.

Operationalization goes beyond deployment by ensuring continuous monitoring, retraining, and version control to maintain model performance over time. It involves addressing challenges like data drift, latency, and security while optimizing infrastructure for efficiency. By automating model serving, updating, and logging, machine learning operationalization bridges the gap between development and real-world application, enabling AI-driven decision-making in various domains, from healthcare and finance to manufacturing and e-commerce.

[RESTful APIs](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) provide a standardized way to expose machine learning models as web services, allowing clients to interact with them over HTTP using common methods such as GET, POST, PUT, and DELETE. RESTful APIs enable seamless communication between different applications and systems, making it easy to integrate predictive models into web and mobile applications. They ensure scalability by decoupling the frontend from the backend, enabling multiple users to send requests simultaneously. In machine learning deployment, RESTful APIs facilitate real-time inference by accepting user input, processing data, invoking model predictions, and returning responses in a structured format such as JSON. Their stateless nature ensures reliability and consistency, while features like authentication, caching, and error handling improve security and performance.

[FastAPI](https://fastapi.tiangolo.com/) is a modern, high-performance Python framework designed for building RESTful APIs efficiently. It is optimized for speed, leveraging asynchronous programming with Python’s `async` and `await` capabilities, which makes it significantly faster than traditional frameworks like Flask. FastAPI simplifies API development by providing automatic data validation using Pydantic and interactive documentation through OpenAPI and ReDoc. Its support for JSON serialization, request validation, and dependency injection makes it an ideal choice for deploying machine learning models as APIs. In a deployment pipeline, FastAPI ensures low-latency inference, efficient request handling, and seamless integration with frontend applications. Its built-in support for WebSockets, background tasks, and OAuth authentication further enhances its capabilities for building scalable and production-ready AI-powered applications.

[Streamlit](https://streamlit.io/) is an open-source Python framework that simplifies the development of interactive web applications for machine learning models and data science projects. Unlike traditional web development tools, Streamlit requires minimal coding, allowing users to create dynamic interfaces with just a few lines of Python. It provides built-in widgets such as sliders, buttons, and file uploaders that enable real-time user interaction with machine learning models. Streamlit automatically refreshes the UI upon any change in input, ensuring a seamless and intuitive experience. Its ability to display charts, tables, and even base64-encoded images makes it an excellent tool for visualizing predictions, statistical insights, and model outputs. As a lightweight and easy-to-deploy solution, Streamlit is widely used for prototyping, model evaluation, and sharing AI-powered applications with non-technical users.

[Docker](https://www.docker.com/) is a powerful containerization platform that enables developers to package applications and their dependencies into isolated environments called containers. This ensures consistency across different computing environments, eliminating compatibility issues between libraries, operating systems, and software versions. For machine learning deployment, Docker allows models, APIs, and dependencies to be bundled together in a single image, ensuring reproducibility and ease of deployment. Containers are lightweight, fast, and scalable, making them ideal for cloud-based inference services. Docker also simplifies version control and resource management, allowing developers to create portable applications that can run on any system with Docker installed. Its integration with orchestration tools like Kubernetes further enhances scalability, enabling automated deployment, load balancing, and fault tolerance in production environments.

[DockerHub](https://hub.docker.com/) is a cloud-based container registry that serves as a centralized repository for storing, managing, and distributing Docker images. It allows developers to push and pull containerized applications, enabling seamless collaboration and version control across teams. In machine learning deployment, DockerHub ensures that API images, model files, and dependencies remain accessible for deployment on different cloud platforms. It supports both public and private repositories, allowing for controlled access and secure storage of containerized applications. With built-in automated builds and continuous integration capabilities, DockerHub facilitates streamlined development workflows, enabling frequent updates and quick rollbacks. By hosting pre-built images, it simplifies the deployment process, allowing developers to deploy models directly from the registry to platforms like Render, AWS, or Kubernetes clusters.

[Render](https://render.com/) is a cloud-based hosting platform that simplifies the deployment of web applications, APIs, and containerized services. It offers a user-friendly interface, automatic scaling, and seamless integration with GitHub, enabling continuous deployment. For machine learning applications, Render provides an efficient way to host FastAPI-based RESTful APIs, ensuring high availability and low-latency model inference. It supports Docker containers, allowing developers to deploy pre-built images from DockerHub with minimal configuration. Render automatically handles infrastructure management, including load balancing, networking, and monitoring, reducing the operational overhead associated with deployment. With its free and paid hosting tiers, it provides a cost-effective solution for both prototyping and production deployment of AI-powered applications, making it a preferred choice for data scientists and developers.



## 1.1. Model Development <a class="anchor" id="1.1"></a>

### 1.1.1 Project Background <a class="anchor" id="1.1.1"></a>

This project implements the **Cox Proportional Hazards Regression**, **Cox Net Survival**, **Survival Tree**, **Random Survival Forest**, and **Gradient Boosted Survival** models as independent base learners using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to estimate the survival probabilities of right-censored survival time and status responses. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power using the **Harrel's Concordance Index** metric. Penalties including **Ridge Regularization** and **Elastic Net Regularization** were evaluated to impose constraints on the model coefficient updates, as applicable. Additionally, survival probability functions were estimated for model risk-groups and sampled individual cases. 

* The complete model development process was consolidated in this [**Jupyter Notebook**](https://johnpaulinepineda.github.io/Portfolio_Project_55/).
* All associated datasets and code files were stored in this [**GitHub Project Repository**](https://github.com/JohnPaulinePineda/Portfolio_Project_55). 
* The final model was deployed as a prototype application with a web interface via [**Streamlit**](https://heart-failure-survival-probability-estimation.streamlit.app/).
  

### 1.1.2 Data Background <a class="anchor" id="1.1.2"></a>

1. The original dataset comprised rows representing observations and columns representing variables.
2. The target variables contain both numeric and dichotomous categorical data types:
    * <span style="color: #FF0000">DEATH_EVENT</span> (Categorical: 0, Censored | 1, Death)
    * <span style="color: #FF0000">TIME</span> (Numeric: Days)
3. The complete set of 11 predictor variables contain both numeric and categorical data types:   
    * <span style="color: #FF0000">AGE</span> (Numeric: Years)
    * <span style="color: #FF0000">ANAEMIA</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span> (Numeric: Percent)
    * <span style="color: #FF0000">DIABETES</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">EJECTION_FRACTION</span> (Numeric: Percent)
    * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">PLATELETS</span> (Numeric: kiloplatelets/mL)
    * <span style="color: #FF0000">SERUM_CREATININE</span> (Numeric: mg/dL)
    * <span style="color: #FF0000">SERUM_SODIUM</span> (Numeric: mEq/L)
    * <span style="color: #FF0000">SEX</span> (Categorical: 0, Female | 1, Male)
    * <span style="color: #FF0000">SMOKING</span> (Categorical: 0, Absent | 1 Present)
4. Exploratory data analysis identified a subset of 6 predictor variables that was significantly associated with the target variables and subsequently used as the final model predictors:   
    * <span style="color: #FF0000">AGE</span> (Numeric: Years)
    * <span style="color: #FF0000">ANAEMIA</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">EJECTION_FRACTION</span> (Numeric: Percent)
    * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">SERUM_CREATININE</span> (Numeric: mg/dL)
    * <span style="color: #FF0000">SERUM_SODIUM</span> (Numeric: mEq/L)


![sp_data_background.png](bcd263b2-3c48-415d-bdd6-17c6d4da7def.png)

### 1.1.3 Model Building <a class="anchor" id="1.1.3"></a>

1. The model development process involved evaluating different **Model Structures**. Hyperparameter tuning was conducted using the 5-fold cross-validation method with optimal model performance determined using the **Harrel's concordance index**.
    * [Cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) developed from the original data.
    * [Cox net survival model ](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) developed from the original data.
    * [Survival tree model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.tree.SurvivalTree.html) developed from the original data.
    * [Random survival forest model,](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html) developed from the original data.
    * [Gradient boosted survival model ](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html) developed from the original data.
2. The [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) developed from the original data was selected as the final model by demonstrating the most stable **Harrel's concordance index** across the different internal and external validation sets. 
3. The final model configuration for the [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) is described as follows:
    * <span style="color: #FF0000">alpha</span> = 10


![sp_model_background.png](8903ea73-f8da-4c59-b2ea-c8014c198874.png)

### 1.1.4 Model Inference <a class="anchor" id="1.1.4"></a>

1. The prediction model was deployed using a web application hosted at [<mark style="background-color: #CCECFF"><b>Streamlit</b></mark>](https://heart-failure-survival-probability-estimation.streamlit.app/).
2. The user interface input consists of the following:
    * range sliders to enable numerical input to measure the characteristics of the test case for certain cardiovascular, hematologic and metabolic markers:
        * <span style="color: #FF0000">AGE</span>
        * <span style="color: #FF0000">EJECTION_FRACTION</span>
        * <span style="color: #FF0000">SERUM_CREATININE</span>
        * <span style="color: #FF0000">SERUM_SODIUM</span>
    * radio buttons to enable binary category selection (Present | Absent) to identify the status of the test case for certain hematologic and cardiovascular markers:
        * <span style="color: #FF0000">ANAEMIA</span>
        * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>
    * action button to:
        * process study population data as baseline
        * process user input as test case
        * render all entries into visualization charts
        * execute all computations, estimations and predictions
        * render test case prediction into the survival probability plot
3. The user interface ouput consists of the following:
    * Kaplan-Meier plots to:
        * provide a baseline visualization of the survival profiles of the various feature categories (Yes | No or High | Low) estimated from the study population given the survival time and event status
        * indicate the entries made from the user input to visually assess the survival probabilities of the test case characteristics against the study population across all time points
    * survival probability plot to:
        * provide a visualization of the baseline survival probability profile using each observation of the study population given the survival time and event status
        * indicate the heart failure survival probabilities of the test case at different time points
    * summary table to:
        * present the estimated heart failure survival probabilities and predicted risk category for the test case


![sp_deployment_background.png](b2b8ec2c-e65d-4286-9fdf-79236887bf7b.png)

## 1.2. Application Programming Interface (API) Development <a class="anchor" id="1.2"></a>

### 1.2.1 API Building <a class="anchor" id="1.2.1"></a>

1. An API code using the FastAPI framework was developed for deploying a survival prediction model with the steps described as follows:
    * **Loading Python Libraries**
        * Imported necessary libraries such as `FastAPI`, `HTTPException`, and `BaseModel` for API development.
        * Included libraries for survival analysis (`sksurv`, `lifelines`), data manipulation (`numpy`, `pandas`), and visualization (`matplotlib`).
        * Used `io` and `base64` for encoding and handling image outputs.
    * **Defining File Paths**
        * Specified the `MODELS_PATH` and `PARAMETERS_PATH` to locate the pre-trained survival model and related parameters.
        * Specified the `DATASETS_PATH` and `PIPELINES_PATH` to locate the data sets and preprocessing pipelines.
    * **Loading the Training Data**
        * Loaded the raw and preprocessed training data (X_train.csv, y_train and heart_failure_EDA.csv) using pd.read_csv.
    * **Loading the Preprocessing Pipeline**
        * Loaded the preprocessing pipeline (coxph_pipeline.pkl) involving a Yeo-Johnson transformer for numeric predictors using joblib.load.
    * **Loading the Pre-Trained Survival Model**    
        * Loaded the pre-trained Cox Proportional Hazards (CoxPH) model (coxph_best_model.pkl) using joblib.load.
        * Handled potential errors during model loading with a try-except block.
    * **Loading Model Parameters**
        * Loaded the median values for numeric features (numeric_feature_median_list.pkl) to support feature binning.
        * Loaded the risk group threshold (coxph_best_model_risk_group_threshold.pkl) for categorizing patients into "High-Risk" and "Low-Risk" groups.
    * **Defining Input Schemas**
        * Created a Pydantic BaseModel class to define input schema for TestCaseRequest: For individual test cases, expecting a dictionary of floats and integers as input features.
        * Created a Pydantic BaseModel class to define input schema for TestSample: For individual test cases, expecting a list of floats as input features.
        * Created a Pydantic BaseModel class to define input schema for TrainList: For batch processing, expecting a list of lists of floats as input features.
        * Created a Pydantic BaseModel class to define input schema for BinningRequest: For dichotomizing numeric features based on the median.
        * Created a Pydantic BaseModel class to define input schema for KaplanMeierRequest: For generating Kaplan-Meier survival plots.
    * **Initializing the FastAPI App**
        * Created a FastAPI instance (app) to define and serve API endpoints.
    * **Defining API Endpoints**
        * Root Endpoint (`/`): A simple GET endpoint to validate API service connectivity.
        * Individual Survival Prediction Endpoint (`/compute-individual-coxph-survival-probability-class/` and `/compute-test-coxph-survival-probability-class/`): POST endpoints to generate survival profiles, estimate survival probabilities, and predict risk categories for individual test cases with varying Pydantic BaseModel classes.
        * Batch Survival Prediction Endpoint (`/compute-list-coxph-survival-profile/`): A POST endpoint to generate survival profiles for a batch of cases.
        * Feature Binning Endpoint (`/bin-numeric-model-feature/`): A POST endpoint to dichotomize numeric features based on the median for a defined predictor.
        * Kaplan-Meier Plot Endpoint (`/plot-kaplan-meier/`): A POST endpoint to generate and return Kaplan-Meier survival plots for a single defined predictor.
        * Test Case Preprocessing Endpoint (`/preprocess-test-case/`): A POST endpoint to perform preprocessing for individual test cases.
        * Kaplan-Meier Plot Grid Endpoint (`/plot-kaplan-meier-grid/`): A POST endpoint to generate and return Kaplan-Meier survival plots for all predictors.
        * Cox Survival Plot Endpoint (`/plot_coxph_survival_profile/`): A POST endpoint to generate and return Cox survival plots.
    * **Defining Utility Functions**
        * Numeric Feature Binning Function (`bin_numeric_feature`): A utility function to dichotomize numeric features based on the median for any predictor.
        * Kaplan-Meier Plot Profile Function (`plot_kaplan_meier_profile`): A utility function to generate and return Kaplan-Meier survival plots for any single predictor.
    * **Individual Survival Prediction Logic**
        * Converted the input data into a pandas DataFrame with appropriate feature names.
        * Used the pre-trained model’s predict_survival_function to generate the survival function for the test case.
        * Predicted the risk category ("High-Risk" or "Low-Risk") based on the model’s risk score and threshold.
        * Interpolated survival probabilities at predefined time points (e.g., 50, 100, 150, 200, 250 days).
    * **Batch Survival Profile Logic**
        * Converted the batch input data into a pandas DataFrame with appropriate feature names.
        * Used the pre-trained model’s predict_survival_function to generate survival functions for all cases in the batch.
        * Extracted and returned survival profiles for each case.
    * **Feature Binning Logic**
        * Converted the input data into a pandas DataFrame.
        * Dichotomized the specified numeric feature into "Low" and "High" categories based on the median value.
        * Returned the binned data as a list of dictionaries.
    * **Kaplan-Meier Plot Logic**
        * Converted the input data into a pandas DataFrame.
        * Initialized a KaplanMeierFitter object to estimate survival probabilities.
        * Plotted survival curves for different categories of the specified variable (e.g., "Low" vs. "High").
        * Included an optional new case value for comparison in the plot.
        * Saved the plot as a base64-encoded image and returned it in the API response.
    * **Test Case Preprocessing Logic**
        * Converted the input data into a pandas DataFrame.
        * Applied Yeo-Johnson transformation (from a pre-defined pipeline) to normalize numeric features.
        * Reconstructed the transformed DataFrame with appropriate feature names.
        * Called the bin_numeric_feature utility function to binarize numeric features by creating dichotomous bins.
        * Encoded categorical features as "Absent" or "Present" based on their values.
        * Returned the preprocessed test case as a dictionary in a format suitable for model inference.
    * **Kaplan-Meier Plot Grid Logic**
        * Preprocessed the test case to binarize numeric features and encode categorical features.
        * Created a 2x3 grid of plots (one per predictor).
        * Called the plot_kaplan_meier_profile utility function to generate baseline survival curves for each predictor.
        * Overlaid the Kaplan-Meier survival curve of the test case for direct comparison.
        * Encoded the final plot as a base64 string for easy transmission in API responses.
    * **Cox Survival Plot Logic**
        * Predicted survival functions for both the training dataset and the individual test case using the final Cox Proportional Hazards model.
        * Classified the test case as "High-Risk" or "Low-Risk" based on the model’s predicted risk score and predefined threshold.
        * Interpolated survival probabilities at predefined time points (50, 100, 150, 200, 250 days).
        * Overlaid the survival function for the test case, using different colors and line styles based on the predicted risk category:
          * Low-Risk: Blue (solid for baseline, dotted for overlay).
          * High-Risk: Red (solid for baseline, dotted for overlay).
        * Added vertical lines to indicate estimated survival probabilities at specific time points.
        * Saved the plot as a base64-encoded image and returned it in the API response.
    * **Error Handling**
        * Implemented robust error handling for invalid inputs or prediction errors using HTTPException.
        * Returned meaningful error messages and appropriate HTTP status codes (e.g., 500 for server errors).
    * **Running the FastAPI App**
        * Used uvicorn to run the FastAPI app on localhost at port 8000.
2. Key features of the API code included the following:
    * Supported both individual and batch predictions, making the API versatile for different use cases.
    * Provided survival probabilities, risk categories, and visualizations (Kaplan-Meier and Cox Survival plots) for interpretable results.
    * Enabled feature preprocessing to transform the test case in a format suitable for model inference


![sp_fastapi_code.png](630f8115-a380-4450-b7c6-4b7bce5d47ac.png)

### 1.2.2 API Testing <a class="anchor" id="1.2.2"></a>

1. The API code developed using the FastAPI framework deploying a survival prediction model was successfully tested with results presented as follows:
    * **Server Initialization**: FastAPI application was started successfully, with Uvicorn running on `http://127.0.0.1:8000`, indicating that the server and its documentation are active and ready to process requests.
    * **Hot Reloading Activated**: Uvicorn's reloader process (WatchFiles) was initialized, allowing real-time code changes without restarting the server.
    * **Server Process Started**: The primary server process was assigned a process ID (25028), confirming successful application launch.
    * **Application Ready State**: The server was shown to wait for incoming requests, ensuring all necessary components, including model loading, are successfully initialized.
    * **Root Endpoint Accessed (GET /)**: The API received a GET request at the root endpoint and responded with 200 OK, confirming that the service is running and accessible.
    * **Individual Survival Probability Request (POST /compute-individual-coxph-survival-probability-class/)**: A POST request was processed successfully, returning 200 OK, indicating that the API correctly computed survival probabilities and risk categorization for an individual test case.
    * **Batch Survival Profile Request (POST /compute-list-coxph-survival-profile/)**: The API successfully processed a POST request for batch survival profile computation, returning 200 OK, confirming that multiple test cases were handled correctly.
    * **Feature Binning Request (POST /bin-numeric-model-feature/)**: A POST request was successfully executed, returning 200 OK, confirming that the API correctly categorized numeric model features into dichotomous bins.
    * **Kaplan-Meier Plot Request (POST /plot-kaplan-meier/)**: The API successfully processed a POST request, returning 200 OK, indicating that a Kaplan-Meier survival plot was generated and returned as a base64-encoded image.
    * **Test Case Preprocessing Request (POST /preprocess-test-case/)**: The API successfully processed a POST request, returning 200 OK, indicating that an individual test cases was completed preprocessing and transformed to a format suitable for model inference.
    * **Kaplan-Meier Plot Grid Request (POST /plot-kaplan-meier-grid/)**: The API successfully processed a POST request, returning 200 OK, indicating that a Kaplan-Meier survival plot for the individual the test case in direct comparison with the baseline survival curves for each predictor was generated and returned as a base64-encoded image.
    * **Cox Survival Plot Request (POST /plot_coxph_survival_profile/)**: The API successfully processed a POST request, returning 200 OK, indicating that a Cox survival plot for the individual the test case in direct comparison with the baseline survival curves for all the training cases categorized by risk categories was generated and returned as a base64-encoded image.


![sp_fastapi_activation.png](7c2c24f8-ba6f-4a39-9fad-8289e588900b.png)

![sp_fastapi_documentation_endpoints.png](6de7e96a-69db-49f1-a932-9d0d3a96f88b.png)

![sp_fastapi_documentation_schemas.png](563f92ce-5815-4391-b6c3-d3a5a0284c73.png)

![sp_fastapi_endpoints.png](2b2a6886-33e4-4eb0-9132-d71ea8380a15.png)


```python
##################################
# Loading Python Libraries
##################################
import requests
import json
import pandas as pd
import base64
from IPython.display import display
from PIL import Image

```


```python
##################################
# Defining the base URL of the API
# for the survival prediction model
##################################
SP_FASTAPI_BASE_URL = "http://127.0.0.1:8000"

```


```python
##################################
# Defining the input values for an individual test case
# as a list
##################################
single_test_case = {
    "features_individual": [43, 0, 75, 1, 0.75, 100]  
}

```


```python
##################################
# Defining the input values for a batch of cases
# as a list of lists
##################################
train_list = {
        "features_list": [
            [43, 0, 75, 1, 0.75, 100],
            [70, 1,	20,	1, 0.75, 100]
        ]
    }

```


```python
##################################
# Defining the input values for a batch of cases for binning request
# as a list of dictionaries and a string
##################################
bin_request = {
        "X_original_list": [
            {"AGE": -0.10, "EJECTION_FRACTION": -0.10, "SERUM_CREATININE ": -0.10, "SERUM_SODIUM": -0.10},
            {"AGE": 0.20, "EJECTION_FRACTION": 0.20, "SERUM_CREATININE ": 0.20, "SERUM_SODIUM": 0.20},
            {"AGE": 0.90, "EJECTION_FRACTION": 0.90, "SERUM_CREATININE ": 0.90, "SERUM_SODIUM": 0.90}
        ],
        "numeric_feature": "AGE"
    }

```


```python
##################################
# Defining the input values for a batch of cases for Kaplan-Meier plotting
# as a list of dictionaries and multiple strings
##################################
km_request = {
        "df": [
            {"TIME": 0, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 25, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 50, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 100, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 125, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 150, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 175, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 200, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 225, "DEATH_EVENT": 1, "AGE": "Low"},
            {"TIME": 250, "DEATH_EVENT": 1, "AGE": "Low"},
            {"TIME": 0, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 25, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 50, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 100, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 125, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 150, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 175, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 200, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 225, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 250, "DEATH_EVENT": 1, "AGE": "High"},
        ],
        "cat_var": "AGE",
        "new_case_value": "Low"
    }

```


```python
##################################
# Defining the input values for an individual test case
# as a dictionary
##################################
test_case_request = {
    "AGE": 65,
    "EJECTION_FRACTION": 35,
    "SERUM_CREATININE": 1.2,
    "SERUM_SODIUM": 135,
    "ANAEMIA": 1,
    "HIGH_BLOOD_PRESSURE": 0
}

```


```python
##################################
# Generating a GET endpoint request for
# validating API service connection
##################################
response = requests.get(f"{SP_FASTAPI_BASE_URL}/")
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'message': 'Welcome to the Survival Prediction API!'}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-individual-coxph-survival-probability-class/", json=single_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'survival_function': [0.9973812917524568,
      0.9920416812438736,
      0.9893236791425079,
      0.972381113071464,
      0.9693179903073035,
      0.9631930672135339,
      0.9631930672135339,
      0.9600469571766689,
      0.9600469571766689,
      0.9568596864927983,
      0.9536305709158891,
      0.9471625843882805,
      0.93729581350105,
      0.9338986486591409,
      0.93048646553474,
      0.9270645831787163,
      0.9202445006124622,
      0.9167715111530355,
      0.9132845175345189,
      0.9097550958520674,
      0.9097550958520674,
      0.9097550958520674,
      0.9060810720432387,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.8985598696587259,
      0.8985598696587259,
      0.8985598696587259,
      0.8945287485160898,
      0.8945287485160898,
      0.8945287485160898,
      0.8945287485160898,
      0.8901959645503091,
      0.8812352215018253,
      0.8812352215018253,
      0.8812352215018253,
      0.8812352215018253,
      0.8764677174183527,
      0.8764677174183527,
      0.8764677174183527,
      0.8764677174183527,
      0.8709113650481243,
      0.8709113650481243,
      0.8652494086650531,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8459534502216807,
      0.8389821875092403,
      0.8319419786276306,
      0.8246669811915435,
      0.8099879066057215,
      0.8099879066057215,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7848178617845467,
      0.7848178617845467,
      0.7848178617845467,
      0.7848178617845467,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7337716342207724,
      0.7337716342207724,
      0.7337716342207724,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696],
     'survival_time': [50, 100, 150, 200, 250],
     'survival_probabilities': [90.97550958520674,
      87.64677174183527,
      84.59534502216806,
      78.48178617845467,
      70.70184115456696],
     'risk_category': 'Low-Risk'}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-list-coxph-survival-profile/", json=train_list)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'survival_profiles': [[0.9973812917524568,
       0.9920416812438736,
       0.9893236791425079,
       0.972381113071464,
       0.9693179903073035,
       0.9631930672135339,
       0.9631930672135339,
       0.9600469571766689,
       0.9600469571766689,
       0.9568596864927983,
       0.9536305709158891,
       0.9471625843882805,
       0.93729581350105,
       0.9338986486591409,
       0.93048646553474,
       0.9270645831787163,
       0.9202445006124622,
       0.9167715111530355,
       0.9132845175345189,
       0.9097550958520674,
       0.9097550958520674,
       0.9097550958520674,
       0.9060810720432387,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.8985598696587259,
       0.8985598696587259,
       0.8985598696587259,
       0.8945287485160898,
       0.8945287485160898,
       0.8945287485160898,
       0.8945287485160898,
       0.8901959645503091,
       0.8812352215018253,
       0.8812352215018253,
       0.8812352215018253,
       0.8812352215018253,
       0.8764677174183526,
       0.8764677174183526,
       0.8764677174183526,
       0.8764677174183526,
       0.8709113650481243,
       0.8709113650481243,
       0.8652494086650531,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8459534502216807,
       0.8389821875092403,
       0.8319419786276306,
       0.8246669811915435,
       0.8099879066057215,
       0.8099879066057215,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7848178617845467,
       0.7848178617845467,
       0.7848178617845467,
       0.7848178617845467,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7337716342207724,
       0.7337716342207724,
       0.7337716342207724,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695],
      [0.9761144218801228,
       0.928980888267716,
       0.905777064852962,
       0.7724242339590301,
       0.7502787164583535,
       0.7076872741961866,
       0.7076872741961866,
       0.6866593185026403,
       0.6866593185026403,
       0.6659260634219393,
       0.6454915885099762,
       0.6062342264686207,
       0.5504405490863784,
       0.5323184765768243,
       0.5146536440658629,
       0.49746533888245986,
       0.464726413395405,
       0.4488047347163327,
       0.433309930422515,
       0.4181140392975694,
       0.4181140392975694,
       0.4181140392975694,
       0.4028020116306455,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.373006008573048,
       0.373006008573048,
       0.373006008573048,
       0.35785929480690143,
       0.35785929480690143,
       0.35785929480690143,
       0.35785929480690143,
       0.3421927616040032,
       0.3117176431598899,
       0.3117176431598899,
       0.3117176431598899,
       0.3117176431598899,
       0.29651072362871467,
       0.29651072362871467,
       0.29651072362871467,
       0.29651072362871467,
       0.2796248763802668,
       0.2796248763802668,
       0.2633052706162029,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.213871875082776,
       0.19816204676152543,
       0.18334919195124752,
       0.16908728217620728,
       0.1432835564355214,
       0.1432835564355214,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.10710184631976834,
       0.10710184631976834,
       0.10710184631976834,
       0.10710184631976834,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.05761125190131533,
       0.05761125190131533,
       0.05761125190131533,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404]]}



```python
##################################
# Sending a POST endpoint request for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/bin-numeric-model-feature/", json=bin_request)
if response.status_code == 200:
    display("Response:", pd.DataFrame(response.json()))
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>High</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>High</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival profiles
# using Kaplan-Meier Plots
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/plot-kaplan-meier/", json=km_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("kaplan_meier_plot.png", "wb") as f:
        f.write(img)
        display(Image.open("kaplan_meier_plot.png"))
else:
    print("Error:", response.status_code, response.text)
    
```


    
![png](output_31_0.png)
    



```python
##################################
# Sending a POST endpoint request for
# preprocessing an individual test case
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/preprocess-test-case/", json=test_case_request)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    [{'AGE': 'High',
      'EJECTION_FRACTION': 'Low',
      'SERUM_CREATININE': 'High',
      'SERUM_SODIUM': 'Low',
      'ANAEMIA': 'Present',
      'HIGH_BLOOD_PRESSURE': 'Absent'}]



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival profiles
# using a Kaplan-Meier Plot Matrix
# for an individual test case
# against the training data as baseline
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/plot-kaplan-meier-grid/", json=test_case_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("kaplan_meier_plot_matrix.png", "wb") as f:
        f.write(img)
        display(Image.open("kaplan_meier_plot_matrix.png"))
else:
    print("Error:", response.status_code, response.text)
    
```


    
![png](output_33_0.png)
    



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-test-coxph-survival-probability-class/", json=test_case_request)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'survival_function': [0.9935852422336905,
      0.980581105741338,
      0.974000615497288,
      0.9335716291520674,
      0.9263704990495979,
      0.9120703170806183,
      0.9120703170806183,
      0.9047761239313578,
      0.9047761239313578,
      0.8974218629392444,
      0.8900072909706268,
      0.8752652317601862,
      0.8530570074516991,
      0.845488798281818,
      0.8379273268795359,
      0.8303847506918888,
      0.8154721703733717,
      0.8079397039318209,
      0.8004184982181605,
      0.7928481866122223,
      0.7928481866122223,
      0.7928481866122223,
      0.7850129580161529,
      0.7772421803304252,
      0.7772421803304252,
      0.7772421803304252,
      0.7772421803304252,
      0.7772421803304252,
      0.7691168179039268,
      0.7691168179039268,
      0.7691168179039268,
      0.7606762112672418,
      0.7606762112672418,
      0.7606762112672418,
      0.7606762112672418,
      0.7516654395309634,
      0.7332315064213829,
      0.7332315064213829,
      0.7332315064213829,
      0.7332315064213829,
      0.72353421335051,
      0.72353421335051,
      0.72353421335051,
      0.72353421335051,
      0.7123287748023364,
      0.7123287748023364,
      0.701016816976142,
      0.6894200841561621,
      0.6894200841561621,
      0.6894200841561621,
      0.6894200841561621,
      0.6894200841561621,
      0.6766325366843785,
      0.6766325366843785,
      0.6766325366843785,
      0.6766325366843785,
      0.6766325366843785,
      0.6632684512316521,
      0.6499342192863836,
      0.6366306508894483,
      0.6230543536793504,
      0.5961870412907047,
      0.5961870412907047,
      0.5684175872956276,
      0.5684175872956276,
      0.5684175872956276,
      0.5684175872956276,
      0.5684175872956276,
      0.5517412721191824,
      0.5517412721191824,
      0.5517412721191824,
      0.5517412721191824,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.5025994585188811,
      0.5025994585188811,
      0.5025994585188811,
      0.5025994585188811,
      0.5025994585188811,
      0.4677906594683743,
      0.4677906594683743,
      0.4677906594683743,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376],
     'survival_time': [50, 100, 150, 200, 250],
     'survival_probabilities': [79.28481866122223,
      72.35342133505101,
      66.32684512316521,
      55.17412721191825,
      42.703537485694376],
     'risk_category': 'High-Risk'}



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival probability profile
# of the final survival prediction model
# for an individual test case
# against the training data as baseline
##################################
response = requests.post(f"{SP_FASTAPI_BASE_URL}/plot-coxph-survival-profile/", json=test_case_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("coxph_survival_function_plot.png", "wb") as f:
        f.write(img)
        display(Image.open("coxph_survival_function_plot.png"))
else:
    print("Error:", response.status_code, response.text)

```


    
![png](output_35_0.png)
    


## 1.3. Application Containerization <a class="anchor" id="1.3"></a>

### 1.3.1 Docker File Creation <a class="anchor" id="1.3.1"></a>

1. A Dockerfile was created to containerize the FastAPI survival prediction application with the following steps:
    * **Base Image Selection**
        * Used the official Python 3.12.5 image as the base environment.
    * **Setting Up the Working Directory**
        * Defined `/app` as the working directory inside the container.
    * **Installing System Dependencies**
        * Updated package lists and installed Git for version control inside the container.
    * **Copying Required Files**
        * Copied requirements.txt into the container for dependency installation.
        * Copied essential application directories (apis, models, datasets, pipelines, parameters) into the container.
    * **Installing Python Dependencies**
        * Upgraded pip and installed the required Python libraries listed in requirements.txt using `pip install --no-cache-dir`.
    * **Exposing Application Port**
        * Configured port 8001 to allow external access to the FastAPI application.
    * **Defining the Container Startup Command**
        * Set the default command to run uvicorn, launching the FastAPI app (`apis.survival_prediction_fastapi:app`) on 0.0.0.0:8001.


![sp_fastapi_docker_file_creation.png](9b7fee0d-0fe8-424b-a946-d83019c020ff.png)


2. In order for the Docker container to locate the essential application directories (apis, models, datasets, pipelines, parameters) within itself when running the application, the FastAPI code was modified to utilize an absolute path to the  `/app` working directory inside the container, instead of the relative path as what was previously used.
3. The localhost port was changed from 8000 to 8001 to run the containerized FastAPI app using uvicorn.
   

![sp_fastapi_docker_code.png](6c889374-4e2e-4d8c-bfc9-7b61106e7d98.png)

### 1.3.2 Docker Image Building <a class="anchor" id="1.3.2"></a>

1. The Docker image was built with the following steps:
    * **Building the Docker Image**
        * Used `docker build -t survival-prediction-fastapi-app .` to create a Docker image named `survival-prediction-fastapi-app` from the Dockerfile in the current directory.


![sp_fastapi_docker_image_building.png](2c4f25a8-addd-4372-8c73-c66e0a17e6f7.png)

### 1.3.3 Docker Image Testing <a class="anchor" id="1.3.3"></a>

1. The Docker image was tested locally with the following steps:
    * **Testing the Container Locally**
        * Used `docker run -p 8001:8001 survival-prediction-fastapi-app` to start a container, mapping port 8001 on the host machine to port 8001 inside the container.
2. The containerized FastAPI application was verified to be running correctly by replicating the previous test sequence with results presented as follows:
    * **Server Initialization**: FastAPI application was started successfully, with Uvicorn running on http://localhost:8001, indicating that the server and its documentation are active and ready to process requests.
    * **Root Endpoint Accessed (GET /)**: The API received a GET request at the root endpoint and responded with 200 OK, confirming that the service is running and accessible.
    * **Individual Survival Probability Request (POST /compute-individual-coxph-survival-probability-class/)**: A POST request was processed successfully, returning 200 OK, indicating that the API correctly computed survival probabilities and risk categorization for an individual test case.
    * **Batch Survival Profile Request (POST /compute-list-coxph-survival-profile/)**: The API successfully processed a POST request for batch survival profile computation, returning 200 OK, confirming that multiple test cases were handled correctly.
    * **Feature Binning Request (POST /bin-numeric-model-feature/)**: A POST request was successfully executed, returning 200 OK, confirming that the API correctly categorized numeric model features into dichotomous bins.
    * **Kaplan-Meier Plot Request (POST /plot-kaplan-meier/)**: The API successfully processed a POST request, returning 200 OK, indicating that a Kaplan-Meier survival plot was generated and returned as a base64-encoded image.
    * **Test Case Preprocessing Request (POST /preprocess-test-case/)**: The API successfully processed a POST request, returning 200 OK, indicating that an individual test cases was completed preprocessing and transformed to a format suitable for model inference.
    * **Kaplan-Meier Plot Grid Request (POST /plot-kaplan-meier-grid/)**: The API successfully processed a POST request, returning 200 OK, indicating that a Kaplan-Meier survival plot for the individual the test case in direct comparison with the baseline survival curves for each predictor was generated and returned as a base64-encoded image.
    * **Cox Survival Plot Request (POST /plot_coxph_survival_profile/)**: The API successfully processed a POST request, returning 200 OK, indicating that a Cox survival plot for the individual the test case in direct comparison with the baseline survival curves for all the training cases categorized by risk categories was generated and returned as a base64-encoded image.
   

![sp_fastapi_docker_documentation_endpoints.png](67fa7a89-163b-4541-9538-edef3dba07bf.png)

![sp_fastapi_render_schema_endpoints.png](585c424a-c9df-4b20-ab04-597e1c2f9a44.png)

![sp_fastapi_docker_activation_endpoints.png](8187212d-c023-4be8-8ccb-59f5f53a728d.png)


```python
##################################
# Defining the base URL of the API
# for the survival prediction model
##################################
SP_FASTAPI_DOCKER_LOCAL_URL = "http://localhost:8001"
```


```python
##################################
# Generating a GET endpoint request for
# validating API service connection
##################################
response = requests.get(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/")
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
```


    'Response:'



    {'message': 'Welcome to the Survival Prediction API!'}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/compute-individual-coxph-survival-probability-class/", json=single_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



    {'survival_function': [0.9973812917524568,
      0.9920416812438736,
      0.9893236791425079,
      0.972381113071464,
      0.9693179903073035,
      0.9631930672135339,
      0.9631930672135339,
      0.9600469571766689,
      0.9600469571766689,
      0.9568596864927983,
      0.9536305709158891,
      0.9471625843882805,
      0.93729581350105,
      0.9338986486591409,
      0.93048646553474,
      0.9270645831787163,
      0.9202445006124622,
      0.9167715111530355,
      0.9132845175345189,
      0.9097550958520674,
      0.9097550958520674,
      0.9097550958520674,
      0.9060810720432387,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.8985598696587259,
      0.8985598696587259,
      0.8985598696587259,
      0.8945287485160898,
      0.8945287485160898,
      0.8945287485160898,
      0.8945287485160898,
      0.8901959645503091,
      0.8812352215018253,
      0.8812352215018253,
      0.8812352215018253,
      0.8812352215018253,
      0.8764677174183527,
      0.8764677174183527,
      0.8764677174183527,
      0.8764677174183527,
      0.8709113650481243,
      0.8709113650481243,
      0.8652494086650531,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8459534502216807,
      0.8389821875092403,
      0.8319419786276306,
      0.8246669811915435,
      0.8099879066057215,
      0.8099879066057215,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7848178617845467,
      0.7848178617845467,
      0.7848178617845467,
      0.7848178617845467,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7337716342207724,
      0.7337716342207724,
      0.7337716342207724,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696],
     'survival_time': [50, 100, 150, 200, 250],
     'survival_probabilities': [90.97550958520674,
      87.64677174183527,
      84.59534502216806,
      78.48178617845467,
      70.70184115456696],
     'risk_category': 'Low-Risk'}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/compute-list-coxph-survival-profile/", json=train_list)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



    {'survival_profiles': [[0.9973812917524568,
       0.9920416812438736,
       0.9893236791425079,
       0.972381113071464,
       0.9693179903073035,
       0.9631930672135339,
       0.9631930672135339,
       0.9600469571766689,
       0.9600469571766689,
       0.9568596864927983,
       0.9536305709158891,
       0.9471625843882805,
       0.93729581350105,
       0.9338986486591409,
       0.93048646553474,
       0.9270645831787163,
       0.9202445006124622,
       0.9167715111530355,
       0.9132845175345189,
       0.9097550958520674,
       0.9097550958520674,
       0.9097550958520674,
       0.9060810720432387,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.8985598696587259,
       0.8985598696587259,
       0.8985598696587259,
       0.8945287485160898,
       0.8945287485160898,
       0.8945287485160898,
       0.8945287485160898,
       0.8901959645503091,
       0.8812352215018253,
       0.8812352215018253,
       0.8812352215018253,
       0.8812352215018253,
       0.8764677174183526,
       0.8764677174183526,
       0.8764677174183526,
       0.8764677174183526,
       0.8709113650481243,
       0.8709113650481243,
       0.8652494086650531,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8459534502216807,
       0.8389821875092403,
       0.8319419786276306,
       0.8246669811915435,
       0.8099879066057215,
       0.8099879066057215,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7848178617845467,
       0.7848178617845467,
       0.7848178617845467,
       0.7848178617845467,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7337716342207724,
       0.7337716342207724,
       0.7337716342207724,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695],
      [0.9761144218801228,
       0.928980888267716,
       0.905777064852962,
       0.7724242339590301,
       0.7502787164583535,
       0.7076872741961866,
       0.7076872741961866,
       0.6866593185026403,
       0.6866593185026403,
       0.6659260634219393,
       0.6454915885099762,
       0.6062342264686207,
       0.5504405490863784,
       0.5323184765768243,
       0.5146536440658629,
       0.49746533888245986,
       0.464726413395405,
       0.4488047347163327,
       0.433309930422515,
       0.4181140392975694,
       0.4181140392975694,
       0.4181140392975694,
       0.4028020116306455,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.373006008573048,
       0.373006008573048,
       0.373006008573048,
       0.35785929480690143,
       0.35785929480690143,
       0.35785929480690143,
       0.35785929480690143,
       0.3421927616040032,
       0.3117176431598899,
       0.3117176431598899,
       0.3117176431598899,
       0.3117176431598899,
       0.29651072362871467,
       0.29651072362871467,
       0.29651072362871467,
       0.29651072362871467,
       0.2796248763802668,
       0.2796248763802668,
       0.2633052706162029,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.213871875082776,
       0.19816204676152543,
       0.18334919195124752,
       0.16908728217620728,
       0.1432835564355214,
       0.1432835564355214,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.10710184631976834,
       0.10710184631976834,
       0.10710184631976834,
       0.10710184631976834,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.05761125190131533,
       0.05761125190131533,
       0.05761125190131533,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404]]}



```python
##################################
# Sending a POST endpoint request for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/bin-numeric-model-feature/", json=bin_request)
if response.status_code == 200:
    display("Response:", pd.DataFrame(response.json()))
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>High</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>High</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival profiles
# using Kaplan-Meier Plots
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/plot-kaplan-meier/", json=km_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("kaplan_meier_plot_docker_local.png", "wb") as f:
        f.write(img)
        display(Image.open("kaplan_meier_plot_docker_local.png"))
else:
    print("Error:", response.status_code, response.text)

```


    
![png](output_54_0.png)
    



```python
##################################
# Sending a POST endpoint request for
# preprocessing an individual test case
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/preprocess-test-case/", json=test_case_request)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



    [{'AGE': 'High',
      'EJECTION_FRACTION': 'Low',
      'SERUM_CREATININE': 'High',
      'SERUM_SODIUM': 'Low',
      'ANAEMIA': 'Present',
      'HIGH_BLOOD_PRESSURE': 'Absent'}]



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival profiles
# using a Kaplan-Meier Plot Matrix
# for an individual test case
# against the training data as baseline
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/plot-kaplan-meier-grid/", json=test_case_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("kaplan_meier_plot_matrix_docker_local.png", "wb") as f:
        f.write(img)
        display(Image.open("kaplan_meier_plot_matrix_docker_local.png"))
else:
    print("Error:", response.status_code, response.text)

```


    
![png](output_56_0.png)
    



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/compute-test-coxph-survival-probability-class/", json=test_case_request)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



    {'survival_function': [0.9935852422336905,
      0.980581105741338,
      0.974000615497288,
      0.9335716291520674,
      0.9263704990495979,
      0.9120703170806183,
      0.9120703170806183,
      0.9047761239313578,
      0.9047761239313578,
      0.8974218629392444,
      0.8900072909706268,
      0.8752652317601862,
      0.8530570074516991,
      0.845488798281818,
      0.8379273268795359,
      0.8303847506918888,
      0.8154721703733717,
      0.8079397039318209,
      0.8004184982181605,
      0.7928481866122223,
      0.7928481866122223,
      0.7928481866122223,
      0.7850129580161529,
      0.7772421803304252,
      0.7772421803304252,
      0.7772421803304252,
      0.7772421803304252,
      0.7772421803304252,
      0.7691168179039268,
      0.7691168179039268,
      0.7691168179039268,
      0.7606762112672418,
      0.7606762112672418,
      0.7606762112672418,
      0.7606762112672418,
      0.7516654395309634,
      0.7332315064213829,
      0.7332315064213829,
      0.7332315064213829,
      0.7332315064213829,
      0.72353421335051,
      0.72353421335051,
      0.72353421335051,
      0.72353421335051,
      0.7123287748023364,
      0.7123287748023364,
      0.701016816976142,
      0.6894200841561621,
      0.6894200841561621,
      0.6894200841561621,
      0.6894200841561621,
      0.6894200841561621,
      0.6766325366843785,
      0.6766325366843785,
      0.6766325366843785,
      0.6766325366843785,
      0.6766325366843785,
      0.6632684512316521,
      0.6499342192863836,
      0.6366306508894483,
      0.6230543536793504,
      0.5961870412907047,
      0.5961870412907047,
      0.5684175872956276,
      0.5684175872956276,
      0.5684175872956276,
      0.5684175872956276,
      0.5684175872956276,
      0.5517412721191824,
      0.5517412721191824,
      0.5517412721191824,
      0.5517412721191824,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.533600088098597,
      0.5025994585188811,
      0.5025994585188811,
      0.5025994585188811,
      0.5025994585188811,
      0.5025994585188811,
      0.4677906594683743,
      0.4677906594683743,
      0.4677906594683743,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376,
      0.42703537485694376],
     'survival_time': [50, 100, 150, 200, 250],
     'survival_probabilities': [79.28481866122223,
      72.35342133505101,
      66.32684512316521,
      55.17412721191825,
      42.703537485694376],
     'risk_category': 'High-Risk'}



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival probability profile
# of the final survival prediction model
# for an individual test case
# against the training data as baseline
##################################
response = requests.post(f"{SP_FASTAPI_DOCKER_LOCAL_URL}/plot-coxph-survival-profile/", json=test_case_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("coxph_survival_function_plot_docker_local.png", "wb") as f:
        f.write(img)
        display(Image.open("coxph_survival_function_plot_docker_local.png"))
else:
    print("Error:", response.status_code, response.text)

```


    
![png](output_58_0.png)
    


### 1.3.4 Docker Image Storage <a class="anchor" id="1.3.4"></a>

1. The Docker image was pushed to DockerHub to ensure persistent storage with the following steps:
    * **Authenticating with DockerHub**
        * Logged into DockerHub using `docker login` to enable pushing images to a remote repository.
    * **Tagging the Docker Image**
        * Used `docker tag survival-prediction-fastapi-app johnpaulinepineda/survival-prediction-fastapi-app:latest` to assign a repository name and version (latest) to the image for DockerHub.   
    * **Pushing the Image to DockerHub**
        * Used `docker push johnpaulinepineda/survival-prediction-fastapi-app:latest` to upload the tagged image to DockerHub, making it available for deployment on external platforms.
    * **Verifying DockerHub Image Upload Completion**
        * Confirmed the successful transfer of the `johnpaulinepineda/survival-prediction-fastapi-app` image to the repository under the `johnpaulinepineda` DockerHub namespace.


![sp_fastapi_docker_image_dockerhub_upload.png](673988f5-5e9b-45e8-8741-1e5d4585cbf2.png)

![sp_fastapi_docker_image_dockerhub_upload_status.png](5ad121f9-f188-48ef-a080-3e1357df8f31.png)

## 1.4. Application Programming Interface (API) Deployment <a class="anchor" id="1.4"></a>

### 1.4.1 Docker Image Execution and Hosting <a class="anchor" id="1.4.1"></a>

1. The containerized FastAPI application was deployed on Render with the following steps:
    * **Preparing the DockerHub Repository**
        * Ensured the `survival-prediction-fastapi-app` image was available in the `johnpaulinepineda` DockerHub namespace.
        * Made the repository public or granted Render access to a private repository.
    * **Creating a New Web Service on Render**
        * Selected `New` → `Web Service` in the Render dashboard.
        * Chose `Deploy an existing image from a registry` and provided the Docker image name from DockerHub.
    * **Configuring the Service**
        * Named the service `heart-failure-survival-probability-estimation` and selected `Singapore` as the region.
        * Chose the `Free instance` type for testing.
        * Set the service to run on port 8001, matching the port exposed in the Dockerfile.
        * Added an environment variable (`Key: PORT` and `Value: 8001`) under advanced settings.
    * **Deploying and Testing the API**
        * Render pulled the image from DockerHub and deployed the container.
        * Accessed the API using the public Render URL.
        * Verified the FastAPI endpoints via the interactive documentation at `/docs`.


![sp_fastapi_docker_image_render_hosting.png](88ff1f46-e894-4b80-a952-cfac07f8071e.png)

![sp_fastapi_render_documentation_endpoints.png](66a44b6b-c6f4-4a9a-99a5-7623d173ee9f.png)

![sp_fastapi_render_schema_endpoints.png](4b571faf-14b7-4763-bafc-2d307925b0f4.png)

## 1.5. User Interface (UI) Development <a class="anchor" id="1.5"></a>

### 1.5.1 UI Building With API Calls <a class="anchor" id="1.5.1"></a>

1. A Streamlit UI was developed to interact with the FastAPI backend for heart failure survival prediction with the following steps:
    * **Loading Python Libraries**
        * Imported `streamlit` for UI development and `requests` for API communication.
        * Used `base64`, `PIL`, and `io` for handling and displaying image outputs.
    * **Defining the FastAPI Endpoint URL**
        * Set `SP_FASTAPI_RENDER_URL` as the base URL for the FastAPI service deployed on Render.   
    * **Setting the Page Layout**
        * Configured the Streamlit page to a wide layout for better readability.
    * **Defining Input Variables**
        * Listed key cardiovascular, hematologic, and metabolic markers required for prediction.
    * **Creating the UI Components**
        * Displayed a title and description for the application, including links to the project documentation and GitHub repository.
        * Used sliders for numeric inputs (e.g., Age, Ejection Fraction, Serum Creatinine, Serum Sodium).
        * Used radio buttons for categorical features (e.g., Anaemia, High Blood Pressure).
    * **Structuring User Input**
        * Collected and formatted user responses into a dictionary to match the API request format.   
    * **Defining the Prediction Button Action**
        * Added an `Assess Characteristics Against Study Population + Plot Survival Probability Profile + Estimate Heart Failure Survival Probability + Predict Risk Category` action button to trigger API requests. 
    * **Sending API Requests & Displaying Results**
        * Called the FastAPI `/plot-kaplan-meier-grid/` endpoint to visualize the test case against the study population.
        * Called `/plot-coxph-survival-profile/` to generate the survival probability curve.
        * Called `/compute-test-coxph-survival-probability-class/` to estimate survival probabilities at multiple time points and predict the risk category.
    * **Displaying Model Predictions**
        * Rendered probability estimates dynamically, coloring the text blue for low-risk and red for high-risk predictions.
        * Used structured markdown to enhance readability and interpretation of results.


![sp_streamlit_code.png](98701bfb-f34c-4ef9-a975-8fca643e0da6.png)

## 1.6. Web Application Deployment <a class="anchor" id="1.6"></a>

### 1.6.1 UI Hosting <a class="anchor" id="1.6.1"></a>

1. The containerized FastAPI application was deployed on Render with the following steps:
    * **Preparing the GitHub Repository**
        * Ensured the project repository was on GitHub and accessible to Render.
        * Verified that `streamlit_app.py` and `requirements.txt` were inside the `uis/` directory.
    * **Creating a New Web Service on Render**
        * Logged into the Render Dashboard and selected `New` → `Web Service`.
        * Connected the service to the GitHub repository and set `uis/` as the root directory in the advanced settings.
    * **Configuring the Deployment**
        * Defined the `Build Command` as `pip install -r requirements.txt` to install dependencies.
        * Set the `Start Command` to run the Streamlit app on the assigned port.
        * Chose `Python 3.12+` as the runtime environment and opted for the `Free Tier` instance type.
    * **Deploying the Application**
        * Clicked `Create Web Service` to begin the deployment process.
        * Waited for Render to install dependencies and launch the application.
        * As a requirement, activated the previously deployed FastAPI Docker image first via the public Render URL ([https://heart-failure-survival-probability.onrender.com/](https://heart-failure-survival-probability.onrender.com/)).
        * Accessed the deployed app via the public Render URL ([https://heart-failure-survival-estimation.onrender.com/](https://heart-failure-survival-estimation.onrender.com/)).


![sp_fastapi_render_service.png](5dd5a14f-723d-4171-84a6-29b4207c2e32.png)

![sp_streamlit_render_service.png](ef15c655-62c1-4547-bbb6-bc82de90294e.png)

### 1.6.2 Application Testing <a class="anchor" id="1.6.2"></a>

1. The complete application was tested by verifying the functionality of the FastAPI backend and the Streamlit UI deployed on Render with the following steps:
    * **Started the FastAPI Service**
        * Accessed the FastAPI Docker image via its public Render URL ([https://heart-failure-survival-probability.onrender.com/](https://heart-failure-survival-probability.onrender.com/)).
    * **Accessed the Streamlit UI**
        * Opened the deployed Streamlit app through its Render URL ([https://heart-failure-survival-estimation.onrender.com/](https://heart-failure-survival-estimation.onrender.com/)).
    * **Tested User Inputs**
        * Used range sliders to enter numerical values for cardiovascular, hematologic, and metabolic markers.
        * Used radio buttons to select binary categories (Present | Absent) for hematologic and cardiovascular conditions.
    * **Executed Model Prediction Workflow**
         * Enabled the `Assess Characteristics Against Study Population + Plot Survival Probability Profile + Estimate Heart Failure Survival Probability + Predict Risk Category` action button to trigger the entire pipeline:
             * Plotted study population data as a baseline comparison.
             * Processed the user’s test case inputs and displayed them in visualization charts.
             * Executed computations for survival probability estimation.
             * Predicted the test case's survival probability and risk category.
    * **Verified Output Accuracy and API Response**
         * Ensured that FastAPI correctly processed the test case inputs and returned expected results.
         * Checked that Streamlit displayed the survival probability plots, numerical estimations, and risk classification without errors.


![sp_application_render_service.png](bb50ebe0-785b-40d3-8746-c6387e2c4aff.png)

## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. This project explored open-source solutions for containerizing and deploying machine learning API endpoints, followed by developing and deploying a UI for user interaction. The goal was to operationalize a survival prediction model for heart failure by enabling users to input health markers and receive survival probability estimates.
2. The project execution process and significant tools used for implementation are summarized as follows:
    * **Model Development and API Creation (FastAPI)**
        * FastAPI was chosen for its speed and ease of use in building an efficient RESTful API for the survival prediction model.
        * The API was developed to process user inputs and return probability estimates.
    * **Local Testing of the API (FastAPI | Requests)**
        * The API was tested to verify correct response formatting and model inference behavior.
        * Python’s requests library was used to simulate test cases before deployment.
    * **Containerization (Docker)**
        * A Dockerfile was created to package the FastAPI application and its dependencies into a portable container.
        * The containerized application ensured a reproducible environment across different platforms.
    * **Building and Testing the Docker Image (Docker Desktop)**
        * The Docker image was built and tested locally to ensure proper functionality before deployment.
        * Running the container locally validated that the application performed consistently inside a Dockerized environment.
    * **Deploying the Containerized API (DockerHub | Render)**
        * The Docker image was pushed to DockerHub for persistent storage and easy access from cloud platforms.
        * Render was selected for deployment due to its free-tier hosting and seamless integration with DockerHub.
        * Environment variables and port configurations were set up to align with the deployed service.
    * **Developing the UI (Streamlit)**
        * Streamlit was chosen for its simplicity in creating interactive web applications with minimal frontend development effort.
        * The UI allowed users to enter cardiovascular, hematologic, and metabolic markers through sliders and buttons.
        * The interface processed user inputs, sent them to the FastAPI backend, and displayed survival probability estimates in real-time.
    * **Deploying the UI (GitHub | Render)**
        * The Streamlit UI was pushed to GitHub, which acted as a code repository for deployment.
        * Render was used again to host the Streamlit app, ensuring both the backend and frontend remained on open-source platforms.
    * **End-to-End Testing of the Deployed Application**
        * The complete application was tested by verifying API responses, UI interactions, and model predictions.
        * The public URLs of both the FastAPI backend and the Streamlit UI were used to validate functionality.
3. The end-to-end deployment of a machine learning model using open-source platforms was successfully demonstrated. The approach highlights how open-source tools can be effectively leveraged for machine learning model deployment in production environments.
    * **FastAPI** ensured efficient API development
    * **Docker** enabled portability
    * **DockerHub** provided persistent storage
    * **Render** facilitated cloud deployment
    * **Streamlit** made UI development seamless. 

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Building Machine Learning Powered Applications: Going From Idea to Product](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) by Emmanuel Ameisen
* **[Book]** [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen
* **[Book]** [Machine Learning Bookcamp: Build a Portfolio of Real-Life Projects](https://www.manning.com/books/machine-learning-bookcamp) by Alexey Grigorev and Adam Newmark 
* **[Book]** [Building Machine Learning Pipelines: Automating Model Life Cycles with TensorFlow](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/) by Hannes Hapke and Catherine Nelson
* **[Book]** [Hands-On APIs for AI and Data Science: Python Development with FastAPI](https://handsonapibook.com/index.html) by Ryan Day
* **[Book]** [Managing Machine Learning Projects: From Design to Deployment](https://www.manning.com/books/managing-machine-learning-projects) by Simon Thompson
* **[Book]** [Building Data Science Applications with FastAPI: Develop, Manage, and Deploy Efficient Machine Learning Applications with Python](https://www.oreilly.com/library/view/building-data-science/9781837632749/) by François Voron
* **[Book]** [Microservice APIs: Using Python, Flask, FastAPI, OpenAPI and More](https://www.manning.com/books/microservice-apis) by Jose Haro Peralta
* **[Book]** [Machine Learning Engineering with Python: Manage the Lifecycle of Machine Learning odels using MLOps with Practical Examples](https://www.oreilly.com/library/view/machine-learning-engineering/9781837631964/) by Andrew McMahon
* **[Book]** [Introducing MLOps: How to Scale Machine Learning in the Enterprise](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/) by Mark Treveil, Nicolas Omont, Clément Stenac, Kenji Lefevre, Du Phan, Joachim Zentici, Adrien Lavoillotte, Makoto Miyazaki and Lynn Heidmann
* **[Book]** [Practical Python Backend Programming: Build Flask and FastAPI Applications, Asynchronous Programming, Containerization and Deploy Apps on Cloud](https://leanpub.com/practicalpythonbackendprogramming) by Tim Peters
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.image](https://matplotlib.org/stable/api/image_api.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.offsetbox](https://matplotlib.org/stable/api/offsetbox_api.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Python Library API]** [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.svm](https://scikit-learn.org/stable/modules/svm.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model_selection](https://scikit-learn.org/stable/model_selection.html) by Scikit-Learn Team
* **[Python Library API]** [imblearn.over_sampling](https://imbalanced-learn.org/stable/over_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [imblearn.under_sampling](https://imbalanced-learn.org/stable/under_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [SciKit-Survival](https://pypi.org/project/scikit-survival/) by SciKit-Survival Team
* **[Python Library API]** [SciKit-Learn](https://scikit-learn.org/stable/index.html) by SciKit-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [Lifelines](https://lifelines.readthedocs.io/en/latest/) by Lifelines Team
* **[Python Library API]** [Streamlit](https://streamlit.io/) by Streamlit Team
* **[Python Library API]** [Streamlit Community Cloud](https://streamlit.io/cloud) by Streamlit Team
* **[Article]** [ML - Deploy Machine Learning Models Using FastAPI](https://dorian599.medium.com/ml-deploy-machine-learning-models-using-fastapi-6ab6aef7e777) by Dorian Machado (Medium)
* **[Article]** [Deploying Machine Learning Models Using FastAPI](https://medium.com/@kevinnjagi83/deploying-machine-learning-models-using-fastapi-0389c576d8f1) by Kevin Njagi (Medium)
* **[Article]** [Deploy Machine Learning API with FastAPI for Free](https://lightning.ai/lightning-ai/studios/deploy-machine-learning-api-with-fastapi-for-free?section=featured) by Aniket Maurya (Lightning.AI)
* **[Article]** [How to Use FastAPI for Machine Learning](https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/) by Cheuk Ting Ho (JetBrains.Com)
* **[Article]** [Deploying and Hosting a Machine Learning Model with FastAPI and Heroku](https://testdriven.io/blog/fastapi-machine-learning/) by Michael Herman (TestDriven.IO)
* **[Article]** [A Practical Guide to Deploying Machine Learning Models](https://machinelearningmastery.com/a-practical-guide-to-deploying-machine-learning-models/) by Bala Priya (MachineLearningMastery.Com)
* **[Article]** [Using FastAPI to Deploy Machine Learning Models](https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea) by Carl Handlin (Medium)
* **[Article]** [How to Deploy a Machine Learning Model](https://www.maartengrootendorst.com/blog/deploy/) by Maarten Grootendorst (MaartenGrootendorst.Com)
* **[Article]** [Accelerating Machine Learning Deployment: Unleashing the Power of FastAPI and Docker](https://medium.datadriveninvestor.com/accelerating-machine-learning-deployment-unleashing-the-power-of-fastapi-and-docker-933865cb990a) by Pratyush Khare (Medium)
* **[Article]** [Containerize and Deploy ML Models with FastAPI & Docker](https://towardsdev.com/containerize-and-deploy-ml-models-with-fastapi-docker-d8c19cc8ef94) by Hemachandran Dhinakaran (Medium)
* **[Article]** [Quick Tutorial to Deploy Your ML models using FastAPI and Docker](https://shreyansh26.github.io/post/2020-11-30_fast_api_docker_ml_deploy/) by Shreyansh Singh (GitHub)
* **[Article]** [How to Deploying Machine Learning Models in Production](https://levelup.gitconnected.com/how-to-deploying-machine-learning-models-in-production-3009b90eadfa) by Umair Akram (Medium)
* **[Article]** [Deploying a Machine Learning Model with FastAPI: A Comprehensive Guide](https://ai.plainenglish.io/deploying-a-machine-learning-model-with-fastapi-a-comprehensive-guide-997ac747601d) by Muhammad Naveed Arshad (Medium)
* **[Article]** [Deploy Machine Learning Model with REST API using FastAPI](https://blog.yusufberki.net/deploy-machine-learning-model-with-rest-api-using-fastapi-288f229161b7) by Yusuf Berki Yazıcıoğlu (Medium)
* **[Article]** [Deploying An ML Model With FastAPI — A Succinct Guide](https://towardsdatascience.com/deploying-an-ml-model-with-fastapi-a-succinct-guide-69eceda27b21) by Yash Prakash (Medium)
* **[Article]** [How to Build a Machine Learning App with FastAPI: Dockerize and Deploy the FastAPI Application to Kubernetes](https://dev.to/bravinsimiyu/beginner-guide-on-how-to-build-a-machine-learning-app-with-fastapi-part-ii-deploying-the-fastapi-application-to-kubernetes-4j6g) by Bravin Wasike (Dev.TO)
* **[Article]** [Building a Machine Learning Model API with Flask: A Step-by-Step Guide](https://medium.com/@nileshshindeofficial/building-a-machine-learning-model-api-with-flask-a-step-by-step-guide-6f85e9bb9773) by Nilesh Shinde (Medium)
* **[Article]** [Deploying Your Machine Learning Model as a REST API Using Flask](https://medium.com/analytics-vidhya/deploying-your-machine-learning-model-as-a-rest-api-using-flask-c2e6a0b574f5) by Emmanuel Oludare (Medium)
* **[Article]** [Machine Learning Model Deployment on Heroku Using Flask](https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da) by Charu Makhijani (Medium)
* **[Article]** [Model Deployment using Flask](https://towardsdatascience.com/model-deployment-using-flask-c5dcbb6499c9) by Ravindra Sharma (Medium)
* **[Article]** [Deploy a Machine Learning Model using Flask: Step-By-Step](https://codefather.tech/blog/deploy-machine-learning-model-flask/) by Claudio Sabato (CodeFather.Tech)
* **[Article]** [How to Deploy a Machine Learning Model using Flask?](https://datadance.ai/machine-learning/how-to-deploy-a-machine-learning-model-using-flask/) by DataDance.AI Team (DataDance.AI)
* **[Article]** [A Comprehensive Guide on Deploying Machine Learning Models with Flask](https://machinelearningmodels.org/a-comprehensive-guide-on-deploying-machine-learning-models-with-flask/) by MachineLearningModels.Org Team (MachineLearningModels.Org)
* **[Article]** [How to Deploy Machine Learning Models with Flask and Docker](https://python.plainenglish.io/how-to-deploy-machine-learning-models-with-flask-and-docker-3c4d6116e809) by Usama Malik (Medium)
* **[Article]** [Deploying Machine Learning Models with Flask: A Step-by-Step Guide](https://medium.com/@sukmahanifah/deploying-machine-learning-models-with-flask-a-step-by-step-guide-cd22967c1f66) by Sukma Hanifa (Medium)
* **[Article]** [Machine Learning Model Deployment on Heroku Using Flask](https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da) by Charu Makhijani (Medium)
* **[Article]** [Complete Guide on Model Deployment with Flask and Heroku](https://towardsdatascience.com/complete-guide-on-model-deployment-with-flask-and-heroku-98c87554a6b9) by Tarek Ghanoum (Medium)
* **[Article]** [Turning Machine Learning Models into APIs in Python](https://www.datacamp.com/tutorial/machine-learning-models-api-python) by Sayak Paul (DataCamp)
* **[Article]** [Machine Learning, Pipelines, Deployment and MLOps Tutorial](https://www.datacamp.com/tutorial/tutorial-machine-learning-pipelines-mlops-deployment) by Moez Ali (DataCamp)
* **[Article]** [Docker vs. Podman: Which Containerization Tool is Right for You](https://www.datacamp.com/blog/docker-vs-podman) by Jake Roach (DataCamp)
* **[Article]** [Introduction to Podman for Machine Learning: Streamlining MLOps Workflows](https://geekflare.com/devops/podman-vs-docker/) by Abid Ali Awan (DataCamp)
* **[Article]** [Podman vs Docker: Which One to Choose?](https://www.datacamp.com/tutorial/tutorial-machine-learning-pipelines-mlops-deployment) by Talha Khalid (GeekFlare)
* **[Article]** [Docker Vs Podman : Which One to Choose?](https://blog.fourninecloud.com/docker-vs-podman-which-one-to-choose-b6387bd29db3) by Saiteja Bellam (Medium)
* **[Article]** [Podman vs Docker: What Are the Key Differences Explained in Detail](https://www.geeksforgeeks.org/podman-vs-docker/) by Geeks For Geeks Team (GeeksForGeeks.Com)
* **[Video Tutorial]** [Machine Learning Models Deployment with Flask and Docker](https://www.youtube.com/watch?v=KTd2a1QKlwo) by Data Science Dojo (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Model Flask](https://www.youtube.com/watch?v=MxJnR1DMmsY) by Stats Wire (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Models with Flask | Using Render to host API and Get URL :Step-By-Step Guide](https://www.youtube.com/watch?v=LBlvuUaIg58) by Prachet Shah (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Model using Flask](https://www.youtube.com/watch?app=desktop&v=UbCWoMf80PY&t=597s) by Krish Naik (YouTube)
* **[Video Tutorial]** [Deploy Your ML Model Using Flask Framework](https://www.youtube.com/watch?v=PtyyVGsE-u0) by MSFTImagine (YouTube)
* **[Video Tutorial]** [Build a Machine Learning App From Scratch with Flask & Docker](https://www.youtube.com/watch?v=S--SD4QbGps) by Patrick Loeber (YouTube)
* **[Video Tutorial]** [Deploying a Machine Learning Model to a Web with Flask and Python Anywhere](https://www.youtube.com/watch?v=3w3vBu2WMvk) by Prof. Phd. Manoel Gadi (YouTube)
* **[Video Tutorial]** [End To End Machine Learning Project With Deployment Using Flask](https://www.youtube.com/watch?v=RnOU2bumBPE) by Data Science Diaries (YouTube)
* **[Video Tutorial]** [Publish ML Model as API or Web with Python Flask](https://www.youtube.com/watch?v=_cLbGKKrggs) by Python ML Daily (YouTube)
* **[Video Tutorial]** [Deploy a Machine Learning Model using Flask API to Heroku](https://www.youtube.com/watch?v=Q_Z5kzKpofk) by Jackson Yuan (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning Model with FlaskAPI - CI/CD for ML Series](https://www.youtube.com/watch?v=vxF5uEoL1C4) by Anthony Soronnadi (YouTube)
* **[Video Tutorial]** [Deploy ML model as Webservice | ML model deployment | Machine Learning | Data Magic](https://www.youtube.com/watch?v=3U1T8cLL-1M) by Data Magic (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning Model Using Flask](https://www.youtube.com/watch?v=ng15EVDrL28) by DataMites (YouTube)
* **[Video Tutorial]** [ML Model Deployment With Flask On Heroku | How To Deploy Machine Learning Model With Flask | Edureka](https://www.youtube.com/watch?v=pMIwu5FwJ78) by Edureka (YouTube)
* **[Video Tutorial]** [ML Model Deployment with Flask | Machine Learning & Data Science](https://www.youtube.com/watch?v=Od0gS3Qeges) by Dan Bochman (YouTube)
* **[Video Tutorial]** [How to Deploy ML Solutions with FastAPI, Docker, & AWS](https://www.youtube.com/watch?v=pJ_nCklQ65w) by Shaw Talebi (YouTube)
* **[Video Tutorial]** [Deploy ML models with FastAPI, Docker, and Heroku | Tutorial](https://www.youtube.com/watch?v=h5wLuVDr0oc) by AssemblyAI (YouTube)
* **[Video Tutorial]** [Machine Learning Model Deployment Using FastAPI](https://www.youtube.com/watch?v=0s-oat69UqU) by TheOyinbooke (YouTube)
* **[Video Tutorial]** [Creating APIs For Machine Learning Models with FastAPI](https://www.youtube.com/watch?v=5PgqzVG9SCk) by NeuralNine (YouTube)
* **[Video Tutorial]** [How To Deploy Machine Learning Models Using FastAPI-Deployment Of ML Models As API’s](https://www.youtube.com/watch?v=b5F667g1yCk) by Krish Naik (YouTube)
* **[Video Tutorial]** [Machine Learning Model with FastAPI, Streamlit and Docker](https://www.youtube.com/watch?v=cCsnmxXxWaM) by CodeTricks (YouTube)
* **[Video Tutorial]** [FastAPI Machine Learning Model Deployment | Python | FastAPI](https://www.youtube.com/watch?v=DUhzTi3w5KA) by Stats Wire (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning Models - Full Guide](https://www.youtube.com/watch?v=oyYur3uVl4w) by NeuralNine (YouTube)
* **[Video Tutorial]** [Model Deployment FAST API - Docker | Machine Learning Model Deployment pipeline | FastAPI VS Flask](https://www.youtube.com/watch?v=YvvOuY9L_Yw) by 360DigiTMG (YouTube)
* **[Video Tutorial]** [Build an AI app with FastAPI and Docker - Coding Tutorial with Tips](https://www.youtube.com/watch?v=iqrS7Q174Ac) by Patrick Loeber (YouTube)
* **[Video Tutorial]** [Create a Deep Learning API with Python and FastAPI](https://www.youtube.com/watch?v=NrarIs9n24I) by DataQuest (YouTube)
* **[Video Tutorial]** [Fast API Machine Learning Web App Tutorial + Deployment on Heroku](https://www.youtube.com/watch?v=LSXU3dEDg9A) by Greg Hogg (YouTube)
* **[Course]** [Deeplearning.AI Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) by DeepLearning.AI Team (Coursera)
* **[Course]** [IBM AI Workflow: Enterprise Model Deployment](https://www.coursera.org/learn/ibm-ai-workflow-machine-learning-model-deployment) by IBM Team (Coursera)
* **[Course]** [DataCamp Machine Learning Engineer Track](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Designing Machine Learning Workflows in Python](https://app.datacamp.com/learn/courses/designing-machine-learning-workflows-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Building APIs in Python](https://app.datacamp.com/learn/skill-tracks/building-apis-in-python) by DataCamp Team (DataCamp)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

