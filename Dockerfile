# Using the official Python 3.12.5 image as the base
FROM python:3.12.5

# Setting the working directory inside the container
WORKDIR /app

# Installing Git inside the container
RUN apt-get update && apt-get install -y git

# Copying only necessary application files into the container
COPY apis/requirements.txt /app/requirements.txt

# Installing dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copying the FastAPI app and required directories
COPY apis /app/apis
COPY models /app/models
COPY datasets /app/datasets
COPY pipelines /app/pipelines
COPY parameters /app/parameters

# Exposing FastAPI application port
EXPOSE 8001

# Running the FastAPI application
CMD ["uvicorn", "apis.survival_prediction_fastapi:app", "--host", "0.0.0.0", "--port", "8001"]
