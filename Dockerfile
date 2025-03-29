# Use the official Python 3.12.5 image as the base
FROM python:3.12.5

# Set the working directory inside the container
WORKDIR /app

# Install Git inside the container
RUN apt-get update && apt-get install -y git

# Copy only necessary application files into the container
COPY apis/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the FastAPI app and required directories
COPY apis /app/apis
COPY models /app/models
COPY datasets /app/datasets
COPY pipelines /app/pipelines
COPY parameters /app/parameters

# Expose FastAPI application port
EXPOSE 8001

# Run the FastAPI application
CMD ["uvicorn", "apis.survival_prediction_fastapi:app", "--host", "0.0.0.0", "--port", "8001"]
