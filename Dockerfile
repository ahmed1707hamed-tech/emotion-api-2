# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy requirements file and install dependencies
# Doing this before copying the rest of the code helps leverage Docker cache
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire project directory into the container
COPY ./ /code/

# Expose the port the app runs on
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn
# Hugging Face Spaces typically uses port 7860 by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
