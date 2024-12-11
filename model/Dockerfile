# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    libhdf5-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ENV HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

pip install jax==0.4.30 jaxlib==0.4.30

# Install the dependencies
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "vibu.py"]