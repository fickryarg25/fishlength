# Use Python 3.14
FROM python:3.14-slim

# Set the working directory
WORKDIR /app

# Install system tools for image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install your Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Tell Google Cloud to use port 8080
EXPOSE 8080

# Start the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
