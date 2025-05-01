# cd ..
# sudo docker build -t llm-benchmarking -f docker/Dockerfile .
# sudo docker run --rm -v $(pwd)/output:/app/output llm-benchmarking

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for Python packages with scientific dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for output images
RUN mkdir -p /app/src
RUN mkdir -p /app/data
RUN mkdir -p /app/output /app/output/images /app/output/images/pdfs /app/output/images/pngs /app/output/latex /app/output/interim /app/output/logs
RUN chmod 777 /app/output  


# Copy the rest of the code
COPY src/ /app/src/
COPY data/ /app/data/

# Set working directory to the regressor directory
WORKDIR /app/src/regressor

# Command to run when container starts
CMD ["python", "regressor.py"]
