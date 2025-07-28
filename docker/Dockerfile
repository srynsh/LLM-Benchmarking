# cd .. (comment retained for context)
# sudo docker build --no-cache -t llm-benchmarking -f docker/Dockerfile .
# sudo docker run --rm -v $(pwd)/output:/app/output llm-benchmarking

FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies required for Python packages with scientific dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3 \
    python3-pip \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Create directory for output images
RUN mkdir -p /app/src
# RUN mkdir -p /app/output /app/output/images /app/output/images/pdf_files /app/output/images/png_files /app/output/latex /app/output/interim /app/output/logs
# RUN mkdir -p /app/output /app/output/images /app/output/images/pdfs /app/output/images/pngs /app/output/latex /app/output/interim /app/output/logs
# RUN chmod -R 777 /app/output

# docker


# Copy the rest of the code
COPY src/ /app/src/
COPY data/ /app/data/

# Set working directory to the src directory
WORKDIR /app/

# Command to run when container starts
# CMD ["/bin/bash"]
CMD ["python3", "-m", "src.regression.regressor"]
