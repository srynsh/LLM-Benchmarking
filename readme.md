# Automated Benchmarking of LLMs: Applying Regression to Estimate LLM Model Accuracy

## Setup Instructions

1. **Clone the repository and download LFS files:**

    ```bash
    git clone <repository-url>
    cd llm-benchmarking
    git lfs pull
    ```

2. **Extract the dataset:**

    ```bash
    tar -xzvf data.tar.gz
    ```

3. **Build the Docker image:**

    ```bash
    docker build -t llm-benchmarking -f docker/Dockerfile .
    ```

4. **Run the Docker container:**

    ```bash
    docker run --rm -it \
      -v $(pwd)/output:/app/output \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/src:/app/src \
      llm-benchmarking
    ```

5. **Inside the Docker container, run the LLM-as-judge and ensemble:**

    ```bash
    F=0 C=0 N=0 H=0 P=0 python -m src.validation.validator && \
    F=1 C=1 N=0 H=0 P=0 python -m src.validation.validator && \
    F=1 C=1 N=1 H=1 P=0 python -m src.validation.validator && \
    F=1 C=1 N=1 H=1 P=1 python -m src.validation.validator
    ```

6. **Inside the Docker container, run the regression script:**

    ```bash
    F=0 C=0 N=0 H=0 P=0 python -m src.regression.regressor && \
    F=1 C=1 N=0 H=0 P=0 python -m src.regression.regressor && \
    F=1 C=1 N=1 H=1 P=0 python -m src.regression.regressor && \
    F=1 C=1 N=1 H=1 P=1 python -m src.regression.regressor
    ```
