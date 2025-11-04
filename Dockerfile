# Agentic Prompt-Injection Robustness Benchmark
# Dockerfile for reproducible experiments

FROM python:3.11-slim

# Set metadata
LABEL maintainer="Agentic Prompt-Injection Benchmark"
LABEL description="Benchmark for evaluating LLM robustness against prompt injection attacks"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p data/logs data/results data/visualizations

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for potential web interface
EXPOSE 8888

# Default command: run tests
CMD ["pytest", "tests/", "-v"]

# Alternative commands (uncomment as needed):
# CMD ["python", "scripts/demo_basic.py"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
