FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backtest_worker.py .
COPY gcp_config.py .

# Set Python path
ENV PYTHONPATH=/app

# Default command (will be overridden by Cloud Batch)
CMD ["python3.11", "backtest_worker.py"]

