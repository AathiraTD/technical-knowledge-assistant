FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF handling and database client
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'psycopg[binary]'

# Copy application code
COPY assistant/ ./assistant/
COPY db/ ./db/
COPY eval/ ./eval/
COPY data/ ./data/

# Expose UI port
EXPOSE 8000

# Health check endpoint for the application
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the web UI by default
CMD ["python", "-m", "assistant.ui", "--host", "0.0.0.0", "--port", "8000"]
