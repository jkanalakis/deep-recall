FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY examples/streamlit-memories/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY memory/ /app/memory/
COPY examples/streamlit-memories/ /app/

# Make sure Python can find the modules
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8404"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8404/health || exit 1 