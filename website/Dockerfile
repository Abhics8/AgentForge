FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential graphviz && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY website/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "src/dashboard.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
