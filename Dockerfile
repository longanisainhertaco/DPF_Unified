FROM python:3.11-slim

WORKDIR /app

# git is needed by app.py (git rev-parse for version display)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY app*.py ./
COPY static/ static/

# Install CPU-only torch explicitly before the package install to avoid pulling GPU wheels
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[viz]"

EXPOSE 7860

ENV DPF_UI_PORT=7860
# Use tutorial preset for public deployments (fast, < 1 sec default run)
ENV DPF_DEFAULT_PRESET=tutorial

CMD ["python3", "app.py"]
