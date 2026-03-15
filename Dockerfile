FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY app*.py ./

RUN pip install --no-cache-dir -e "." && \
    pip install --no-cache-dir plotly

EXPOSE 7860

ENV DPF_UI_PORT=7860

CMD ["python3", "app.py"]
