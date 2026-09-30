FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYHTONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY app.py ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install .

COPY data/ ./data/
COPY docs/ ./docs/

RUN mkdir -p /app/logs /app/models /app/reports/figures

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "app"]