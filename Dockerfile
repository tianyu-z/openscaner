FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY models /app/models

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --group ml

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "openscaner.service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
