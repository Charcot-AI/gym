FROM python:3.12-slim AS ehr-web

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[ehr-web]'
RUN playwright install --with-deps chromium

EXPOSE 8000
CMD ["uvicorn", "charcot_gym.ehr_web.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim AS default

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[server]'

EXPOSE 8000
CMD ["uvicorn", "charcot_gym.server:app", "--host", "0.0.0.0", "--port", "8000"]
