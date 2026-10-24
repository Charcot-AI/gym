FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[server]'

EXPOSE 8000
CMD ["uvicorn", "charcot_gym.server:app", "--host", "0.0.0.0", "--port", "8000"]
