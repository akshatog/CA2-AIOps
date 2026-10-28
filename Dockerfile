FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
COPY artifacts/requirements-model.txt ./artifacts/requirements-model.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ./
COPY artifacts/ ./artifacts/
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
