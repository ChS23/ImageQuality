FROM bitnami/pytorch:latest

USER root

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
