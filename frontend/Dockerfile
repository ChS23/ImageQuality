FROM python:3.11-slim-bullseye

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]
