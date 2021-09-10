FROM python:3.8

WORKDIR /usr/src/app
COPY . .

EXPOSE 8000

RUN pip install --no-cache-dir  -r requirements.txt
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app.api:app"]
