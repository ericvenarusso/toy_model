FROM python:3.8

WORKDIR /usr/src/app
COPY ./producer/produce.py .

RUN pip install --no-cache-dir requests==2.26.0

CMD ["python", "produce.py"]
