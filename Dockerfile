# syntax=docker/dockerfile:1
FROM python:3.9.6-buster

WORKDIR /home/
RUN cd /home

# install packages
COPY ./api ./api
RUN pip install -r api/requirements.txt
RUN rm -rf /root/.cache/pip

RUN mkdir ./models
COPY ./models/franklin_at_fincham ./models/franklin_at_fincham
COPY ./models/collingwood_below_alma ./models/collingwood_below_alma


WORKDIR /home/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8050"]
#CMD ["gunicorn","--workers=10", "--threads=1", "-b 0.0.0.0:8050","-t 600", "app:server"]


# build this using docker build -t river_forecaster:0.0.7 .
