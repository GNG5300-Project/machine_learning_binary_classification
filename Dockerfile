FROM python:3.10.14 

COPY  requirements.txt .

RUN pip install -r requirements.txt