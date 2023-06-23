FROM ubuntu:20.04

RUN apt-get update && apt-get install libsm6 libxext6 python3-pip  -y

ADD ./ ./

RUN pip install -r requirements.txt

WORKDIR ./

CMD ["python3","api.py"]

EXPOSE 8000