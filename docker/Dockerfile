FROM tensorflow/tensorflow:1.14.0-gpu-py3

WORKDIR /

COPY . .

RUN ls -al
RUN chmod +x ./train.sh ./inference.sh
RUN pip install -r src/requirements.txt

