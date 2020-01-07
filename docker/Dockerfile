FROM tensorflow/tensorflow:1.14.0-gpu-py3
WORKDIR /
COPY . .
RUN apt install build-essential wget -y
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar -xzf ta-lib-0.4.0-src.tar.gz
WORKDIR /ta-lib
RUN ./configure --prefix=/usr
RUN make
RUN make install
RUN pip install TA-Lib 
WORKDIR /
RUN chmod +x ./train.sh ./inference.sh
RUN pip install -r src/requirements.txt