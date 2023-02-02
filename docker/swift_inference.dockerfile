FROM ubuntu:22.04

RUN apt update && apt install -y ssh zip rsync vim openjdk-17-jdk openssh-client wget git htop python3-pip wormhole && mkdir /SWIFT

COPY . /SWIFT
WORKDIR /SWIFT
RUN pip3 install -r requirements.txt
RUN pip3 install torchserve torch-model-archiver torch-workflow-archiver nvgpu
RUN rm -rf serve/model_store/* serve/model_store/ serve/models.zip logs/
RUN zip -r serve/models.zip model/ model_zoo/ && mkdir serve/model_store
RUN torch-model-archiver --model-name swift --version 1.0 --model-file model/SWIFT.py --handler serve/handler.py --extra-files serve/models.zip
RUN mv swift.mar serve/model_store/

EXPOSE 8080 8081
CMD [ "torchserve", "--start", "--model-store", "serve/model_store/", "--models","swift=swift.mar", "--ts-config","serve/config/config.properties", "--ncs"]