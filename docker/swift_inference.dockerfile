ARG image=pytorch/torchserve:latest
FROM ${image}
USER root
RUN apt update && apt install -y ssh zip rsync vim openjdk-17-jdk openssh-client wget git htop python3-pip wormhole && mkdir /SWIFT
COPY . /SWIFT
WORKDIR /SWIFT
RUN pip3 install -r requirements.txt
RUN pip3 install torchserve torch-model-archiver torch-workflow-archiver nvgpu
RUN zip -r serve/models.zip model/ model_zoo/ && mkdir serve/model_store
RUN torch-model-archiver --model-name swift --version 1.0 --model-file model/SWIFT.py --handler serve/handler.py --extra-files serve/models.zip
RUN mv swift.mar serve/model_store/
RUN echo "" >> serve/config/config.properties && echo "inference_address=http://0.0.0.0:8080" >> serve/config/config.properties
RUN echo "management_address=http://0.0.0.0:8081" >> serve/config/config.properties
RUN echo "metrics_address=http://0.0.0.0:8082" >> serve/config/config.properties
EXPOSE 8080 8081
WORKDIR /SWIFT/serve
CMD [ "torchserve", "--start", "--model-store", "model_store", "--models", "swift=swift.mar", "--ts-config", "config/config.properties", "--ncs"]