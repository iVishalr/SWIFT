FROM ubuntu:22.04

RUN apt update && apt install -y ssh rsync vim openjdk-17-jdk openssh-client wget git htop python3-pip wormhole && mkdir /SWIFT

COPY . /SWIFT
WORKDIR /SWIFT
RUN pip3 install -r requirements.txt
EXPOSE 6006