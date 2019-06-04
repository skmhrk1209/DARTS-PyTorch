FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt update -y && apt install -y python3 python3-pip
RUN pip3 install --upgrade pip3 && pip3 install torch networkx 
