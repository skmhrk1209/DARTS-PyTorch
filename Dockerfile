FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt install -y python3 python3-pip
RUN pip3 install torch networkx 
