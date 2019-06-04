FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt install python3
RUN apt install python3-pip
RUN pip3 install torch
RUN pip3 install networkx 
