FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt update && apt install python3 python3-pip
