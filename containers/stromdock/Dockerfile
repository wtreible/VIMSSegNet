FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
 && apt-get install -y python3 \
 && apt-get install -y python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install setuptools \
 && pip3 install "numpy<1.18,>=1.17" \
 && pip3 install "scikit-image<0.17,>=0.15" \
 && pip3 install tensorflow-gpu==1.14 \
 && pip3 install Keras==2.3 \
 && pip3 install Pillow==6.2
