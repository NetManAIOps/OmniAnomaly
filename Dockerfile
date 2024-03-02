FROM tensorflow/tensorflow:1.15.0-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev vim
WORKDIR /usr/local/bin
RUN rm /usr/local/bin/python
RUN ln -s /usr/bin/python3 python
RUN pip3 install --upgrade pip
RUN apt-get install -y git curl zip unzip

WORKDIR /app
WORKDIR OmniAnomaly

COPY ./requirements.txt /app

# Does this help split the command?
WORKDIR /app
RUN pip3 install -r requirements.txt

WORKDIR /app/OmniAnomaly
