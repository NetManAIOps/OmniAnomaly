# Base image
FROM tensorflow/tensorflow:1.15.0-gpu

# Resolves error with key
# See: https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# See: https://askubuntu.com/questions/1444943/nvidia-gpg-error-the-following-signatures-couldnt-be-verified-because-the-publi
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Update image contents to have latest python3 and pip3 for image
RUN apt-get update
RUN apt-get install -y python3-pip python3-dev vim
WORKDIR /usr/local/bin
RUN rm /usr/local/bin/python
RUN ln -s /usr/bin/python3 python
RUN pip3 install --upgrade pip
RUN apt-get install -y git curl zip unzip

# Create /app directory
WORKDIR /app

# Copy OmniAnomaly requirements into image
COPY ./requirements.txt /app

# Install OmniAnomaly requirements 
RUN pip3 install -r requirements.txt

# Set initial folder to be OmniAnomaly
WORKDIR /app/OmniAnomaly
