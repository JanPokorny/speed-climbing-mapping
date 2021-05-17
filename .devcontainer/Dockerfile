FROM tensorflow/tensorflow:2.4.1-gpu

# Disable interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        git \
        imagemagick \
        libgl1-mesa-glx \
        libopencv-dev \
        protobuf-compiler \
        python3-cairocffi \
        python3-lxml \
        python3-pil \
        python3-tk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow

# Copy and compile the model garden
COPY --chown=tensorflow ./tensorflow-models /home/tensorflow/models
WORKDIR /home/tensorflow/models/research/
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

# Install the API and the dependencies
COPY ./requirements.txt /home/tensorflow
RUN python -m pip install -U pip
RUN python -m pip install -r /home/tensorflow/requirements.txt

# Set log level
ENV TF_CPP_MIN_LOG_LEVEL 3
