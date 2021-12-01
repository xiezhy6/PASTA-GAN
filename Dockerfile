FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04 as base

FROM base as base-amd64

FROM base-amd64

FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install gcc -y
RUN pip3 install cython
RUN pip3 install torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install scikit-build click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
RUN pip3 install psutil scipy matplotlib opencv-python scikit-image pycocotools

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test
RUN echo 'dev:Password123!' | chpasswd

RUN apt-get install openssh-server -y
RUN service ssh start
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
