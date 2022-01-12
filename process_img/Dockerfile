FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get install -y gnupg wget
RUN apt-get install -y python3-pip software-properties-common 
RUN apt -y update
RUN add-apt-repository -y ppa:ubuntugis/ppa
RUN apt -y update
RUN apt install -y gdal-bin python3-gdal libspatialindex-dev
WORKDIR /pipelines/components/src
COPY requirements.txt /pipelines/components/src
RUN pip install -r requirements.txt
COPY process_img.py /pipelines/components/src
