FROM continuumio/miniconda3
MAINTAINER Tobias Sebastian Finn <tobias.sebastian.finn@uni-hamburg.de

RUN apt-get update -q -y && \
    apt-get install -y build-essential git plantuml

RUN conda update -n base conda

RUN git clone https://gitlab.com/tobifinn/tf-assimilate.git

RUN conda env create -f /tf-assimilate/dev_environment.yml
