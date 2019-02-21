FROM continuumio/miniconda3

MAINTAINER Tobias Sebastian Finn <tobias.sebastian.finn@uni-hamburg.de

RUN apt-get update -q -y && \
    apt-get install -y build-essential git graphviz plantuml

SHELL ["/bin/bash", "-c"]
RUN conda update -n base conda
RUN git clone https://gitlab.com/tobifinn/torch-assimilate.git
RUN conda env create -f /torch-assimilate/dev_environment.yml
RUN source activate pytassim-dev && echo "Curr env: $CONDA_DEFAULT_ENV" && conda install -y pytorch-cpu torchvision-cpu -c pytorch
