FROM frolvlad/alpine-miniconda3

MAINTAINER Tobias Sebastian Finn <tobias.sebastian.finn@uni-hamburg.de

RUN pwd
RUN ls
RUN apk add --no-cache git

RUN conda update -n base conda

RUN git clone https://gitlab.com/tobifinn/torch-assimilate.git
RUN conda env create -f /torch-assimilate/dev_environment.yml
SHELL ["/bin/bash", "-c"]
RUN source activate pytassim-dev && echo "Curr env: $CONDA_DEFAULT_ENV" && conda install -y pytorch-cpu torchvision-cpu -c pytorch
