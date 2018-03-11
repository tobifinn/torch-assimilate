FROM continuumio/miniconda3

RUN apt-get update -q -y && \
    apt-get install -y build-essential git

RUN conda update -n base conda

RUN git clone https://gitlab.com/tobifinn/tf-assimilate.git

RUN conda env create -f /tf-assimilate/dev_environment.yml
