FROM continuumio/miniconda3
MAINTAINER Tobias Sebastian Finn <tobias.sebastian.finn@uni-hamburg.de

RUN apt-get update -q -y && \
    apt-get install -y build-essential git graphviz

RUN TEMP_DEB="$(mktemp)" && \
    wget -O "$TEMP_DEB" 'http://ftp.br.debian.org/debian/pool/main/p/plantuml/plantuml_1.2017.15-1_all.deb' && \
    dpkg -i "$TEMP_DEB" || : && \
    apt-get install -f -y && \
    dpkg -i "$TEMP_DEB" && \
    rm -f "$TEMP_DEB"

RUN conda update -n base conda

RUN git clone https://gitlab.com/tobifinn/tf-assimilate.git

RUN conda env create -f /tf-assimilate/dev_environment.yml
