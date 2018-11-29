FROM continuumio/miniconda3
MAINTAINER Tobias Sebastian Finn <tobias.sebastian.finn@uni-hamburg.de

RUN apt-get update -q -y && \
    apt-get install -y build-essential git graphviz

RUN TEMP_DEB="$(mktemp)" && \
    wget -O "$TEMP_DEB" 'http://ftp.br.debian.org/debian/pool/main/p/plantuml/plantuml_1.2018.9-1_all.deb' && \
    dpkg -i "$TEMP_DEB" || : && \
    apt-get install -f -y && \
    dpkg -i "$TEMP_DEB" && \
    rm -f "$TEMP_DEB"

RUN conda update -n base conda

RUN git clone https://gitlab.com/tobifinn/tf-assimilate.git

RUN conda env create -f /tf-assimilate/dev_environment.yml

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

RUN conda activate

RUN conda activate tfassim-dev

RUN conda install pytorch-cpu torchvision-cpu -c pytorch

RUN conda deactivate