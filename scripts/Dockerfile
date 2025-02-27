ARG BASE_URL
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive SHELL=/bin/bash

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends \
    git \
    curl \
    bash && \
    rm -rf /var/lib/apt/lists/*

# Install pnpm
RUN curl -fsSL https://get.pnpm.io/install.sh | sh -

# Install statics
WORKDIR /root
RUN git clone https://github.com/megaease/easevoice-trainer-portal.git
RUN cd easevoice-trainer-portal && echo 'use-node-version=22.14.0' >> ~/.npmrc
RUN cd easevoice-trainer-portal && pnpm install && pnpm build

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime

# Install some basic utilities
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends \
    git \
    wget \
    curl \
    bash \
    libgl1 \
    software-properties-common \
    ffmpeg \
    openssh-server \
    zip \
    unzip \
    iputils-ping \
    libtcmalloc-minimal4 \
    net-tools \
    vim \
    p7zip-full && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

# Install python3.10
RUN apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

ENV PATH /root/.local/bin:/megacloud/workspace/.local/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# Install jupyterlab
RUN pip install -U ipympl==0.9.2 jupyterlab==3.4.8 matplotlib==3.6.1 ipywidgets jupyter-archive && \
    pip install -U jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension && \
    rm -rf ~/.cache/pip

# Install pytorch
RUN pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124 && \
    rm -rf ~/.cache/pip

WORKDIR /megaease
RUN git clone https://github.com/megaease/easevoice-trainer.git
COPY --from=builder /root/easevoice-trainer-portal/dist /megaease/easevoice-trainer/src/dist
RUN pip install -r /megaease/easevoice-trainer/requirements.txt && \
    pip install uv && \ rm -rf ~/.cache/pip

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

CMD ["/usr/local/bin/start.sh"]
