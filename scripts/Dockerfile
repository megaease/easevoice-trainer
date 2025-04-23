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
RUN export SHELL=/bin/bash && curl -fsSL https://get.pnpm.io/install.sh | sh -

# Install statics
WORKDIR /root
RUN git clone https://github.com/megaease/easevoice-trainer-portal.git
RUN cd easevoice-trainer-portal && echo 'use-node-version=22.14.0' >> ~/.npmrc
RUN cd easevoice-trainer-portal && /root/.local/share/pnpm/pnpm install && /root/.local/share/pnpm/pnpm build

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive SHELL=/bin/bash

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

RUN pip install cmake && cmake --version
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

WORKDIR /megaease
RUN git clone https://github.com/megaease/easevoice-trainer.git
RUN pip install -r /megaease/easevoice-trainer/requirements.txt
RUN pip install httpx[socks] && \
    pip install pip-system-certs

RUN apt-get update --yes && \
    apt install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    apt-get install git-lfs -y && git lfs install && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /megaease/easevoice-trainer/models
RUN cd /megaease/easevoice-trainer/models && \
    git clone https://huggingface.co/lj1995/GPT-SoVITS && \
    git clone https://huggingface.co/Delik/uvr5_weights && \
    rm -rf /megaease/easevoice-trainer/models/pretrained && \
    mv /megaease/easevoice-trainer/models/GPT-SoVITS /megaease/easevoice-trainer/models/pretrained

RUN mkdir -p /megaease/easevoice-trainer/src/easevoice/text/data/chinese
RUN cd /megaease/easevoice-trainer/src/easevoice/text/data/chinese && \
    wget -O G2PWModel_1.1.zip https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/G2PWModel_1.1.zip && \
    unzip G2PWModel_1.1.zip && \
    mv G2PWModel_1.1 G2PWModel && \
    rm G2PWModel_1.1.zip

COPY --from=builder /root/easevoice-trainer-portal/dist /megaease/easevoice-trainer/dist

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 8000

CMD ["/usr/local/bin/start.sh"]
