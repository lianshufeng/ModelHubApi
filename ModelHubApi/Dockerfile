FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    fonts-wqy-microhei \
    python3-pip \
    python3-dev \
    git \
    dos2unix  \
    && if [ ! -f /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi \
    && if [ ! -f /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi \
    && rm -rf /var/lib/apt/lists/*



COPY . /opt/ModelHubApi/ModelHubApi
WORKDIR /opt/ModelHubApi/ModelHubApi
RUN find ./ -type f -exec dos2unix {} \;


# 安装依赖
RUN bash install.cmd;

# 安装依赖库，暂时不支持awq, 版本依赖问题
RUN pip install flash-attn