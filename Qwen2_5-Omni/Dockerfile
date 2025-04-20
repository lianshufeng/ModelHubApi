FROM lianshufeng/model_api:base


COPY . /opt/ModelHubApi/Qwen2_5-Omni

# 设置工作目录
WORKDIR /opt/ModelHubApi/Qwen2_5-Omni

# 修改文件编码
RUN find ./ -type f -exec dos2unix {} \;


# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# 安装依赖
RUN bash install.cmd;
