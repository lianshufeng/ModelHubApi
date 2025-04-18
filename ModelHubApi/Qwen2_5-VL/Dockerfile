FROM lianshufeng/model_api:base


# 修改文件编码
RUN find ./ -type f -exec dos2unix {} \;

# 设置工作目录
WORKDIR /opt/ModelHubApi/Qwen2_5-VL


# 安装依赖
RUN bash install.cmd;
