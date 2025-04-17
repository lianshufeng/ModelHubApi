# main.py

from fastapi import FastAPI

from ModelHubApi.ApiModel import ChatRequest, ChatResponse
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

app = FastAPI()

# 载入配置
config: Config = load_config()


class ApiServer():
    # 模型处理器
    model_handler: BaseModelHandler

    # 启动服务
    def start(self):
        import uvicorn
        uvicorn.run(app, host=config.host, port=config.port)
        pass


# 创建ApiServer
api = ApiServer()


# 启动服务
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    return api.model_handler.chat(request)


# 启动服务
def start_server(model_handler: BaseModelHandler):
    api.model_handler = model_handler
    api.start()
