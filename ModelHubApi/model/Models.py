# model_base.py

from pydantic import BaseModel

from ModelHubApi.ApiModel import ChatRequest, ChatResponse


class BaseModelHandler(BaseModel):

    def load_model(self):
        """加载模型（子类实现）"""
        pass

    def chat(self, chat_request: ChatRequest)->ChatResponse:
        """处理 chat 请求（子类实现）"""
        pass
