# model_base.py

from pydantic import BaseModel
from transformers import TextIteratorStreamer

from ModelHubApi.ApiModel import ChatRequest


class BaseModelHandler(BaseModel):

    def load_model(self):
        """加载模型（子类实现）"""
        pass

    def chat(self, chat_request: ChatRequest)->TextIteratorStreamer:
        """处理 chat 请求（子类实现）"""
        pass
