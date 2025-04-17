# 请求体
from typing import Any, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int



class ChatRequest(BaseModel):
    messages: Any = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
            ],
        }
    ]

    # 是否流式输出
    stream: bool = False

    # 最大生成 token 数
    max_new_tokens: int = 128

    # 温度采样值：控制输出的随机性，越低越稳定，越高越发散
    temperature: float = 0.8

    # nucleus sampling（核采样）：保留累计概率不超过 top_p 的 token 作为候选
    top_p: float = 0.95

    # 从概率最高的前 top_k 个词中采样。top_k=1 表示贪婪解码
    top_k: int = 50


# class ChatResponse(BaseModel):
#     class Choice(BaseModel):
#         text: str
#
#     choices: list = None


class ChatResponse(BaseModel):
    id: str=None
    object: str = "chat.completion"
    created: int=None
    model: str=None
    choices: List[Choice]=None
    usage: Optional[Usage] = None