# 添加到模块搜索路径
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer

from ModelHubApi import start_server, build_text_iterator_streamer, get_task_pool
from ModelHubApi.ApiModel import ChatRequest
from ModelHubApi.TransformersUtil import build_stopping_criteriaList
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

# 载入配置
config: Config = load_config()


# 清理chat_request
def clean_chat_request(chat_request: ChatRequest):
    other = chat_request.model_dump()
    other.pop("messages")
    other.pop("stream")
    other.pop("model")
    other.pop("stop")

    # 删除value为None 的成员
    for key in list(other.keys()):
        if other[key] is None:
            other.pop(key)
    return other


class Qwen2_5VLModelHandler(BaseModelHandler):
    model: Any = None
    processor: Any = None
    tokenizer: Any = None

    def load_model(self):
        # default model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation='flash_attention_2' if config.flash_attention else None,
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(config.model)

        # default tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)

    def chat(self, chat_request: ChatRequest) -> TextIteratorStreamer:
        # 获取messages
        messages = chat_request.messages

        # 处理停止符
        stopping_criteria = None
        if chat_request.stop is not None and chat_request.stop != '':
            stopping_criteria = build_stopping_criteriaList([chat_request.stop], self.tokenizer)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 过滤其他参数
        other = clean_chat_request(chat_request)

        streamer = build_text_iterator_streamer(self.tokenizer)

        gen_kwargs = {'streamer': streamer, 'stopping_criteria': stopping_criteria, **inputs, **other}

        get_task_pool().submit(self.model.generate, **gen_kwargs)

        return streamer


def build_model() -> BaseModelHandler:
    model = Qwen2_5VLModelHandler()
    model.load_model()
    return model


if __name__ == '__main__':
    start_server(build_model())
