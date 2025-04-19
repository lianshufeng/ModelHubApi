# 添加到模块搜索路径
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)
from ModelHubApi import start_server, build_text_iterator_streamer, get_task_pool, is_awq_model
from ModelHubApi.ApiModel import ChatRequest
from ModelHubApi.TransformersUtil import build_stopping_criteriaList
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

import torch

# 加载配置
config: Config = load_config()


# 清理 chat_request
def clean_chat_request(chat_request: ChatRequest):
    other = chat_request.model_dump()
    other.pop("messages", None)
    other.pop("stream", None)
    other.pop("model", None)
    other.pop("stop", None)

    # 删除值为 None 的参数
    for key in list(other.keys()):
        if other[key] is None:
            other.pop(key)
    return other


class Qwen2_5VLModelHandler(BaseModelHandler):
    model: Any = None
    processor: Any = None
    tokenizer: Any = None

    def load_model(self):
        model_path = config.model

        # 是否是 AWQ 模型：目录中包含 quant_config.json
        is_awq = is_awq_model(model_path)

        if is_awq:
            print("[INFO] Detected AWQ model. Loading with AutoAWQForCausalLM...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation='flash_attention_2' if config.flash_attention else None
            )
        else:
            print("[INFO] Loading standard model...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation='flash_attention_2' if config.flash_attention else None,
                trust_remote_code=True,
            )
            # 如果模型支持 tie_weights 且还没绑定
            if hasattr(self.model, "tie_weights"):
                self.model.tie_weights()

        # 加载 tokenizer 和 processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def chat(self, chat_request: ChatRequest) -> TextIteratorStreamer:
        messages = chat_request.messages

        stopping_criteria = None
        if chat_request.stop:
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
