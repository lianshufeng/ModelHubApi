# 添加到模块搜索路径
import base64
import os
import re
import sys
from io import BytesIO
from urllib.request import urlopen

import librosa
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer, Qwen2AudioForConditionalGeneration,
)
from ModelHubApi import start_server, build_text_iterator_streamer, get_task_pool
from ModelHubApi.ApiModel import ChatRequest
from ModelHubApi.TransformersUtil import build_stopping_criteriaList
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

# 加载配置
config: Config = load_config()


def load_audio_input(audio_input, sampling_rate=16000) -> np.ndarray:
    # 判断是否是 base64（简单方式）
    if re.match(r'^data:audio/.+;base64,', audio_input):
        # 去掉头部
        header, base64_data = audio_input.split(',', 1)
        audio_bytes = base64.b64decode(base64_data)
        audio_stream = BytesIO(audio_bytes)
    elif audio_input.startswith('http://') or audio_input.startswith('https://'):
        audio_stream = BytesIO(urlopen(audio_input).read())
    else:
        raise ValueError("Unsupported input format. Must be a URL or base64 audio string.")

    # 加载音频数据
    waveform, _ = librosa.load(audio_stream, sr=sampling_rate)
    return waveform


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


def get_audio_data(ele) -> str | None:
    audio = ele.get('audio') or ele.get('audio_url') or ele.get('input_audio')
    if audio is None:
        return None
    if isinstance(audio, str):
        return audio
    elif hasattr(audio, 'data'):
        return audio.get('data')
    return None


class Qwen2_AudioModelHandler(BaseModelHandler):
    model: Any = None
    processor: Any = None
    tokenizer: Any = None

    def load_model(self):
        model_path = config.model

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=config.device,
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

        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] in ("audio", "audio_url"):
                        audio = get_audio_data(ele)
                        if audio is not None:
                            audios.append(
                                load_audio_input(audio)
                            )

        stopping_criteria = None
        if chat_request.stop:
            stopping_criteria = build_stopping_criteriaList([chat_request.stop], self.tokenizer)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 确保为数组
        if isinstance(text, list) is False:
            text = [text]

        # 如果没有音频，则设置为 None
        if len(audios) == 0:
            audios = None

        inputs = self.processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.to(self.model.device)

        other = clean_chat_request(chat_request)
        streamer = build_text_iterator_streamer(self.tokenizer)
        gen_kwargs = {'streamer': streamer, 'stopping_criteria': stopping_criteria, **inputs, **other}

        get_task_pool().submit(self.model.generate, **gen_kwargs)

        return streamer


def build_model() -> BaseModelHandler:
    model = Qwen2_AudioModelHandler()
    model.load_model()
    return model


if __name__ == '__main__':
    start_server(build_model())
