# 添加到模块搜索路径
import os
import sys

import torch
from qwen_omni_utils import process_mm_info
from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any
import soundfile as sf
from transformers import (
    TextIteratorStreamer, AutoTokenizer,
)
from ModelHubApi import start_server, build_text_iterator_streamer, get_task_pool
from ModelHubApi.ApiModel import ChatRequest
from ModelHubApi.TransformersUtil import build_stopping_criteriaList
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

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


class Qwen2_5OmniHandler(BaseModelHandler):
    model: Any = None
    processor: Any = None
    tokenizer: Any = None

    def load_model(self):
        model_path = config.model

        if config.flash_attention:
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path,
                                                                         torch_dtype=torch_dtype,
                                                                         device_map=config.device,
                                                                         attn_implementation='flash_attention_2' if config.flash_attention else None,
                                                                         )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        pass

    def chat(self, chat_request: ChatRequest) -> TextIteratorStreamer:
        messages = chat_request.messages

        stopping_criteria = None
        if chat_request.stop:
            stopping_criteria = build_stopping_criteriaList([chat_request.stop], self.tokenizer)

        # set use audio in video
        USE_AUDIO_IN_VIDEO = True

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                                padding=True,
                                use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        other = clean_chat_request(chat_request)
        streamer = build_text_iterator_streamer(self.tokenizer)
        gen_kwargs = {'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id, 'streamer': streamer,
                      'stopping_criteria': stopping_criteria,
                      **inputs, **other,
                      }

        # text_ids, audio = self.model.generate(**gen_kwargs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        # text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(text)
        # sf.write(
        #     "c:/output.wav",
        #     audio.reshape(-1).detach().cpu().numpy(),
        #     samplerate=24000,
        # )

        get_task_pool().submit(self.model.generate, **gen_kwargs)

        return streamer


def build_model() -> BaseModelHandler:
    model = Qwen2_5OmniHandler()
    model.load_model()
    return model


if __name__ == '__main__':
    start_server(build_model())
