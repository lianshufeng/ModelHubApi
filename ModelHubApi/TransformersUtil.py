from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config

config: Config = load_config()


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_strings: list[str], tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.current_output = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 解码当前输出
        self.current_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 检查是否包含 stop 字符串
        for stop_str in self.stop_strings:
            if stop_str in self.current_output:
                return True
        return False


# 构建停止策略
def build_stopping_criteriaList(stop: list[str], tokenizer: Any) -> StoppingCriteriaList:
    return StoppingCriteriaList([
        StopOnTokens(stop_strings=stop, tokenizer=tokenizer)
    ])


# 文本流生成器
def build_text_iterator_streamer(tokenizer: Any) -> TextIteratorStreamer:
    return TextIteratorStreamer(
        tokenizer, timeout=config.max_time_out, skip_prompt=True, skip_special_tokens=True)
