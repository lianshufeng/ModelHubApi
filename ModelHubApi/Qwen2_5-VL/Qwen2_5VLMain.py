from typing import Any

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from ModelHubApi import start_server
from ModelHubApi.ApiModel import ChatRequest, ChatResponse
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

# 载入配置
config: Config = load_config()


class Qwen2_5VLModelHandler(BaseModelHandler):
    model: Any = None
    processor: Any = None

    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model, torch_dtype="auto", device_map="auto"
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(config.model)

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        messages = chat_request.messages
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

        # 其他参数
        other = chat_request.model_dump()
        other.pop("messages")
        other.pop("stream")
        # 删除value为None 的成员
        for key in list(other.keys()):
            if other[key] is None:
                other.pop(key)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **other)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

        return ChatResponse()


def build_model() -> BaseModelHandler:
    model = Qwen2_5VLModelHandler()
    model.load_model()
    return model


if __name__ == '__main__':
    start_server(build_model())
