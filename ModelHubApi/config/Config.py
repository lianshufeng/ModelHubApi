import argparse

from pydantic import BaseModel


class Config(BaseModel):
    port: int = 8080
    host: str = "0.0.0.0"
    model: str = None


# 定义并解析命令行参数。
def _parse_args():
    parser = argparse.ArgumentParser(description="Api服务配置")

    parser.add_argument("--port", type=int, default=8080, help="API 服务的端口号，默认是 8000")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听的 IP，默认是 0.0.0.0")
    parser.add_argument("--model", type=str, help="模型路径不能为空，如: meta-llama/Llama-2-7b-hf",
                        required=True)

    return parser.parse_args()


def load_config():
    args = _parse_args()
    return Config(**(args.__dict__))
