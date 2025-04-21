import argparse

from pydantic import BaseModel


class Config(BaseModel):
    port: int = 8080  # API 服务的端口号，默认是 8000
    host: str = "0.0.0.0"  # 服务监听的 IP，默认是 0.0.0.0
    model: str = None  # 模型路径，不能为空，如: meta-llama/Llama-2-7b-hf
    max_task_count: int = 1  # 最大并发数，默认是 1
    max_time_out: float = 180.0  # 最大超时时间，单位秒
    flash_attention: bool = False  # 默认使用 Flash Attention
    device: str = "auto"  # 推理设备


# 定义并解析命令行参数。
def _parse_args():
    parser = argparse.ArgumentParser(description="Api服务配置")

    parser.add_argument("--port", type=int, default=8080, help="API 服务的端口号，默认是 8000")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听的 IP，默认是 0.0.0.0")
    parser.add_argument("--model", type=str, help="模型路径不能为空，如: meta-llama/Llama-2-7b-hf",
                        required=True)
    parser.add_argument("--max_task_count", type=int, default=1, help="最大的并发数，默认是 1")
    parser.add_argument("--max_time_out", type=float, default=180.0, help="推理的超时时间，单位秒，默认是 60.0")

    parser.add_argument("--device", type=str, default="auto", help="推理设备,auto/cpu/cuda")

    parser.add_argument("--flash_attention", type=bool, default=False, help="是否使用 Flash Attention")

    return parser.parse_args()


def load_config():
    args = _parse_args()
    return Config(**(args.__dict__))
