# main.py
import concurrent
import json
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI
from starlette.responses import StreamingResponse, JSONResponse
from transformers import TextIteratorStreamer

from ModelHubApi.ApiModel import ChatRequest, ChatResponse, Usage, Choice, Message
from ModelHubApi.config import load_config
from ModelHubApi.config.Config import Config
from ModelHubApi.model import BaseModelHandler

app = FastAPI()

# 载入配置
config: Config = load_config()

#  线程池
task_pool: ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_task_count)


class ApiServer():
    # 模型处理器
    model_handler: BaseModelHandler

    # 启动服务
    def start(self):
        import uvicorn
        uvicorn.run(app, host=config.host, port=config.port)
        pass


# 创建ApiServer
api = ApiServer()



class OpenAIStyleSSEIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration

        try:
            chunk = next(self.iterator)
            # 这里包装成 OpenAI 风格的 JSON 格式
            data = {
                "choices": [
                    {
                        "delta": {"content": chunk}
                    }
                ]
            }
            # 返回流式数据
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except StopIteration:
            self.done = True
            # 在流结束时返回 [DONE]
            return "data: [DONE]\n\n"
        except Exception as e:
            self.done = True
            # 错误处理，返回错误信息
            error_data = {
                "error": str(e)
            }
            return f"data: {json.dumps(error_data)}\n\n"



# 启动服务
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    streamer: TextIteratorStreamer = api.model_handler.chat(request)
    iterator = iter(streamer)  # 无需判断是否是 str 或 list，直接迭代 streamer

    if request.stream:
        return StreamingResponse(
            OpenAIStyleSSEIterator(iterator),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        try:
            full_text = ''.join(iterator)
            response = ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,  # 你可以从 request 或 handler 提供
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=full_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=0,  # 可选：如果你能算
                    completion_tokens=len(full_text.split()),  # 简单估算
                    total_tokens=len(full_text.split())
                )
            )
            return response
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


# 提交任务
def get_task_pool() -> ProcessPoolExecutor:
    return task_pool


# 启动服务
def start_server(model_handler: BaseModelHandler):
    api.model_handler = model_handler
    api.start()
