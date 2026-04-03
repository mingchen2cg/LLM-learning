# vllm_openai_chat_completions.py
from openai import OpenAI

openai_api_key = "sk-xxx"  # 随便填写，只是为了通过接口参数校验
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_outputs = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Llama-8B",
    messages=[
        {"role": "user", "content": "什么是深度学习？ "},
    ],
)
print(chat_outputs)
