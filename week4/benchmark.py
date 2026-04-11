#!/usr/bin/env python3
"""
benchmark.py – quick-n-dirty load-tester for your self-hosted LLM.

修改点：
- 移除了 OpenAI 的 tiktoken 依赖
- 接入了 Hugging Face 的 transformers 库
- 适配本地 DeepSeek/Llama 3 词表，使 Token 计算完全精准
"""

import asyncio, time, os, math, sys
import httpx
from transformers import AutoTokenizer

API_KEY = os.getenv("LLM_API_KEY", "YOUR_API_KEY")
BASE_URL = os.getenv(
    "LLM_BASE_URL", "http://127.0.0.1:8000"
)  # 默认改成了本地常用的 8000 端口，可自行修改
MODEL = os.getenv("LLM_MODEL", "my-model")  # 请确保这里与你推理引擎注册的模型名称一致

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

# 替换为你的本地模型路径
LOCAL_MODEL_PATH = "./models/DeepSeek-R1-Distill-Llama-8B-1"

# 加载本地 Llama-3 架构的 Tokenizer
try:
    print(f"正在加载本地 Tokenizer: {LOCAL_MODEL_PATH} ...")
    ENC = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    print("Tokenizer 加载成功！\n")
except Exception as e:
    print(
        f"Tokenizer 加载失败，请检查该目录下是否存在 tokenizer.json 和 tokenizer_config.json！\n错误: {e}"
    )
    sys.exit(1)


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
USER_BASE = "请用简体中文回答："


###############################################################################
# helpers
###############################################################################
async def chat(client, msg, max_tokens=None):
    payload = {
        "model": MODEL,
        "messages": [SYSTEM, {"role": "user", "content": msg}],
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    r = await client.post(ENDPOINT, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"], data["usage"]


def token_len(text):
    # 粗略 token 计数，add_special_tokens=False 避免重复计算 BOS/EOS
    return len(ENC.encode(text, add_special_tokens=False))


###############################################################################
# 1) & 2) 探测最大上下文 / 最大思维链长度
###############################################################################
async def find_limit(client, base_prompt, field="prompt"):
    step = 1024  # 每次递增 tokens
    max_tk = step
    while True:
        if field == "prompt":
            prompt = base_prompt * math.ceil(max_tk / token_len(base_prompt))
            try:
                # 截断时也使用字符截断，因为对并发请求拼接大量字符比较快
                await chat(client, prompt[: max_tk * 2])
                max_tk += step
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 413):  # 长度超限
                    return max_tk - step
                raise
        else:  # measure output limit
            try:
                _, usage = await chat(client, base_prompt, max_tokens=max_tk)
                if usage["completion_tokens"] < max_tk:
                    max_tk += step
                else:
                    return max_tk
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 413):
                    return max_tk - step
                raise


###############################################################################
# 3) 可配置 & 默认输出长度
###############################################################################
async def output_limits(client):
    # configurable
    max_cfg = await find_limit(
        client, USER_BASE + "输出任意内容直到被截断。", field="output"
    )
    # default
    content, usage = await chat(
        client, USER_BASE + "请输出尽可能长的文本，以测试默认最大长度。"
    )
    return max_cfg, usage["completion_tokens"]


###############################################################################
# 4) 吞吐量 (RPS, 单并发 60 s)
###############################################################################
async def throughput_1min(client):
    stop = time.time() + 60
    cnt = 0
    while time.time() < stop:
        await chat(client, USER_BASE + "返回 OK")
        cnt += 1
    return cnt / 60


###############################################################################
# 5) Token 生成速度 (tokens/sec)
###############################################################################
async def token_speed(client):
    t0 = time.time()
    content, usage = await chat(
        client, USER_BASE + "请输出不少于 256 个汉字。", max_tokens=512
    )
    dt = time.time() - t0
    return usage["completion_tokens"] / dt


###############################################################################
# 6) 最大并发
###############################################################################
async def max_concurrency(client):
    async def worker(idx):
        await chat(client, f"{USER_BASE}并发测试 #{idx}")

    low, high = 1, 128  # 自行调整上限
    while low < high:
        mid = (low + high + 1) // 2
        tasks = [worker(i) for i in range(mid)]
        try:
            await asyncio.gather(*tasks)
            low = mid
        except (httpx.HTTPStatusError, httpx.ReadTimeout):
            high = mid - 1
    return low


###############################################################################
async def main():
    async with httpx.AsyncClient() as client:
        ctx_max = await find_limit(client, USER_BASE + "填充")
        cot_max = await find_limit(
            client, USER_BASE + "你需要一步步展示思考链，直到被截断。", field="output"
        )
        out_cfg, out_def = await output_limits(client)
        rps = await throughput_1min(client)
        tps = await token_speed(client)
        conc_max = await max_concurrency(client)

    print("\n=== Benchmark Report ===")
    print(f"最大上下文长度             : {ctx_max} tokens")
    print(f"最大思维链内容长度         : {cot_max} tokens")
    print(f"可配置最大输出长度         : {out_cfg} tokens")
    print(f"默认最大输出长度           : {out_def} tokens")
    print(f"吞吐量 (1 并发, 60s 平均)  : {rps:.2f} requests / sec")
    print(f"token 生成速度             : {tps:.2f} tokens / sec")
    print(f"最大并发 (无错误)          : {conc_max} simultaneous requests")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
