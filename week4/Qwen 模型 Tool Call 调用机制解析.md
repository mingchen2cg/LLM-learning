### 第一部分：LLMs 工具调用（Tool Calling）核心机制全景笔记

根据提供的文档，大语言模型从“纯文本回复”走向“与外部世界交互”，核心依赖于一套五阶段的闭环机制。

#### 1. 工具调用的 5 个核心阶段
* **阶段一：触发阶段（意图识别与 Token 生成）**
    * 模型在对话中分析用户请求，判断是否需要外部信息（如实时数据、复杂计算等）。
    * 如果需要，模型会生成特定的控制 Token（如 Qwen 底层的 `<tool_call>`）来标识调用意图，相当于向外界发出“我需要帮助”的信号。
* **阶段二：格式受控阶段（JSON 参数生成）**
    * 在触发标记之后，模型必须按照预先定义的格式（通常是 JSON）输出需要调用的**函数名 (name)** 和 **参数 (arguments)**。
    * **受控生成技术**：为了防止模型输出损坏的 JSON，推理引擎（如 vLLM、TGI）常结合 Outlines 等库提供“引导解码（Guided Decoding）”，通过正则表达式或 JSON Schema 强制约束模型的输出字符，确保 100% 格式合法。
* **阶段三：执行阶段（外部调度器解析与调用）**
    * 这一步脱离了 LLM 内部，由开发者编写的 **调度器 (Orchestrator)** 接管。
    * 调度器通过正则提取或解析模型生成的 JSON 文本。
    * 根据 `"name"` 映射到本地的实际代码逻辑（Python 函数），并传入 `"arguments"` 进行实际的 API 调用、数据库查询或业务逻辑执行。
* **阶段四：响应整合阶段（结果包装与反馈）**
    * 调度器拿到外部执行结果（如航班列表、天气数据）后，将其序列化为字符串（通常也是 JSON）。
    * 将结果包装在特定的标签内（如 `<tool_response>`），并**作为新的上下文追加到对话历史中**。
* **阶段五：再次交互阶段（最终答案生成）**
    * 将包含“用户原始请求 + 模型的工具调用请求 + 调度器注入的工具执行结果”的完整对话历史再次提交给模型。
    * 模型阅读这些“真实数据”后，将其融合成自然语言，向用户输出最终的答复。此时，工具调用的痕迹对用户是透明的。

#### 2. 核心架构设计与 Prompt 规范
* **工具声明 (Schema Definition)**：必须在 System Prompt 中告诉模型有哪些可用工具。目前业界标准是采用 JSON Schema 格式，明确包含每个函数的 `name`（名称）、`description`（功能描述）和 `parameters`（参数名、类型、是否必填）。
* **对话流转形态**：
    从用户的视角看：`提出问题 -> 得到最终答案`。
    从系统后台视角看，实际发生的是：`User提问 -> Assistant请求调用工具 -> 系统执行工具 -> 系统提交工具结果 -> Assistant生成最终回答`。

---


### 第二部分：可直接运行的工具调用 Demo 代码

为了更贴近实际生产环境，我们通常不再手动去拼接 `<tool_call>` 或正则解析，而是使用主流的 **OpenAI 兼容 API 格式**。目前的阿里云百炼（DashScope）提供的 Qwen 模型已完美支持这一标准。

您只需填入您的 `API_KEY`，运行以下 Python 代码，即可体验完整的闭环流程。

**依赖安装：**
```bash
pip install openai
```

**演示代码：**

```python
import json
from openai import OpenAI

# ==========================================
# 1. 初始化客户端 (请填入您的 API KEY)
# ==========================================
# 这里以阿里云 DashScope 的兼容 OpenAI 接口为例，调用通义千问模型
client = OpenAI(
    api_key="YOUR_DASHSCOPE_API_KEY", # <--- 请在此处填入您的 API KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==========================================
# 2. 定义本地工具函数 (外部 Orchestrator 逻辑)
# ==========================================
def search_flights(origin: str, destination: str, date: str):
    """模拟查询航班信息"""
    print(f"🔧 [执行工具]: 查询从 {origin} 到 {destination}，日期 {date} 的航班...")
    # 模拟外部 API 返回结果
    return {
        "flights": [
            {"flight_no": "CZ3101", "depart": "07:00", "arrive": "09:20", "price": 550},
            {"flight_no": "MU5103", "depart": "10:30", "arrive": "12:45", "price": 680}
        ]
    }

def search_hotels(city: str, date: str):
    """模拟查询酒店信息"""
    print(f"🔧 [执行工具]: 查询 {city}，日期 {date} 的酒店推荐...")
    return {
        "hotels": [
            {"hotel_name": "上海外滩大酒店", "location": "外滩", "price": 800},
            {"hotel_name": "上海静安香格里拉", "location": "静安寺", "price": 1200}
        ]
    }

# 构建一个函数映射字典，方便通过名字调用
available_functions = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
}

# ==========================================
# 3. 定义工具 Schema 描述给 LLM (JSON Schema 标准)
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "查找指定出发地、目的地和日期的航班信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "出发城市或机场（三字码）"},
                    "destination": {"type": "string", "description": "抵达城市或机场（三字码）"},
                    "date": {"type": "string", "description": "出发日期，格式 YYYY-MM-DD"}
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "搜索指定城市在指定日期的酒店选项。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "目标城市名称"},
                    "date": {"type": "string", "description": "入住日期，格式 YYYY-MM-DD"}
                },
                "required": ["city", "date"]
            }
        }
    }
]

# ==========================================
# 4. 主流程：多轮交互闭环
# ==========================================
def run_travel_assistant():
    # 用户初始问题
    user_prompt = "请帮我查找7月20日北京飞上海的航班，并推荐上海市中心的酒店。"
    print(f"🧑‍💻 [用户]: {user_prompt}\n")

    messages = [{"role": "user", "content": user_prompt}]

    # 第一轮调用：LLM 决定是否需要调用工具
    print("🤖 [模型思考中...]")
    response = client.chat.completions.create(
        model="qwen-plus", # 可根据您的权限修改为 qwen-turbo 或 qwen-max
        messages=messages,
        tools=tools,
        tool_choice="auto" 
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 检查模型是否决定调用工具
    if tool_calls:
        print(f"💡 [模型请求]: 检测到需要调用 {len(tool_calls)} 个工具。\n")
        
        # 将模型要求调用工具的这条消息，必须原封不动地加入上下文
        messages.append(response_message)

        # 遍历解析所有要求调用的工具，并执行本地代码
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # 根据名称找到本地映射的函数
            function_to_call = available_functions.get(function_name)
            
            if function_to_call:
                # 实际执行 Python 函数
                function_response = function_to_call(**function_args)
                
                # 将执行结果作为 "tool" 角色追加到消息列表中 (主流标准做法，非 user 角色)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(function_response, ensure_ascii=False)
                })

        # 第二轮调用：带着外部数据，让 LLM 生成最终的自然语言回答
        print("\n🤖 [数据已获取，模型生成最终回答...]")
        second_response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        
        final_answer = second_response.choices[0].message.content
        print(f"\n✨ [最终回答]:\n{final_answer}")
    else:
        # 如果不需要工具，直接输出回答
        print(f"\n✨ [最终回答]:\n{response_message.content}")

if __name__ == "__main__":
    run_travel_assistant()
```