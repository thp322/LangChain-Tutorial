> 本文档为教程拆分章节，完整文档见 [LangChain-Tutorial.md](../LangChain-Tutorial.md)

## 三、OpenAI 库的基础使用

OpenAI 库是 OpenAI 官方推出的 python SDK，核心作用是让开发者能简单、高效地调用 OpenAI 的各类 API（如 GPT 聊天、DALL · E 绘图、语音转文字等），无需手动处理 HTTP 请求、身份验证等底层细节

由于其发布较早且比较易用，现如今许多模型服务商（如阿里云百炼平台）均兼容 OpenAI SDK 的调用

OpenAI 库的使用流程

- 获取客户端对象
- 调用模型
- 处理结果

### 获取客户端对象

```python
from openai import OpenAI

client:OpenAI = OpenAI(
	api_key = "......"
    base_url = "......"
)
```

### 调用模型

```python
from openai.types.chat.chat_completion import ChatCompletion

response:ChatCompletion = client.chat.completions.create(
	model = "qwen3-max"
    messages = [
        {"role": "system", "content": "你是一个 python 编程专家。"}，
        {"role": "assistant", "content": "我是一个 python 编程专家。请问有什么可以帮助您的吗？"}，
        {"role": "user", "content": "for 循环输出 1 到 5 的数字。"}
    ]
)
```

`client.chat.completions.create` 创建 ChatCompletion 对象，主要参数有 2 个：

- `model`：选用模型
- `messages`：提供给模型的消息，类型为 list，可以包含多个字典消息，每个字典消息包含 2 个 key ，role 为角色，content 为内容

`role 角色`：

- system：设定助手的整体行为、角色和规则，为对话提供上下文框架（如指定助手身份、回答风格、核心要求），是全局的背景设定，影响后续所有交互
- assistant：代表 AI 助手的回答，可以在代码中人为设定
- user：代表用户，发送问题、指令或需求

### 处理结果

模型的回复为 response 变量，是一个 ChatCompletion 对象，其为类 json 格式，包含的信息如下：

```python
{
"id": "chatcmpl-xxxx",
"object": ,
"created": 17......,
"model": "gpt-3.5-turbo-0125",
"choices": [
    {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "生成的回复内容"
        },
        "finish_reason": "stop"  # stop=正常，length=令牌数超额，function_call=触发函数调用
    }
],
"usage": {
    "prompt_tokens": 50,
    "completion_tokens":80,
    "total_tokens": 130
}
}
```

可以通过如下输出模型给出的回答信息：

```python
print(response.choices[0].message.content)
```

### 完整代码

```python
from openai import OpenAI

# 1,获取 client 对象，OpenAI 类对象
client = OpenAI(
	# 设置了环境变量，这里不用再写 api_key
    base_url = "https://ws-mz1exca911jor8vw.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# 2,调用模型
response = client.chat.completions.create(
	model = "qwen3-max",
    messages = [
        {"role": "system", "content": "你是一个 python 编程专家。"}，
        {"role": "assistant", "content": "我是一个 python 编程专家。请问有什么可以帮助您的吗？"}，
        {"role": "user", "content": "for 循环输出 1 到 5 的数字。"}
    ]
)

# 3,处理结果
print(response.choices[0].message.content)
```

### 流式输出

可以设定结果输出为 stream 模式（流式输出），获得更好的使用体验

开启流式输出步骤：

- 在 `client.chat.completions.create()` 调用模型时设定参数：`stream = True`
- for 循环 response 对象，并在循环内输出内容

```python
from openai import OpenAI

# 1,获取 client 对象，OpenAI 类对象
client = OpenAI(
	# 设置了环境变量，这里不用再写 api_key
    base_url = "https://ws-mz1exca911jor8vw.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# 2,调用模型
response = client.chat.completions.create(
	model = "qwen3-max",
    messages = [
        {"role": "system", "content": "你是一个 python 编程专家。"}，
        {"role": "assistant", "content": "我是一个 python 编程专家。请问有什么可以帮助您的吗？"}，
        {"role": "user", "content": "for 循环输出 1 到 5 的数字。"}
    ],
    stream = True   # 开启流式输出
)

# 3,处理结果
# print(response.choices[0].message.content)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(
            chunk.choices[0].delta.content, 
            end="",     # 不要以回车符结尾
            flush=True   # 立刻刷新缓冲区
        )
```

### 附带历史消息

调用模型传入的参数 messages，其要求是 list 对象，即表明其支持非常多的消息在内

基于此，将历史消息填入，让模型知晓对话的上下文，更好的回答

```python
from openai import OpenAI

client = OpenAI(
    base_url = "https://ws-mz1exca911jor8vw.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
	model = "qwen3-max",
    messages = [
        {"role": "system", "content": "你是 AI 助理，回答很简洁"},
        {"role": "user", "content": "小明有 2 条宠物狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "小红有 3 只宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "总共有几个宠物？"}
    ],
    stream = True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

当前的历史消息是一次性的，如果是生产系统可以将消息保存到文件、数据库等持久化工具内，需要的时候提取使用
