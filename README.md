# LangChain 教程：RAG 与 Agent 开发实战

## 📚 项目简介

本项目是一份系统性的 LangChain 开发教程，从零开始讲解大模型应用开发的核心技术栈。内容涵盖从大模型 API 基础调用、提示词工程、到 RAG（检索增强生成）知识库构建、Agent 智能体开发的完整链路。

**适用人群：** 具备 Python 基础，希望系统学习大模型应用开发的开发者。

## 🎯 教程特色

- **理论 + 实战：** 每个知识点都配有完整可运行的代码示例
- **双模型支持：** 同时覆盖云端大模型（阿里云通义千问）和本地模型（Ollama）
- **循序渐进：** 从基础 API 调用到复杂 Agent 开发，层层递进
- **配套资源：** 包含手写笔记图片、示例数据文件等辅助材料

## 📖 阅读方式

- � **完整文档（单文件）：** [LangChain-Tutorial.md](./LangChain-Tutorial.md)
- 📚 **分章节阅读：** 点击下方目录跳转对应章节

## 🧭 教程目录（点击跳转）

### 一、[RAG 与 Agent 概述](./chapters/01_RAG与Agent概述.md)
- 大模型的优缺点分析
- 企业核心需求
- RAG 与 Agent 的优势对比
- 开发框架选型：LangChain

### 二、[前置准备 - 大模型接入](./chapters/02_前置准备-大模型接入.md)
- 阿里云百炼平台配置与 API Key 获取
- 使用环境变量保护密钥
- 本地 Ollama 部署与代码调用

### 三、[OpenAI 库的基础使用](./chapters/03_OpenAI库基础使用.md)
- [代码示例：1.OpenAI库的基础使用/](./1.OpenAI库的基础使用/)
  - `01.APIKEY.py` - API Key 配置
  - `02.本地调用大模型.py` - Ollama 本地模型调用
  - `03.OpenAI库的基础使用.py` - 基础调用流程
  - `04.OpenAI库的流式输出.py` - 流式输出实现
  - `05.OpenAI库附带历史消息调用模型.py` - 多轮对话

### 四、[提示词工程](./chapters/04_提示词工程.md)
- 提示词优化技巧
- Zero-shot 与 Few-shot 思想
- [代码示例：2.提示词优化/](./2.提示词优化/)
  - `01提示词优化案例_金融文本分类.py` - Few-shot 分类任务
  - `02Json的基础使用.py` - JSON 数据格式处理
  - `03提示词优化案例_金融信息抽取.py` - 结构化信息抽取
  - `04提示词优化案例_金融文本匹配判断.py` - 文本匹配任务

### 五、[RAG 核心概念](./chapters/05_RAG核心概念.md)
- LangChain 框架简介
- RAG 工作原理与标准流程
- 向量基础概念与嵌入模型
- 余弦相似度算法实现

### 六、[RAG 开发实战](./chapters/06_RAG开发实战.md)
- [代码示例：3.LangChainRAG开发/](./3.LangChainRAG开发/)

#### 模型调用
- `02LangChain访问阿里云通义千问大模型.py` - 云端模型
- `03LangChain访问Ollama本地模型.py` - 本地模型
- `04LangChain的流式输出.py` - 流式输出
- `05LangChain调用聊天模型.py` - Chat Models
- `07LangChain消息的简写形式.py` - 消息简写
- `08LangChain访问阿里云嵌入模型.py` - 嵌入模型
- `09LangChain访问Ollama的本地嵌入模型.py` - 本地嵌入

#### 提示词模板
- `10通用提示词模板.py` - PromptTemplate
- `11FewShot提示词模板.py` - FewShotPromptTemplate
- `12模板类的format和invoke方法.py` - 方法对比
- `13ChatPromptTemplate的使用.py` - 对话模板

#### Chain 链构建
- `14Chain的基础使用.py` - | 管道链入门
- `16Runnable接口源码查看.py` - Runnable 接口
- `17StrOutputParser解析器.py` - 字符串解析
- `18JsonOutputParser解析器.py` - JSON 解析
- `19RunnableLambda的基础使用.py` - 自定义函数入链

#### 会话记忆
- `20临时会话记忆.py` - InMemoryChatMessageHistory
- `21长期会话记忆.py` - 文件持久化存储

#### 文档加载与处理
- `22CSVLoader的使用.py` - CSV 加载
- `23JSONLoader的使用.py` - JSON 加载
- `24PyPDFLoader的使用.py` - PDF 加载
- `25TextLoader和文档分割器.py` - 文本加载与分割

#### 向量存储与检索
- `26内存向量存储.py` - InMemoryVectorStore
- `27外部向量持久化存储.py` - Chroma 向量库
- `28向量检索构建提示词.py` - 检索增强
- `29RunnablePassthrough的使用.py` - RAG 链构建

### 七、[Agent 智能体开发](./chapters/07_Agent智能体开发.md)
- [代码示例：4.Agent智能体/](./4.Agent智能体/)
  - `01Agent智能体初体验.py` - Agent 创建与工具调用
  - `02Agent的stream流式输出.py` - Agent 流式输出
  - `03ReAct案例.py` - ReAct 行动框架
  - `04middleware中间件.py` - Agent 中间件

## 📁 项目结构

```
LangChain教程/
├── README.md                          # 项目说明文档（本文件）
├── LangChain-Tutorial.md              # 完整教程文档（单文件版）
├── chapters/                          # 拆分后的分章节文档
│   ├── 01_RAG与Agent概述.md
│   ├── 02_前置准备-大模型接入.md
│   ├── 03_OpenAI库基础使用.md
│   ├── 04_提示词工程.md
│   ├── 05_RAG核心概念.md
│   ├── 06_RAG开发实战.md
│   └── 07_Agent智能体开发.md
├── img/                               # 教程配图文件夹
│   ├── 1.png ~ 35.png                # 原理示意图、流程图
│   └── 1.jpg ~ 11.jpg                # 手写笔记图片
├── 1.OpenAI库的基础使用/               # 章节一：OpenAI 基础
│   ├── 01.APIKEY.py
│   ├── 02.本地调用大模型.py
│   ├── 03.OpenAI库的基础使用.py
│   ├── 04.OpenAI库的流式输出.py
│   └── 05.OpenAI库附带历史消息调用模型.py
├── 2.提示词优化/                       # 章节二：提示词工程
│   ├── 01提示词优化案例_金融文本分类.py
│   ├── 02Json的基础使用.py
│   ├── 03提示词优化案例_金融信息抽取.py
│   └── 04提示词优化案例_金融文本匹配判断.py
├── 3.LangChainRAG开发/                 # 章节三：LangChain RAG
│   ├── data/                          # 示例数据文件夹
│   │   ├── Python基础语法.txt
│   │   ├── info.csv / stu.csv
│   │   ├── pdf1.pdf / pdf2.pdf
│   │   └── *.json / stu_json_lines.json
│   └── 01~29.py                       # RAG 开发实战代码
└── 4.Agent智能体/                      # 章节四：Agent 开发
    ├── 01Agent智能体初体验.py
    ├── 02Agent的stream流式输出.py
    ├── 03ReAct案例.py
    └── 04middleware中间件.py
```

## 🚀 快速开始

### 环境要求

- **Python** >= 3.9
- **操作系统：** Windows / macOS / Linux

### 安装依赖

```bash
# 核心依赖
pip install openai
pip install langchain langchain-community langchain-ollama dashscope chromadb

# 文档处理依赖
pip install jq
pip install pypdf
pip install langchain_text_splitters
```

### 配置 API Key

#### 方式一：环境变量（推荐）

**Windows：** 系统属性 → 高级系统设置 → 环境变量 → 新建用户变量

| 变量名 | 变量值 | 用途 |
|--------|--------|------|
| `OPENAI_API_KEY` | 你的 API Key | openai 库调用 |
| `DASHSCOPE_API_KEY` | 你的 API Key | langchain 阿里云模型 |

**macOS / Linux：** 在 `.zshrc` 或 `.bashrc` 中添加

```bash
export OPENAI_API_KEY="你的API-Key"
export DASHSCOPE_API_KEY="你的API-Key"
```

配置完成后**重启终端**生效。

#### 方式二：代码中直接配置

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxx",  # 你的 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

### 运行示例代码

```bash
# 进入代码示例目录
cd 1.OpenAI库的基础使用

# 运行第一个示例
python 01.APIKEY.py
```

## 🔧 本地模型部署（可选）

如需离线运行，可使用 [Ollama](https://ollama.com/) 部署本地模型：

```bash
# 1. 下载安装 Ollama（官网下载）

# 2. 拉取模型（首次运行自动下载）
ollama run qwen3:4b              # 聊天模型
ollama run qwen3-embedding:4b    # 嵌入模型

# 3. 命令行交互
ollama run deepseek-r1:1.5b
```

本地模型调用时将 `base_url` 改为 `http://localhost:11434/v1` 即可。

## 📚 学习路线建议

```
第1步：阅读 LangChain-Tutorial.md 全文，建立整体认知
    ↓
第2步：完成 OpenAI 库基础使用（章节三），打通 API 调用
    ↓
第3步：练习提示词工程（章节四），掌握 Prompt 技巧
    ↓
第4步：攻克 LangChain RAG 开发（章节六），动手构建知识库
    ↓
第5步：学习 Agent 智能体（章节七），打造自动化助手
```

## 🛠 技术栈

| 技术 | 说明 |
|------|------|
| Python | 3.9+ |
| LangChain | LLM 应用开发框架 |
| LangGraph | Agent 执行引擎 |
| DashScope | 阿里云大模型 SDK |
| Ollama | 本地模型部署 |
| ChromaDB | 轻量级向量数据库 |
| PyPDF | PDF 文档解析 |

## ⚠️ 注意事项

1. **API Key 安全：** 切勿将 API Key 硬编码在代码中并提交到 Git 仓库，优先使用环境变量
2. **网络环境：** 阿里云百炼平台需要网络通畅，本地 Ollama 需确保服务已启动
3. **模型额度：** 云端模型调用会消耗 Token，注意监控用量避免超额
4. **依赖版本：** LangChain 生态更新较快，如遇 API 变更请参考官方文档适配

## 📄 手写笔记

教程末尾附有作者整理的手写笔记（共 11 张图片），方便快速回顾核心知识点，请在 [完整教程 LangChain-Tutorial.md](./LangChain-Tutorial.md) 底部查看。

## 🔗 相关资源

- [LangChain 官方文档](https://docs.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [阿里云百炼平台](https://bailian.console.aliyun.com/)
- [Ollama 官网](https://ollama.com/)
- [ChromaDB 官网](https://www.trychroma.com/)

---

**学习愉快！** 如果觉得本教程有帮助，欢迎 Star ⭐ 支持～
