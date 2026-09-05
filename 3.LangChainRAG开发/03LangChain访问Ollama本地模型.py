# langchain_ollama
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen")

res = model.invoke(input="你是谁呀能做什么？")

print(res)
