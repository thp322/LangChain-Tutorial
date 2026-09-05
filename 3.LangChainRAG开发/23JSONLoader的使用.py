from langchain_community.document_loaders import JSONLoader

# stu.json
# loader = JSONLoader(
#     file_path="./data/stu.json",
#     jq_schema=".name",
#     # jq_schema=".other.addr",
#     # jq_schema=".",          # 整个数据
#     text_content=False,       # 告知JSONLoader 我抽取的内容不是字符串
# )
#
# document = loader.load()
# print(document)


# stus.json
# loader = JSONLoader(
#     file_path="./data/stus.json",
#     jq_schema=".[].name",
#     text_content=False,       # 告知JSONLoader 我抽取的内容不是字符串
#     # json_lines=True         # 告知JSONLoader 这是一个JSONLines文件（每一行都是一个独立的标准JSON）
# )
#
# document = loader.load()
# print(document)


# stu_json_lines.json
loader = JSONLoader(
    file_path="./data/stu_json_lines.json",
    jq_schema=".name",
    # jq_schema=".other.addr",
    # jq_schema=".",          # 整个数据
    text_content=False,     # 告知JSONLoader 我抽取的内容不是字符串
    json_lines=True         # 告知JSONLoader 这是一个JSONLines文件（每一行都是一个独立的标准JSON）
)

document = loader.load()
print(document)
