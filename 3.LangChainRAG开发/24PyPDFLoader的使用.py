from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# 默认是page模式，每个页面形成一个Document文档对象，
loader = PyPDFLoader(
    file_path="./data/pdf2.pdf",
    # mode="page",
    password="itheima"
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("="*20, i)


# single模式，不管有多少页，只返回1个Document对象
# loader = PyPDFLoader("./data/pdf1.pdf")
# pages = loader.load()
# # 合并所有页面文本为单个文档
# all_text = "\n".join([p.page_content for p in pages])
# single_doc = Document(page_content=all_text, metadata={"source": "./data/pdf1.pdf"})
# print(single_doc)

