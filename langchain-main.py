import bs4
from langchain_community.document_loaders import WebBaseLoader

# 1.Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# 2.拆分
# 文档长度超过 42k 个字符，太长了，无法适应许多模型的上下文窗口
# 我们将文档拆分为 1000 个字符的块，块之间有 200 个字符重叠。
# 重叠有助于减少将语句与与之相关的重要上下文分开的可能性
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 3.存储
# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型在单个命令中嵌入和存储所有文档拆分
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 选择一个适合中文和英文的模型（支持中英文，可自选）
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embedding其实是将每个单词或其他类型的标记（如字符、句子或者文档）转换为一个固定长度的向量
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

# 4.检索
# LangChain 定义了一个 Retriever 接口，该接口包装了一个索引，该索引可以返回给定 Documents 的字符串查询相关。
# "k": 6 表示对于每个查询，检索器应该返回最相似的前6个结果
# retriever 是一个检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# 执行
# retriever.invoke(问题) 手动式 手动一步步执行 适合调试 / 探索 / 临时验证
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")  # 可以再换个词检索，比如：Self-Reflection
# 输出打印
# print(retrieved_docs)

# 下面使用LC的链 实现一个小RAG
# rag_chain = {...}	链式自动流	自动串联数据流	适合完整的应用流程 / 自动执行 / 生产环境
# 5 加载llm模型
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
model_name = "Qwen/Qwen1.5-0.5B"  # 这里也可以换成别的开源模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).half().cuda()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# LangChain 封装
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# 使用一个 RAG 的提示，该提示已签入 LangChain 提示中心
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 使用 LCEL Runnable 协议来定义链  rag_chain 属于 LCEL（LangChain Expression Language）的表达式组合。
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # 字典
    | prompt  # 必须是在llm前面，用于接收字典
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)