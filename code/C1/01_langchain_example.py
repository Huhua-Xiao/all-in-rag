import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# 默认参数 chunk_size=4000（块大小）和 chunk_overlap=200（块重叠）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=400)
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
print(answer.content)



# chunk size 和 overlap 参数可以根据具体需求进行调整，以优化文本分块效果。
# 比如，在默认的4000和200的基础上，更加接近主题理解内容和减少冗余信息。所以回答的answer更接近全文主题
# 在尝试更小的 chunk size 和 overlap 参数时，模型回答的方向发生了明显的改变。更加偏向于细节信息，而不是整体主题理解。
# 在尝试更大的 chunk size 和 overlap 参数时，模型回答的内容变得更加冗长，回答的信息更加的具体和详细了。原因是因为这个设置下，文本块包含了大部分的上下文信息。
# 通过实验发现，chunk size 和 overlap 参数对模型的回答质量和方向有显著影响。
# 但是chunk 大小不是越大越好，而是要根据任务目标和文本内容来进行合理设置。