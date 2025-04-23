from langchain import PromptTemplate

template = """ You are a naming consultant for new companies. What is a good name for a company that makes {product}? """

prompt = PromptTemplate.from_template(template)
prompt.format(product="colorful socks")

########################################################################

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(openai_api_key="...", temperature=0, model="gpt-3.5-turbo")

# HumanMessage 表示用户输入的消息，
# AIMessage 表示系统回复用户的消息，
# SystemMessage 表示设置的 AI 应该遵循的目标，
# ChatMessage 表示任务角色的消息。
messages = [SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Hi AI, how are you today?"), AIMessage(content="I'm great thank you. How can I help you?"), HumanMessage(content="I'd like to understand string theory.")]

res = chat(messages)
print(res.content)

########################################################################

from langchain.document_loaders import TextLoader

loader = TextLoader("./index.md")
loader.load()

########################################################################


class Chain(BaseModel, ABC):
    """Base interface that all chains should implement."""

    memory: BaseMemory
    callbacks: Callbacks

    def __call__(
        self,
        inputs: Any,
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
    ) -> Dict[str, Any]: ...


########################################################################

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

########################################################################

from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.9)  # 创建LLM实例
prompt = "用户的问题"  # 设置用户的问题

# 创建LLMChain实例
chain = LLMChain(llm=llm, prompt=prompt)

# 调用LLMs生成回复
response = chain.generate()

print(response)  # 打印生成的回复

########################################################################

from langchain.prompts import ChatPromptTemplate

# 创建一个空的ChatPromptTemplate实例
template = ChatPromptTemplate()

# 添加聊天消息提示
template.add_message("system", "You are a helpful AI bot.")
template.add_message("human", "Hello, how are you doing?")
template.add_message("ai", "I'm doing well, thanks!")
template.add_message("human", "What is your name?")

# 修改提示模板
template.set_message_content(0, "You are a helpful AI assistant.")
template.set_message_content(3, "What is your name? Please tell me.")

# 格式化聊天消息
messages = template.format_messages()

print(messages)

########################################################################

from langchain.chains import Chain
from langchain.components import Component1, Component2, Component3

# 创建组件实例
component1 = Component1()
component2 = Component2()
component3 = Component3()

# 创建Chain实例并添加组件
chain = Chain()
chain.add_component(component1)
chain.add_component(component2)
chain.add_component(component3)

# 处理下游任务
output = chain.process_downstream_task()

print(output)

########################################################################

from langchain.embeddings import Embedding
from langchain.vectorstore import VectorStore

# 创建Embedding实例
embedding = Embedding()

# 将文本嵌入到向量空间中
embedding.embed("Hello, world!")

# 创建VectorStore实例
vector_store = VectorStore()

# 存储嵌入向量
vector_store.store("hello", embedding.get_embedding())

# 检索嵌入向量
vector = vector_store.retrieve("hello")

print(vector)

########################################################################

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 从本地读取相关数据
loader = DirectoryLoader("./Langchain/KnowledgeBase/", glob="**/*.pdf", show_progress=True)

docs = loader.load()

# 将文件进行切分
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs_split = text_splitter.split_documents(docs)

# 初始化 OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# 将数据存入 Chroma 向量存储
vector_store = Chroma.from_documents(docs, embeddings)

# 初始化检索器，使用向量存储
retriever = vector_store.as_retriever()
system_template = """ Use the following pieces of context to answer the users question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answering these questions in Chinese. 
----------
{question}
----------
{chat_history}
"""

# 构建初始 Messages 列表
messages = [SystemMessagePromptTemplate.from_template(system_template), HumanMessagePromptTemplate.from_template("{question}")]

# 初始化 Prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)

# 初始化大语言模型，使用 OpenAI API
llm = ChatOpenAI(temperature=0.1, max_tokens=2048)

# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=prompt)

chat_history = []
while True:
    question = input("问题：")
    # 开始发送问题 chat_history 为必须参数, 用于存储对话历史
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(result["answer"])
