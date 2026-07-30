import os
import sys
import asyncio
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGEngine, PGVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage,SystemMessage
from langchain.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# 加载环境变量
load_dotenv()

# ── 全局初始化（模块加载时执行一次）:阿里云百炼配置 ─────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024

# 1. Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=DASHSCOPE_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    dimensions=EMBEDDING_DIMENSION,
    check_embedding_ctx_length=False
)
# 2. PGEngine
pg_engine = PGEngine.from_connection_string(
    url=f"postgresql+asyncpg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
)
# 3. PGVectorStore
vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name="faq_knowledge_base",
    embedding_service=embeddings,
    embedding_column="content_vector",
    content_column="combined_text",
    id_column="id",
    metadata_columns=["id", "question", "answer", "category"]
)
# 定义工具
@tool
def search_faq(query: str) -> str:
    """
    从FAQ知识库中检索与用户问题最相关的问答对。
    当用户询问产品、流程、规则、常见问题等时应调用此工具。

    Args:
        query: 用户的查询问题

    Returns:
        格式化后的相关问答文本，若无结果则返回提示信息
    """
    try:
        docs = vector_store.similarity_search_with_score(query, k=5)
        results = []
        for doc, score in docs:
            results.append(
                f"[相关度:{score:.2f}] Q:{doc.metadata['question']}\nA:{doc.metadata['answer']}"
            )
        return "\n\n".join(results) if results else "未找到相关信息"

    except Exception as e:
        return f"搜索出错：{str(e)}"

# 初始化 LLM
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
)

# 定义AI Agent的系统提示词
SYSTEM_PROMPT = """
# 角色设定
你是玲珑子，绿手指有机农场的 AI 客服专员。  
你性格温和、耐心细致，说话像一位熟悉农事、懂生活的朋友，既专业又接地气，擅长用通俗易懂的语言解释农业相关的问题。

# 服务范围
你可以为用户提供以下方面的咨询服务：
- 产品相关问题（品种、规格、口感、种植方式等）
- 发货规则（发货时间、地区限制、发货频率等）
- 运费规则（收费标准、包邮条件、偏远地区说明等）
- 活动规则（优惠活动、会员权益、积分使用等）

⚠️ 你**不能**处理以下内容，遇到时必须礼貌引导：
- 售后问题（退货、换货、赔偿、投诉等）
- 份额操作（开通、暂停、转让、退款等账户类操作）

# 工作习惯
- 收到用户提问后，**优先调用 search_faq 工具**，从知识库中查找准确信息。
- 回答前先在脑中整理：用户真正关心的是什么？有没有隐藏顾虑？
- 引用知识库内容时，用自己的话重新组织，避免生硬复制原文。
- 如果知识库没有明确答案，坦诚告知，并给出合理的建议或替代方案。

# 语言风格
- 称呼用户为“您”，语气亲切、自然，不过度客套。
- 适当使用生活化、农业相关的表达，例如“这批菜刚采摘”“地里现摘直发”等。
- 每次回答的最后不要主动提问，不要引导用户选择话题，等待用户输入后再回复。

# 示例语气参考
- “这款番茄是我们农场露天种植的，口感偏沙甜，很适合做沙拉。”
- “您所在的地区属于常规发货范围，一般下单后 48 小时内就能发出。”
- “关于售后问题我这边暂时帮不上忙，建议您直接联系人工客服，他们会更快为您处理～”

# 约束提醒
- 不编造信息，不确定就诚实说明。
- 不越权处理售后与份额问题。
- 始终保持友好、耐心的服务态度。
"""""
system_message = SystemMessage(content=SYSTEM_PROMPT)


# 创建 Agent
graph = create_agent(
    model=llm,
    tools=[search_faq],
    system_prompt=system_message,
)