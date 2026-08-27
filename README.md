# 玲珑子 · 绿手指有机农场 FAQ 智能客服

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的 RAG 问答 Agent，为「绿手指有机农场」提供 AI 客服能力。Agent 通过向量检索在 PostgreSQL/pgvector 知识库中查找 FAQ，并由大模型组织成自然、亲切的中文回答。

<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>

## 核心能力

- **FAQ 知识库检索**：内置 `search_faq` 工具，对用户问题做向量相似度检索（Top-5），返回带相关度得分的问答对。
- **大模型问答**：使用阿里云百炼（DashScope）`qwen-plus` 模型，将检索结果组织为符合农场客服人设的回答。
- **人设约束**：可回答产品、发货、运费、活动规则类问题；对售后问题与份额（账户）类操作，Agent 会礼貌引导用户联系人工客服，不做越权处理。

## 技术架构

- **Agent 框架**：LangGraph `create_agent`（`langchain.agents`），单图单工具，入口定义于 [src/agent/graph.py](./src/agent/graph.py)，由 `langgraph.json` 映射为 `agent` 图。
- **向量存储**：`langchain-postgres` 的 `PGVectorStore`，表 `faq_knowledge_base`。
- **Embedding**：DashScope `text-embedding-v4`（1024 维）。
- **模型**：DashScope `qwen-plus`（`temperature=0.3`）。
- **检查点**：使用 LangGraph 默认 checkpointer（见 `langgraph.json`），会话线程由 LangGraph Server 管理。

## 环境要求

- Python >= 3.10
- PostgreSQL（需启用 [pgvector](https://github.com/pgvector/pgvector) 扩展）
- 阿里云百炼（DashScope）API Key

### 知识库表结构

项目启动前需在 PostgreSQL 中准备好 FAQ 表（表名 `faq_knowledge_base`）：

| 列 | 说明 |
| --- | --- |
| `id` | 主键 |
| `question` | 问题文本 |
| `answer` | 答案文本 |
| `category` | 分类（产品 / 发货 / 运费 / 活动等） |
| `combined_text` | 检索用的组合文本（通常为问题+答案拼接） |
| `content_vector` | 1024 维向量列（`text-embedding-v4` 生成） |

数据写入与向量填充不在本仓库范围内，需在接入前通过脚本或后台任务完成。

## 快速开始

1. 安装依赖（使用 [uv](https://docs.astral.sh/uv/)）：

```bash
uv sync
```

2. 创建环境变量文件：

```bash
cp .env.example .env
```

3. 编辑 `.env`，填入必填项：

```text
DASHSCOPE_API_KEY=sk-xxx            # 阿里云百炼 API Key
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=your_database
```

可选：如需 LangSmith 链路追踪，添加 `LANGSMITH_API_KEY` 与 `LANGSMITH_PROJECT`。

4. 启动 LangGraph Server：

```bash
langgraph dev
```

启动后可在 [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) 中与 Agent 对话、调试工具调用，本地代码改动会自动热重载。

## 测试与代码质量

```bash
make test          # 运行单元测试（tests/unit_tests）
make test TEST_FILE=tests/integration_tests/test_graph.py   # 运行指定测试文件
python -m pytest tests/integration_tests                    # 集成测试（需环境变量与数据库）
make lint          # ruff 检查 + isort + mypy --strict
make format        # ruff format + isort 自动修复
```

注意：由于 `src/agent/graph.py` 在模块加载时会初始化 Embedding、Postgres 连接与模型客户端，任何测试（包括单元测试）都要求先配置好 `.env` 且数据库可达。

## 自定义与扩展

- **调整客服人设与话术**：修改 [graph.py](./src/agent/graph.py) 中的 `SYSTEM_PROMPT`。
- **更换模型 / Embedding**：修改 `llm` 与 `embeddings` 初始化（模型名、维度、`base_url`）。
- **新增工具**：在 `graph.py` 中用 `@tool` 定义新工具，并加入 `create_agent(tools=[...])`。
- **切换检索策略**：调整 `search_faq` 中 `similarity_search_with_score` 的 `k` 值，或替换为 `vector_store` 支持的混合检索方式。

## 相关文档

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph Server 本地部署](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)
- [langchain-postgres / PGVectorStore](https://python.langchain.com/docs/integrations/vectorstores/pgvector/)
