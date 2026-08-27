# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Overview

A LangGraph application implementing a Chinese-language RAG FAQ assistant ("玲珑子", an AI customer-service agent for Green Fingers Organic Farm). It is built on the `new-langgraph-project` template: a single prebuilt `create_agent` graph that answers product/shipping/fee/promotion questions by retrieving Q&A pairs from a PostgreSQL + pgvector knowledge base. LLM and embeddings are served via the DashScope (阿里云百炼) OpenAI-compatible API. All documentation is in `README.md`.

## Common Commands

Dependencies are managed with `uv` (`uv.lock` is committed). Create `.env` from `.env.example` and populate `DASHSCOPE_API_KEY` and `PG_USER`, `PG_PASSWORD`, `PG_HOST`, `PG_PORT`, `PG_DATABASE` before running anything — the agent requires them at import time.

- **Install dependencies**: `uv sync` (adds dev group with pytest, ruff, mypy, langgraph-cli). Alternative per README: `pip install -e . "langgraph-cli[inmem]"`.
- **Run the LangGraph dev server**: `langgraph dev` — serves the graph defined in `langgraph.json`, with hot reload; open LangGraph Studio to test.
- **Run unit tests**: `make test` (defaults to `tests/unit_tests/`) or `python -m pytest tests/unit_tests`.
- **Run a single test file**: `make test TEST_FILE=tests/unit_tests/test_configuration.py` or `python -m pytest tests/unit_tests/test_configuration.py::test_placeholder -v`.
- **Run integration tests**: `python -m pytest tests/integration_tests` — requires live Postgres + API key; the test is marked `anyio` (asyncio backend) and `langsmith`.
- **Lint**: `make lint` — runs `ruff check .`, `ruff format --diff`, isort check (`ruff check --select I`), and `mypy --strict` over the changed/configured files.
- **Format**: `make format` — `ruff format` + `ruff check --select I --fix`.
- **Spell check**: `make spell_check` / `make spell_fix` (codespell).

Ruff config lives in `pyproject.toml`: rules E, F, I, D (google docstring convention), D401, T201, UP; ignored: UP006, UP007, UP035, D417, E501. Docstring style is Google. Tests are excluded from D/UP rules.

## Architecture

The entire application is one module: `src/agent/graph.py`. `langgraph.json` maps the graph entrypoint `agent` to `./src/agent/graph.py:graph`, and `src/agent/__init__.py` re-exports it. There is no other application code.

### Import-time initialization (important)

Everything is constructed at module import time, not inside a graph node:

1. `load_dotenv()` loads `.env`.
2. `embeddings` — `OpenAIEmbeddings` pointing at DashScope's OpenAI-compatible endpoint, model `text-embedding-v4`, 1024 dimensions.
3. `pg_engine` — `PGEngine.from_connection_string` builds an `asyncpg` URL from `PG_USER`, `PG_PASSWORD`, `PG_HOST`, `PG_PORT`, `PG_DATABASE`.
4. `vector_store` — `PGVectorStore.create_sync` bound to table `faq_knowledge_base`, with embedding column `content_vector`, text column `combined_text`, id column `id`, and metadata columns `question`, `answer`, `category`.
5. `search_faq(query)` — a `@tool` decorator wrapping `vector_store.similarity_search_with_score(query, k=5)`, formatting results as `[相关度:score] Q:... A:...`; returns "未找到相关信息" on no hits and catches exceptions into an error string.
6. `llm` — `ChatOpenAI` with model `qwen-plus`, temperature 0.3, same DashScope endpoint.
7. `SYSTEM_PROMPT` — a Chinese prompt establishing the persona, the in-scope topics (products, shipping, fees, promotions), hard out-of-scope topics (after-sales returns/exchanges, share/account operations) that must be politely declined, and a rule to always call `search_faq` first.
8. `graph = create_agent(model=llm, tools=[search_faq], system_prompt=system_message)` — LangChain's prebuilt ReAct-style agent (from `langchain.agents`), not a hand-built `StateGraph`. It is a `Pregel` instance.

### Consequences for development

- Because of module-level side effects, importing `agent.graph` (which both test files do) immediately tries to construct embeddings, connect to Postgres, and read `DASHSCOPE_API_KEY`. Tests fail at import with missing-env or connection errors, not assertion errors. `.env` must be populated and Postgres reachable even for "unit" tests.
- The vector store is created synchronously (`create_sync`) even though the engine uses asyncpg; do not replace it with the async builder without also changing how the graph calls the tool.
- Tool/LLM/model names and the knowledge-base table are domain-specific; if you add a second tool or vector store, keep the same global-variable pattern or refactor the module into an initializer function first.
- `langgraph.json` enables the default checkpointer; thread state/history is managed by LangGraph Server. There is no custom persistence code in the repo.
- `pyproject.toml` packages the module under two names (`agent` and `langgraph.templates.agent`), both mapped to `src/agent`; imports inside the repo use the bare `agent` package name.

### Tests layout

- `tests/conftest.py` — session-scoped fixture setting the anyio backend to `asyncio`.
- `tests/unit_tests/test_configuration.py` — placeholder asserting `graph` is a `Pregel`.
- `tests/integration_tests/test_graph.py` — `test_agent_simple_passthrough`, marked `langsmith` and `anyio`, invokes `await graph.ainvoke(inputs)` with `{"changeme": "some_val"}`.

### Deployment note

`langgraph.json` also configures `image_distro: wolfi` and a `checkpointer`; the server runs via the LangGraph CLI. CI-style checks (ruff, mypy, pytest) are invoked through the `Makefile` targets above; there is no separate build step beyond `uv sync` / editable install.
