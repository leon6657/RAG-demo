# RAG 知识库 + Agent + LangGraph 渐进式搭建方案

**日期：** 2026-07-02
**技术栈：** Python 3.12+ / LangChain / ChromaDB / DeepSeek API / BGE Embedding

---

## 一、项目定位与目标

面向 ByteDance 飞书项目 AI 实习岗位的简历项目。从最小 RAG 起步，逐步演进为
支持 Agent 的后端系统，最终引入 LangGraph 构建有状态的工作流引擎。

**核心产出：**
- 可演示的 RAG 知识问答系统
- 可作为面试素材的工程实践（代码质量、架构迭代、评测意识）

---

## 二、整体路线（4 个 Phase）

| Phase | 主题 | 内容 | 预计时长 |
|-------|------|------|---------|
| 1 | 最小 RAG | 文档加载 → 分块 → Embedding → 检索 → LLM 生成 | 1-2 周 |
| 2 | 优化 RAG | 重排序、查询改写、混合检索、评测集 | 1-2 周 |
| 3 | Agent | 知识库工具化、对话记忆、ReAct 循环 | 1-2 周 |
| 4 | LangGraph | 将 Agent 拆为有向状态图、条件路由 | 1 周 |
| 收尾 | 简历打磨 | README、项目叙事梳理 | 2-3 天 |

---

## 三、Phase 1 详细设计：最小 RAG

### 3.1 技术选型

- **LLM 生成层：** DeepSeek Chat API（`deepseek-chat`，OpenAI 兼容格式）
- **Embedding 层：** BAAI/bge-small-zh-v1.5（本地运行，sentence-transformers）
- **向量数据库：** ChromaDB（纯本地，SQLite 持久化）
- **框架：** LangChain（统一抽象 Document Loader / Text Splitter / Vector Store / Chain）
- **文档源：** 自写 Markdown 笔记（10-20 篇，覆盖 LLM/RAG/Agent 基础知识）

### 3.2 项目结构

```
ai-knowledge-base/
├── data/
│   ├── notes/             # Markdown 笔记（初始语料）
│   └── chroma_db/         # ChromaDB 持久化目录（自动生成）
├── src/
│   ├── __init__.py
│   ├── config.py          # 全局配置（模型名、分块参数、路径）
│   ├── ingest.py          # 文档加载 → 分块 → Embedding → 入库
│   ├── retrieve.py        # 查询 Embedding → Chroma 向量检索
│   ├── generate.py        # 检索结果 + 查询 → Prompt → LLM 回答
│   └── main.py            # CLI 入口（ingest / ask / chat 子命令）
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

### 3.3 数据流

```
[Markdown 文件] → TextLoader → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    → BGE Embedding → ChromaDB 持久化

                              ↓

[用户查询] → BGE Embedding → ChromaDB similarity_search (top_k=4)
    → 召回文本 + 查询 → RAG Prompt → DeepSeek Chat → 回答
```

### 3.4 各模块接口约定

**`config.py`**

```python
@dataclass
class Settings:
    notes_dir: str = "data/notes"
    chroma_dir: str = "data/chroma_db"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.3
```

**`ingest.py`**

```python
class IngestionPipeline:
    def load_markdown(self, notes_dir: str) -> list[Document]: ...
    def split_documents(self, docs: list[Document]) -> list[Document]: ...
    def embed_and_store(self, chunks: list[Document]): ...
    def run(self): ...
```

**`retrieve.py`**

```python
class Retriever:
    def retrieve(self, query: str) -> list[Document]: ...
```

**`generate.py`**

```python
class Generator:
    def answer(self, query: str, context: list[Document]) -> str: ...
```

**`main.py`**

```
命令：
  python src/main.py ingest       # 建库
  python src/main.py ask <query>  # 单次问答
  python src/main.py chat         # 交互式多轮
```

### 3.5 关键技术决策

1. **分块策略：** 用 LangChain 的 `RecursiveCharacterTextSplitter`，按 Markdown 标题自然边界切分。500 token 对中文技术笔记合适。
2. **检索只做语义 Top-K（k=4）：** 不做重排和查询改写，这些留给 Phase 2。
3. **Prompt 模板：** 要求 LLM 基于上下文回答，上下文不足时明确说不知道。
4. **每次 ingest 全量重建：** MVP 阶段不设计增量更新，简单可靠。
5. **LLM 用 OpenAI 兼容接口：** 便于后续切换模型。

### 3.6 依赖清单（requirements.txt）

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.2.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
openai>=1.0.0
click>=8.0
```

---

## 四、Phase 2 至 Phase 4 概述

### Phase 2：优化 RAG

- 引入 **Cross-encoder 重排序**（`BAAI/bge-reranker-v2-m3`）
- **查询改写**（Multi-Query / HyDE）
- **混合检索**（BM25 + 向量语义检索的融合）
- **评测集**：准备 20-30 组 Q&A 对，做 Recall / MRR / 人工评分

### Phase 3：Agent

- 将知识库封装为 LangChain `Tool`
- 加入对话记忆（ConversationBufferMemory）
- 实现 ReAct 循环：思考 → 工具调用 → 观察 → 推理 → 回答
- Agent 可同时调用知识库 + 计算器 + 搜索等工具

### Phase 4：LangGraph

- 将 Agent 的 ReAct 循环拆为有向图：State + Node + Edge
- 加入条件路由（如判断是否需要搜索再回答）
- 有状态会话管理（多轮对话的上下文持久化）

---

## 五、面试叙事要点

项目完成后，你应能回答：

1. **RAG 的完整链路是什么？** 从文档加载到生成回答的每个环节
2. **为什么选 bge-small-zh 做 Embedding？** 中文场景表现、本地部署、成本
3. **分块大小如何确定？** 中文 500 token + 50 overlap 的考量
4. **LangChain 和 LangGraph 的关系？** Chain → AgentExecutor → StateGraph 的演进
5. **你可能遇到的失败场景？** 检索不到相关文档、LLM 幻觉、长文本截断
6. **如何评测 RAG 效果？** 构建评测集 → 跑 Recall/MRR → 人工抽样检查

---

*本设计文档是开发过程中的活文档，随项目推进持续更新。*
