# Phase 1: 最小 RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可工作的最小 RAG 知识库：读取 Markdown 笔记 → 分块 → Embedding → ChromaDB 存储 → 语义检索 → DeepSeek Chat 生成回答。

**Architecture:** LangChain 提供 Document Loader / Text Splitter / Vector Store 的抽象，ChromaDB 做本地持久化向量库，BGE 本地模型做 Embedding，DeepSeek Chat API 做生成层。CLI 驱动的 ingest/ask/chat 三种交互模式。

**Tech Stack:** Python 3.12+, LangChain 0.3+, ChromaDB 0.5+, sentence-transformers 3.0+, DeepSeek Chat API, BAAI/bge-small-zh-v1.5, Click CLI

## Global Constraints

- 所有 Embedding 使用本地 BGE 模型，不消耗 API 费用
- 所有 LLM 调用通过 DeepSeek Chat API（OpenAI 兼容格式）
- 项目路径：`C:\Users\dad\Documents\Codex\2026-07-02\w-xi\ai-knowledge-base\`
- Python >= 3.12
- 开发环境：Windows + PowerShell

---

### Task 1: 项目脚手架 + Config 模块

**Files:**
- Create: `ai-knowledge-base/requirements.txt`
- Create: `ai-knowledge-base/src/__init__.py`
- Create: `ai-knowledge-base/src/config.py`
- Create: `ai-knowledge-base/.gitignore`

**Interfaces:**
- Consumes: 无
- Produces: `from src.config import Settings` —— 全局配置 dataclass，所有模块用它初始化

- [ ] **Step 1: 创建 `requirements.txt`**

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.2.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
openai>=1.0.0
click>=8.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: 创建 `src/__init__.py`**（空文件）

- [ ] **Step 3: 创建 `.gitignore`**

```
.env
data/chroma_db/
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 4: 创建 `src/config.py`**

```python
"""应用全局配置"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # 路径
    notes_dir: Path = Path("data/notes")
    chroma_dir: Path = Path("data/chroma_db")

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # 分块参数
    chunk_size: int = 500
    chunk_overlap: int = 50

    # 检索参数
    top_k: int = 4

    # LLM
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.3

    # API Key（通过环境变量或 .env 文件注入）
    deepseek_api_key: str = ""

    def __post_init__(self):
        if not self.deepseek_api_key:
            self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    def project_root(self) -> Path:
        """返回项目根目录（src/ 的父目录）"""
        return Path(__file__).resolve().parent.parent

    def resolved_notes_dir(self) -> Path:
        return self.project_root() / self.notes_dir

    def resolved_chroma_dir(self) -> Path:
        return self.project_root() / self.chroma_dir
```

- [ ] **Step 5: 安装依赖并验证**

Run: `pip install -r requirements.txt`

Expected: 所有包安装成功

Run: `python -c "from src.config import Settings; s=Settings(); print(s.embedding_model)"`

Expected: 输出 `BAAI/bge-small-zh-v1.5`

- [ ] **Step 6: 提交**

```bash
git add requirements.txt src/__init__.py src/config.py .gitignore
git commit -m "feat: project scaffold and config module"
```

---

### Task 2: 初始语料 —— 自写 Markdown 知识笔记

**Files:**
- Create: `ai-knowledge-base/data/notes/01-llm-basics.md`
- Create: `ai-knowledge-base/data/notes/02-what-is-rag.md`
- Create: `ai-knowledge-base/data/notes/03-agent-intro.md`
- Create: `ai-knowledge-base/data/notes/04-embedding-models.md`
- Create: `ai-knowledge-base/data/notes/05-langchain-basics.md`

**Interfaces:**
- Consumes: 无（独立于代码）
- Produces: `data/notes/*.md` —— 初始 5 篇笔记，供 Ingestion 模块测试

- [ ] **Step 1: 创建 `data/notes/01-llm-basics.md`** —— LLM 基础概念

一篇 800-1200 字的中文 Markdown，覆盖：什么是 LLM、Transformer 核心、预训练与微调、参数规模的含义。标题用 `#` 和 `##` 层级。

- [ ] **Step 2: 创建 `data/notes/02-what-is-rag.md`** —— RAG 原理

覆盖：RAG 全链路（Indexing → Retrieval → Generation）、为什么需要 RAG（解决幻觉和知识过时）、朴素 RAG 与高级 RAG 的区别。

- [ ] **Step 3: 创建 `data/notes/03-agent-intro.md`** —— AI Agent 入门

覆盖：Agent 的定义、ReAct 模式（思考→行动→观察）、工具调用、Agent vs Chain 的区别。

- [ ] **Step 4: 创建 `data/notes/04-embedding-models.md`** —— Embedding 模型

覆盖：文本嵌入（Text Embedding）的原理、余弦相似度、BGE 系列模型、向量维度的意义。

- [ ] **Step 5: 创建 `data/notes/05-langchain-basics.md`** —— LangChain 基础

覆盖：LangChain 核心抽象（Document Loader, Text Splitter, Vector Store, Chain, Tool），链式调用 vs Agent 执行器。

- [ ] **Step 6: 提交**

```bash
git add data/notes/
git commit -m "docs: add sample markdown notes for RAG ingestion"
```

---

### Task 3: Ingestion Pipeline（文档加载 → 分块 → Embedding → 入库）

**Files:**
- Create: `ai-knowledge-base/src/ingest.py`

**Interfaces:**
- Consumes:
  - `from src.config import Settings`
  - `settings.resolved_notes_dir()` —— Markdown 目录路径
  - `settings.resolved_chroma_dir()` —— Chroma 持久化路径
- Produces:
  - `class IngestionPipeline.__init__(self, settings: Settings)`
  - `pipeline.run() -> Chroma` —— 全量重建 ChromaDB，返回向量库实例

- [ ] **Step 1: 下载 BGE Embedding 模型到本地缓存**

Run: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"`

Expected: 模型下载完成，无报错（首次下载约 100MB，会缓存在 `~/.cache/huggingface/`）

- [ ] **Step 2: 编写 `src/ingest.py`**

```python
"""文档加载、分块、向量化、入库"""

import logging
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

from src.config import Settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """将 Markdown 文档分块并存入 ChromaDB 向量库"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = HuggingFaceBgeEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def load_documents(self) -> list:
        """加载 notes_dir 下所有 .md 文件"""
        loader = DirectoryLoader(
            str(self.settings.resolved_notes_dir()),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        logger.info("Loaded %d documents", len(docs))
        return docs

    def split_documents(self, docs: list) -> list:
        """按递归字符分块器切分"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info("Split into %d chunks", len(chunks))
        return chunks

    def build_vectorstore(self, chunks: list) -> Chroma:
        """从 chunks 创建 ChromaDB 集合（全量重建）"""
        chroma_dir = self.settings.resolved_chroma_dir()
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            logger.info("Removed existing chroma db at %s", chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(chroma_dir),
        )
        logger.info(
            "Vectorstore created with %d vectors at %s",
            vectorstore._collection.count(),
            chroma_dir,
        )
        return vectorstore

    def run(self) -> Chroma:
        """完整执行 ingestion 流程，返回向量库实例"""
        docs = self.load_documents()
        chunks = self.split_documents(docs)
        return self.build_vectorstore(chunks)
```

- [ ] **Step 3: 手动验证 ingestion**

Run: `python -c "from src.config import Settings; from src.ingest import IngestionPipeline; p=IngestionPipeline(Settings()); p.run(); print('ingest OK')"`

Expected: 输出 "ingest OK"，`data/chroma_db/` 目录生成，有 `.parquet` 和 `chroma.sqlite3` 文件

- [ ] **Step 4: 提交**

```bash
git add src/ingest.py
git commit -m "feat: markdown ingestion pipeline with BGE embedding"
```

---

### Task 4: Retriever + Generator（检索与生成）

**Files:**
- Create: `ai-knowledge-base/src/retrieve.py`
- Create: `ai-knowledge-base/src/generate.py`

**Interfaces:**
- Consumes:
  - `from src.config import Settings`（含 `deepseek_api_key`）
  - `settings.resolved_chroma_dir()` —— Chroma 持久化路径
- Produces:
  - `class Retriever.__init__(self, settings: Settings)`
  - `retriever.retrieve(query: str, k: int | None = None) -> list[Document, float]`
  - `class Generator.__init__(self, settings: Settings)`
  - `generator.answer(query: str, context_docs: list[Document, float]) -> str`

- [ ] **Step 1: 编写 `src/retrieve.py`**

```python
"""向量检索层"""

import logging

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

from src.config import Settings

logger = logging.getLogger(__name__)


class Retriever:
    """封装查询向量化与 ChromaDB 检索"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = HuggingFaceBgeEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = Chroma(
            embedding_function=self._embeddings,
            persist_directory=str(settings.resolved_chroma_dir()),
        )
        logger.info(
            "Retriever initialized, collection size: %d",
            self._vectorstore._collection.count(),
        )

    def retrieve(self, query: str, k: int | None = None) -> list:
        """检索与 query 最相似的 k 个文档块，返回 [(Document, float)]"""
        k = k or self.settings.top_k
        docs = self._vectorstore.similarity_search_with_score(query, k=k)
        logger.info("Retrieved %d docs for query: %.60s", len(docs), query)
        return docs
```

- [ ] **Step 2: 编写 `src/generate.py`**

```python
"""LLM 生成层：基于检索结果构造 Prompt 并调用 LLM"""

import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.config import Settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是一个基于知识库的技术问答助手。

请根据以下上下文信息回答用户的问题。
上下文信息中包含了相关的技术笔记内容。
如果上下文中没有足够的信息来回答问题，请明确告诉用户你不知道，不要编造答案。
用中文回答，保持回答简洁、准确、有技术深度。"""


class Generator:
    """基于检索上下文 + 用户问题调用 LLM 生成回答"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.deepseek_api_key,
            openai_api_base=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )

    def answer(self, query: str, context_docs: list) -> str:
        """生成回答"""
        context_parts = []
        for i, (doc, score) in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[文档 {i}]（来源：{source}，相似度：{score:.4f}）\n{doc.page_content}"
            )
        context_str = "\n\n".join(context_parts)
        user_message = f"""上下文信息：
{context_str}

用户问题：{query}"""

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        logger.info(
            "Sending prompt to %s (context: %d docs)",
            self.settings.llm_model,
            len(context_docs),
        )
        response = self._llm.invoke(messages)
        return response.content
```

- [ ] **Step 3: 创建 `.env`**

创建 `ai-knowledge-base/.env`（不要提交到 git，已在 `.gitignore` 中排除）：

```
DEEPSEEK_API_KEY=sk-你的DeepSeek API密钥
```

- [ ] **Step 4: 手动验证检索 + 生成**

Run: `python -c "from src.config import Settings; from src.retrieve import Retriever; r=Retriever(Settings()); docs=r.retrieve('什么是RAG', k=2); print('retrieved', len(docs), 'docs')"`

Expected: 输出 "retrieved 2 docs"

Run: `python -c "from src.config import Settings; from src.retrieve import Retriever; from src.generate import Generator; r=Retriever(Settings()); g=Generator(Settings()); docs=r.retrieve('RAG解决了什么问题', k=3); print(g.answer('RAG解决了什么问题', docs))"`

Expected: 基于笔记内容生成一段有意义的回答（不是空或报错）

- [ ] **Step 5: 提交**

```bash
git add src/retrieve.py src/generate.py
git commit -m "feat: retriever and generator modules"
```

---

### Task 5: CLI 入口 —— ingest / ask / chat 命令

**Files:**
- Create: `ai-knowledge-base/src/main.py`

**Interfaces:**
- Consumes:
  - `IngestionPipeline(settings).run()`
  - `Retriever(settings).retrieve(query, k)`
  - `Generator(settings).answer(query, context_docs)`
- Produces: 三个 CLI 命令：
  - `python src/main.py ingest` —— 重建知识库
  - `python src/main.py ask "问题"` —— 单次问答
  - `python src/main.py chat` —— 交互式多轮

- [ ] **Step 1: 编写 `src/main.py`**

```python
"""CLI 入口：ingest / ask / chat 命令"""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from src.config import Settings
from src.ingest import IngestionPipeline
from src.retrieve import Retriever
from src.generate import Generator

# 加载 .env 文件中的环境变量
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """AI 知识库 —— 基于 RAG 的技术问答系统"""


@cli.command()
def ingest():
    """读取 Markdown 笔记并重建向量知识库"""
    settings = Settings()
    pipeline = IngestionPipeline(settings)
    pipeline.run()
    click.echo("知识库构建完成！")


@cli.command()
@click.argument("query")
def ask(query: str):
    """单次问答：检索并生成回答"""
    settings = Settings()
    retriever = Retriever(settings)
    generator = Generator(settings)

    with_spans = retriever.retrieve(query)
    click.echo(f"\n{'='*60}")
    click.echo(f"问题：{query}")
    click.echo(f"{'='*60}\n")
    answer = generator.answer(query, with_spans)
    click.echo(answer)
    click.echo(f"\n{'='*60}")
    click.echo("来源文档：")
    for i, (doc, score) in enumerate(with_spans, 1):
        source = doc.metadata.get("source", "unknown")
        click.echo(f"  {i}. {Path(source).name} (相似度: {score:.4f})")


@cli.command()
def chat():
    """交互式多轮问答"""
    settings = Settings()
    retriever = Retriever(settings)
    generator = Generator(settings)

    click.echo("进入交互模式（输入 exit 退出）\n")
    while True:
        query = click.prompt("问题", prompt_suffix="> ")
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query.strip():
            continue

        with_spans = retriever.retrieve(query)
        click.echo()
        answer = generator.answer(query, with_spans)
        click.echo(answer)
        click.echo()


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: 手动验证 CLI**

Run: `cd ai-knowledge-base && python src/main.py ingest`
Expected: "知识库构建完成！"

Run: `python src/main.py ask "RAG是什么"`
Expected: 打印问题、回答、来源文档列表

Run: `python src/main.py chat`
Expected: 进入交互模式，输入问题后获得回答，输入 exit 退出

- [ ] **Step 3: 提交**

```bash
git add src/main.py
git commit -m "feat: CLI with ingest/ask/chat commands"
```

---

### Task 6: 集成测试 + README

**Files:**
- Create: `ai-knowledge-base/tests/__init__.py`
- Create: `ai-knowledge-base/tests/test_rag.py`
- Create: `ai-knowledge-base/README.md`

- [ ] **Step 1: 创建 `tests/__init__.py`**（空文件）

- [ ] **Step 2: 编写 `tests/test_rag.py`**

```python
"""Phase 1 RAG 集成测试"""

import os
from pathlib import Path

import pytest

from src.config import Settings
from src.ingest import IngestionPipeline
from src.retrieve import Retriever
from src.generate import Generator


@pytest.fixture
def settings():
    return Settings()


def test_ingestion_creates_chroma_db(settings):
    """Ingestion 后 ChromaDB 目录应存在且非空"""
    pipeline = IngestionPipeline(settings)
    pipeline.run()
    chroma_dir = settings.resolved_chroma_dir()
    assert chroma_dir.exists(), "ChromaDB 目录未创建"
    assert any(chroma_dir.iterdir()), "ChromaDB 目录为空"


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="需要 DEEPSEEK_API_KEY 环境变量",
)
def test_retrieve_returns_docs(settings):
    """检索应返回带分数的文档列表"""
    pipeline = IngestionPipeline(settings)
    pipeline.run()
    retriever = Retriever(settings)
    results = retriever.retrieve("什么是RAG", k=2)
    assert len(results) == 2, f"期望 2 个结果，得到 {len(results)}"
    for doc, score in results:
        assert doc.page_content, "文档内容为空"
        assert isinstance(score, float), "score 应为 float"


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="需要 DEEPSEEK_API_KEY 环境变量",
)
def test_generate_returns_answer(settings):
    """生成器应返回有意义的回答"""
    pipeline = IngestionPipeline(settings)
    pipeline.run()
    retriever = Retriever(settings)
    generator = Generator(settings)
    docs = retriever.retrieve("RAG解决了什么问题", k=3)
    answer = generator.answer("RAG解决了什么问题", docs)
    assert isinstance(answer, str), "回答应为字符串"
    assert len(answer) > 10, "回答过短"
```

- [ ] **Step 3: 安装 pytest 并设置环境变量**

Run: `pip install pytest`

设置 DeepSeek API Key：

```powershell
$env:DEEPSEEK_API_KEY="sk-你的key"
```

Run: `cd ai-knowledge-base && pytest tests/ -v`

Expected:
- `test_ingestion_creates_chroma_db`: PASS（无需 API key）
- `test_retrieve_returns_docs`: PASS（无需 API key）
- `test_generate_returns_answer`: PASS 或 SKIPPED（无 key 时跳过）

- [ ] **Step 4: 编写 `README.md`**

内容要点（使用 Markdown）：
- 项目简介：RAG 知识问答系统，面向 ByteDance 等 AI 实习岗位的简历项目
- 技术栈：Python / LangChain / ChromaDB / BGE Embedding / DeepSeek Chat
- 快速开始：
  1. 设置 `DEEPSEEK_API_KEY` 环境变量
  2. `pip install -r requirements.txt`
  3. `python src/main.py ingest`
  4. `python src/main.py ask "你的问题"`
- 命令参考：`ingest` / `ask` / `chat`
- 项目结构图
- 路线图：Phase 1-4 各阶段目标链接到设计文档
- 设计文档：`docs/2026-07-02-rag-knowledge-base-design.md`

- [ ] **Step 5: 提交**

```bash
git add tests/ README.md
git commit -m "feat: integration tests and README"
```

---

## 自检

- [ ] 所有文件名和路径与实际项目结构一致
- [ ] 每段代码可直接复制使用，无占位符
- [ ] 接口签名在前后任务中一致（deepseek_api_key 在 Task 1 config 中已定义，Task 4 Generator 引用它）
- [ ] 依赖清单完整
- [ ] 验证命令可执行且输出可预期
- [ ] 测试覆盖了 ingestion、retrieval、generation 三条核心链路
- [ ] 无 key 时 LLM 相关测试自动跳过，不会失败
