"""文档加载、分块、向量化、入库（FAISS + 本地 BGE）"""

import logging
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

from src.config import Settings
from src._embeddings import create_embeddings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """将 Markdown 文档分块并存入 FAISS 向量索引"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = create_embeddings(settings.embedding_model)

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

    def build_vectorstore(self, chunks: list) -> FAISS:
        """从 chunks 创建 FAISS 索引并持久化"""
        index_dir = self.settings.resolved_chroma_dir()
        if index_dir.exists():
            shutil.rmtree(index_dir)
            logger.info("Removed existing index at %s", index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self._embeddings,
        )
        vectorstore.save_local(str(index_dir))
        logger.info(
            "FAISS index created with %d vectors at %s",
            vectorstore.index.ntotal,
            index_dir,
        )
        return vectorstore

    def run(self) -> FAISS:
        """完整执行 ingestion 流程，返回向量索引实例"""
        docs = self.load_documents()
        chunks = self.split_documents(docs)
        return self.build_vectorstore(chunks)
