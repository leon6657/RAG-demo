"""向量检索层（FAISS）"""

import logging
from typing import Optional

from langchain_community.vectorstores import FAISS

from src.config import Settings
from src._embeddings import create_embeddings

logger = logging.getLogger(__name__)


class Retriever:
    """封装查询向量化与 FAISS 检索"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = create_embeddings(settings.embedding_model)
        index_dir = str(settings.resolved_chroma_dir())
        self._vectorstore = FAISS.load_local(
            index_dir, self._embeddings, allow_dangerous_deserialization=True,
        )
        logger.info("Retriever initialized (%d vectors)", self._vectorstore.index.ntotal)

    def retrieve(self, query: str, k: Optional[int] = None) -> list:
        """检索与 query 最相似的 k 个文档块"""
        k = k or self.settings.top_k
        docs = self._vectorstore.similarity_search_with_score(query, k=k)
        logger.info("Retrieved %d docs for query: %.60s", len(docs), query)
        return docs
