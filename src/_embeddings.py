"""Embedding 工厂——使用本地 BGE 模型"""

import logging
from pathlib import Path

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

logger = logging.getLogger(__name__)


def create_embeddings(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """创建 BGE Embedding 实例，优先使用本地 models/bge 目录"""
    local_dir = Path(__file__).resolve().parent.parent / "models" / "bge"
    if local_dir.exists():
        path = str(local_dir)
        logger.info("Loading BGE from local: %s", path)
        return HuggingFaceBgeEmbeddings(
            model_name=path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    logger.info("Loading BGE from hub: %s", model_name)
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
