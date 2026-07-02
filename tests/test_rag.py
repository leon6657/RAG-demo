"""Phase 1 RAG 集成测试（FAISS + 本地 BGE）"""

import numpy as np
import pytest

from src.config import Settings


def test_config_defaults():
    s = Settings()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert s.top_k == 4
    assert s.llm_model == "deepseek-chat"


def test_config_paths():
    s = Settings()
    root = s.project_root()
    assert (root / "src" / "config.py").exists()
    assert s.resolved_notes_dir().name == "notes"


def test_ingestion_creates_index():
    import shutil
    from src.ingest import IngestionPipeline

    settings = Settings()
    index_dir = settings.resolved_chroma_dir()
    if index_dir.exists():
        shutil.rmtree(index_dir)

    pipeline = IngestionPipeline(settings)
    pipeline.run()
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
    shutil.rmtree(index_dir)


def test_retrieve_returns_docs():
    import shutil
    from src.ingest import IngestionPipeline
    from src.retrieve import Retriever

    settings = Settings()
    pipeline = IngestionPipeline(settings)
    pipeline.run()

    retriever = Retriever(settings)
    results = retriever.retrieve("RAG", k=2)
    assert len(results) == 2
    for doc, score in results:
        assert doc.page_content
        assert isinstance(score, (float, np.floating))

    shutil.rmtree(settings.resolved_chroma_dir())
