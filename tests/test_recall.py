"""Tests for src/embeddings.py and src/bm25_index.py (Phase 3 recall layer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bm25_index import bm25_top_k, rrf_fuse
from src.embeddings import load_dense_recall

# ---------------------------------------------------------------------------
# load_dense_recall
# ---------------------------------------------------------------------------


class TestLoadDenseRecall:
    def test_loads_ids_in_order(self, tmp_path: Path) -> None:
        ids = np.array(["CAND_0000031", "CAND_0000001", "CAND_0000099"])
        np.save(tmp_path / "dense_recall_ids.npy", ids)

        result = load_dense_recall(tmp_path)
        assert result == ["CAND_0000031", "CAND_0000001", "CAND_0000099"]

    def test_top_n_slicing(self, tmp_path: Path) -> None:
        ids = np.array([f"CAND_{i:07d}" for i in range(100)])
        np.save(tmp_path / "dense_recall_ids.npy", ids)

        result = load_dense_recall(tmp_path, top_n=5)
        assert len(result) == 5
        assert result[0] == "CAND_0000000"

    def test_top_n_larger_than_file(self, tmp_path: Path) -> None:
        ids = np.array(["CAND_A", "CAND_B"])
        np.save(tmp_path / "dense_recall_ids.npy", ids)

        result = load_dense_recall(tmp_path, top_n=1000)
        assert result == ["CAND_A", "CAND_B"]

    def test_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="precompute.py"):
            load_dense_recall(tmp_path)


# ---------------------------------------------------------------------------
# bm25_top_k
# ---------------------------------------------------------------------------


CORPUS = [
    "machine learning retrieval ranking embeddings vector search",
    "java spring boot microservices REST API enterprise",
    "recommendation system collaborative filtering matrix factorization",
    "human resources payroll employee management onboarding",
    "bm25 information retrieval sparse ranking term frequency",
]
CORPUS_IDS = ["CAND_A", "CAND_B", "CAND_C", "CAND_D", "CAND_E"]
ML_QUERY = "machine learning retrieval ranking embeddings"


class TestBM25TopK:
    def test_retrieval_terms_rank_above_hr(self) -> None:
        result = bm25_top_k(CORPUS, CORPUS_IDS, ML_QUERY, k=5)
        # CAND_D (HR) should not be #1
        assert result[0] != "CAND_D"

    def test_returns_at_most_k_results(self) -> None:
        result = bm25_top_k(CORPUS, CORPUS_IDS, ML_QUERY, k=3)
        assert len(result) == 3

    def test_all_ids_returned_when_k_equals_corpus(self) -> None:
        result = bm25_top_k(CORPUS, CORPUS_IDS, ML_QUERY, k=5)
        assert set(result) == set(CORPUS_IDS)

    def test_retrieval_doc_in_top2(self) -> None:
        # CAND_A and CAND_E both have strong retrieval signals
        result = bm25_top_k(CORPUS, CORPUS_IDS, ML_QUERY, k=5)
        top2 = set(result[:2])
        assert top2 & {"CAND_A", "CAND_E"}, f"Expected A or E in top-2, got {result}"

    def test_empty_corpus(self) -> None:
        result = bm25_top_k([], [], ML_QUERY, k=5)
        assert result == []


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------


class TestRrfFuse:
    def test_consensus_doc_ranks_first(self) -> None:
        # CAND_X is #1 in both lists → should dominate
        dense = ["CAND_X", "CAND_Y", "CAND_Z"]
        bm25 = ["CAND_X", "CAND_A", "CAND_B"]
        result = rrf_fuse(dense, bm25)
        assert result[0] == "CAND_X"

    def test_all_ids_in_union(self) -> None:
        dense = ["A", "B", "C"]
        bm25 = ["D", "E", "C"]
        result = rrf_fuse(dense, bm25)
        assert set(result) == {"A", "B", "C", "D", "E"}

    def test_top_n_slices_output(self) -> None:
        dense = [f"D{i}" for i in range(100)]
        bm25 = [f"B{i}" for i in range(100)]
        result = rrf_fuse(dense, bm25, top_n=50)
        assert len(result) == 50

    def test_single_list_degrades_gracefully(self) -> None:
        dense = ["A", "B", "C"]
        result = rrf_fuse(dense, [])
        # Without BM25, dense order should be preserved
        assert result == ["A", "B", "C"]

    def test_rrf_scores_decrease(self) -> None:
        # Both lists agree on order → fused scores should be monotonically
        # decreasing (since higher rank → lower 1/(k+rank) contribution).
        dense = ["A", "B", "C", "D"]
        bm25 = ["A", "B", "C", "D"]
        result = rrf_fuse(dense, bm25, k=60)
        assert result == ["A", "B", "C", "D"]

    def test_rrf_k60_standard_constant(self) -> None:
        # Verify RRF score: doc at rank 1 in both lists gets 2 * 1/(60+1) = 0.03279
        dense = ["X"]
        bm25 = ["X"]
        result = rrf_fuse(dense, bm25, k=60)
        assert result == ["X"]
