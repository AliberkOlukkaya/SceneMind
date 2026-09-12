# Vector retrieval

Input: normalized frame vectors and one query vector. Output: top-K frame indices and cosine scores, mapped back to exact sampled timestamps and thumbnails. Implementation: backend/app/encoder.py (rank_vectors) and backend/app/visual.py (index lifecycle and API).

FAISS IndexFlatIP performs exact inner-product search. Since vectors have unit length, inner product equals cosine similarity. We normalize both paths, reject non-finite/zero vectors and check dimensions. This is exact retrieval over sampled frames, not over every source frame.

Vectors persist as an atomic NumPy array with pickle loading disabled. FAISS reconstructs its small exact index from the matrix per query; there are at most roughly 1,800 samples under current duration/interval limits. Approximate indexes, a persistent index cache or pgvector are premature until measurements justify them. Changing the model revision invalidates retrieval until reindexing.

K defaults to 10, bounded at 50. Score ordering is meaningful within a query; scores are not probabilities and always return nearest neighbors even if none is relevant. Evaluate with timestamp relevance judgments, Recall@K and MRR instead of relying on appealing demos. See ml/evaluation/PROTOCOL.md.
