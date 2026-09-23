# Semantic transcript retrieval and hierarchical evidence

SceneMind's V1 experiment uses `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The Apache-2.0 model has 22.7 million parameters and maps English sentences and short paragraphs to 384-dimensional vectors. It runs locally on CPU through Transformers and Torch; no embedding text is sent to a paid API.

Input is ordered Whisper transcript segments. Preprocessing groups adjacent segments into non-overlapping fine units capped at 30 seconds or 600 characters. Three adjacent fine units form each non-overlapping context unit. The encoder tokenizes at most 256 word pieces, mean-pools token embeddings using the attention mask, and L2-normalizes the result. FAISS inner-product search therefore acts as cosine similarity search.

Output indexes preserve unit ID, level, text, segment IDs, timestamps, and the fine IDs represented by a context unit. Index persistence uses FAISS serialization through Python bytes because FAISS filename APIs are unreliable with non-ASCII Windows paths. Metadata is stored separately as JSON. Model cache, indexes, transcripts, and video remain ignored local data.

At query time the same encoder produces one vector. The experiment retrieves ten fine units and five context units. A deterministic selector combines semantic fine rank, context membership, and a small BM25 rank contribution. It returns at most five fine units and 3,600 characters in chronological order. Context units support discovery only; citations always resolve to fine transcript timestamps.

The measured CPU cost was 516.99 MiB added peak RSS for encoder and validation indexes. Index construction for 53:14 of validation media took 2.807 seconds; semantic retrieval was 11.996/15.607 ms median/p95. The approach is operationally practical, but the frozen quality validation failed because long repetitive talks still confused section selection.

Alternatives rejected for V1 include paid embedding APIs, an external vector database, a learned reranker, unrestricted LLM retrieval, overlapping context expansion, and larger embedding models. They add cost or complexity before this simple architecture proves product value.
