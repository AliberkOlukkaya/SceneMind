# Retrieval evaluation protocol

Before publishing quality numbers, curate a small licensed video set and label query relevance independently of model output. Store videos outside Git. A JSON manifest should include source/license, local video identifier, query, modality and all relevant [start, end] intervals.

Include objects, settings, speech-only concepts, mixed queries and negatives. Split development queries from held-out evaluation. Freeze model revision and sampling interval before evaluating held-out queries. A returned timestamp counts relevant if it falls inside a labeled interval; duplicate results inside the same interval count once for interval recall.

Report Recall@K over labeled intervals, Precision@K over returned moments, MRR over the first relevant hit, warm/cold search latency separately, processing and embedding time, vector bytes and machine/software configuration. Negative queries need a separate abstention evaluation: current top-K retrieval does not support calibrated rejection.

Synthetic color/shape fixtures are pipeline smoke checks only. They cannot establish real-world video retrieval quality. No benchmark results have been published yet.
