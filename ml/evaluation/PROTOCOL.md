# Retrieval evaluation protocol

Before publishing quality numbers, curate a small licensed video set and label query relevance independently of model output. Store videos outside Git. A JSON manifest should include source/license, local video identifier, query, modality and all relevant [start, end] intervals.

Include objects, settings, speech-only concepts, mixed queries and negatives. Split development queries from held-out evaluation. Freeze model revision and sampling interval before evaluating held-out queries. A returned timestamp counts relevant if it falls inside a labeled interval; duplicate results inside the same interval count once for interval recall.

Report Recall@K over labeled intervals, Precision@K over returned moments, MRR over the first relevant hit, warm/cold search latency separately, processing and embedding time, vector bytes and machine/software configuration. Negative queries need a separate abstention evaluation: current top-K retrieval does not support calibrated rejection.

Synthetic color/shape fixtures are pipeline smoke checks only. They cannot establish real-world video retrieval quality. The measured synthetic baseline is in RESULTS.md; no real-video benchmark has been published.

## Runner and metrics

Run `python -m ml.evaluation.run --manifest path/to/manifest.json --k 5 --repeats 5` from the repository root. Follow synthetic.json: dataset name/license, videos with local paths, and query text/mode/intervals. Paths resolve from the working directory. Install the visual extra; speech/hybrid queries also require the speech extra. Models are real, and missing weights are downloaded. Media, database and embeddings go into isolated ignored data/benchmarks directories. Output defaults to data/benchmark.json.

Intervals are half-open [start, end), non-overlapping and nonnegative. Recall counts unique relevant intervals hit. Precision is unique relevant intervals hit divided by requested K, so duplicate hits and missing returned slots do not inflate it. MRR is the reciprocal rank of the first relevant hit within K. Negative queries have undefined recall/MRR and are summarized through negative-return rate.
