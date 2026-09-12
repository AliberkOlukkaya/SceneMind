# Natural Video Benchmark V2 protocol

Natural V2 is a frozen, diagnostic benchmark for the current SceneMind pipeline. It contains 47 annotations over five natural videos: two videos for calibration and three source-disjoint videos for held-out evaluation. The held-out split covers `OBJECT`, `SCENE`, `SPEECH`, `COMPOSITIONAL`, `ACTION_TEMPORAL`, and `NEGATIVE`. Some annotations run through more than one retrieval path, producing 51 held-out query-path trials.

The versioned manifest in `natural_v2.json` records each source page, direct download, license, attribution, published SHA-1, locally measured SHA-256, exact source segment, preparation, split, source group, query text, half-open relevant intervals, required modality, evaluation modes, and annotation version. Schema validation rejects duplicate IDs, invalid intervals, incomplete held-out taxonomy, and source-group or checksum leakage across splits. Annotations were frozen before retrieval was run.

## Sources and preparation

| Video | Split | License and attribution | Segment and preparation |
| --- | --- | --- | --- |
| Gameplay of Send Me To Heaven | calibration | CC BY 3.0, petrsvar / Carrotpop, [source](https://commons.wikimedia.org/wiki/File:Gameplay_of_Send_Me_To_Heaven.webm) | Full 0–21.267 s source file; no transcode |
| Base jump | calibration | CC BY 3.0, Quest Films, [source](https://commons.wikimedia.org/wiki/File:Base_jump.webm) | Full 0–37 s source file; no transcode |
| Street traffic | held-out | CC BY 3.0, YouTube user Editor, [source](https://commons.wikimedia.org/wiki/File:Street_traffic.webm) | Full 0–35.004 s source file; no transcode |
| Throwing, slow motion | held-out | CC BY 4.0, Nesnad, [source](https://commons.wikimedia.org/wiki/File:Throwing-slowmotion-man2013-stabilized.webm) | Full 0–13.733 s source file; no transcode |
| Find link lightning talk | held-out | CC BY 3.0, Magnus Sälgö, [source](https://commons.wikimedia.org/wiki/File:Find_link_-_lightning_talk.webm) | Full 0–219.443 s source file; no transcode |

Media, decoded frames, transcripts, embeddings, and local databases are ignored by Git. Reproduce the artifacts from the repository root:

```powershell
.venv/Scripts/python scripts/prepare_natural_v2.py
.venv/Scripts/python -m ml.evaluation.run_natural_v2 --repeats 3
.venv/Scripts/python -m ml.evaluation.regression_natural_v2 data/natural-v2-report.json ml/evaluation/reports/natural-v2.json
```

The runner verifies every media checksum, uses the current API for upload/index/transcription/search, retrieves 50 candidates so filtering can precede R@1/3/5 measurement, and restores application settings afterward. It writes a complete JSON report, a compact query-path CSV, and Markdown under ignored `data/`. The committed files under `ml/evaluation/reports/` remain stable baselines and are never rewritten by the default command. Warm latency is the median of three in-process searches after one warm-up and includes API, text encoding, retrieval, and fusion. It is not network latency or throughput.

The visual cutoff is one floating-point step above the largest top visual score from a calibration negative. It is fitted only from calibration visual rows. The same frozen cutoff filters visual evidence on held-out visual and hybrid paths; positive BM25 speech evidence remains eligible. Speech uses its current natural zero-score abstention behavior. No held-out label changes model, query, threshold, or fusion parameters.

The set is deliberately small and source-diverse rather than statistically representative. Videos come from Wikimedia Commons, and unknown pretraining overlap remains possible. Treat category slices with few examples as failure probes, not population estimates.
