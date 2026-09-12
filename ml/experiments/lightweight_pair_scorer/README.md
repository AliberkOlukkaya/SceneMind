# Lightweight pair scorer experiment

This isolated experiment asks whether a compact local scorer can verify CLIP's top five without changing production retrieval. It evaluates a pinned UForm3-small ONNX dual encoder and an eight-parameter logistic scorer over existing CLIP retrieval features. Neither is wired into the API.

## Evidence and leakage boundary

`calibration_v2.json` freezes two new source-disjoint videos and 20 reviewed queries before model inference. The phone animation contributes phone, key, bag, beside, behind and containment/holding negatives. The White House clip contributes a small cup, holding, person/helicopter and person/Marine relationships, plus drinking, transfer, phone and handshake negatives.

| Source | License and attribution | Immutable source |
| --- | --- | --- |
| `1 Me and my phone.webm`, complete 30.101 s file | CC BY 4.0; Derek J Moore for mLiteracy | Commons SHA-1 `c03f34d60659629c350b6a9e6798fb53d5210937`; SHA-256 `51302ba9475e5660785d928f3e306fae64553ebb682a66c7bbaee2520020cd6a` |
| `Latte salute.webm`, complete 6.571 s file | US government public domain; unidentified creator, released by the White House | Commons SHA-1 `cc4f2673a978a54b0ba7f4a8b23c07bae54b4954`; SHA-256 `7b339f9decba66c0c62f735dcbf36ec9151eab5b8ee858ae230d56679c12844f` |

The expansion is a child of the prior 30-query verifier calibration set and Natural V2. Both parent manifest hashes are stored. Evaluation fits on 66 calibration queries: 35 positive and 31 negative across seven videos. The 31 Natural V2 held-out queries across three other source groups remain byte-for-byte unchanged. Schema validation checks source groups and media hashes across all three collections. Thresholds, feature normalization and logistic weights use calibration rows only.

Source media, contact sheets, model files, embeddings and temporary databases remain under ignored `data/`. Reproduce the media with:

```powershell
.venv/Scripts/python -m ml.experiments.lightweight_pair_scorer.prepare
.venv/Scripts/python -m ml.experiments.lightweight_pair_scorer.prepare --verify-only
```

## Scorers

UForm uses `unum-cloud/uform3-image-text-english-small@a8d990ed0dc6340b9c25cb47f06f86b9ecb67fa6`. Image and text encoders emit normalized 256-dimensional embeddings; cosine similarity reranks only CLIP's five candidates. It is a dual encoder, so it does not jointly attend over a particular image/text pair and has no match/no-match logit. Its threshold 0.3302116394 is the lowest calibration-only cutoff permitting at most 10% calibration negative accepts.

The learned scorer uses seven fixed features for each CLIP candidate: cosine score, reciprocal rank, gap from top, top-two gap, query mean, standard deviation and range. A deterministic, class-balanced L2 logistic regression standardizes those features using calibration statistics. It has seven weights and one intercept, trains in about 4 ms, and uses threshold 0.6778145360. This tests a nonlinear probability boundary over retrieval context without training or changing CLIP.

## Run and validate

Install experiment-only packages; they are deliberately absent from production dependencies:

```powershell
.venv/Scripts/python -m pip install uform==3.1.3 onnxruntime==1.30.0 psutil==7.0.0
.venv/Scripts/python -m ml.experiments.lightweight_pair_scorer.evaluate --threads 4
.venv/Scripts/python -m ml.experiments.lightweight_pair_scorer.validate `
  ml/evaluation/reports/lightweight-pair-scorer-v1.json `
  ml/evaluation/reports/lightweight-learned-scorer-v1.json `
  ml/evaluation/reports/lightweight-uform-scorer-v1.json
.venv/Scripts/python -m ml.experiments.lightweight_pair_scorer.regression `
  ml/evaluation/reports/lightweight-pair-scorer-v1.json
```

The real evaluation downloads the pinned 60.6 MB ONNX pair, ingests ten local videos, builds CLIP indexes, and scores 97 queries. Unit tests mock or avoid expensive inference. The runtime sweep command in `benchmark_runtime.py` takes exactly five real JPEG paths and records median/p95 after three warmups.

Because both calibrated scorers failed the positive false-abstention gate, the protocol forbids a second held-out run. See [results](../../evaluation/LIGHTWEIGHT_PAIR_SCORER_RESULTS.md), [failures](../../evaluation/LIGHTWEIGHT_PAIR_SCORER_FAILURES.md), and the [shortlist](MODEL_SHORTLIST.md).

## Concepts and tradeoffs

A retriever encodes the query and all frames separately so it can search many frames cheaply. A reranker sees only a few retrieved candidates and may spend more work improving their order. SceneMind keeps CLIP as the retriever and limits this experiment to five candidates.

A dual encoder produces one vector per image and one per text, then compares them with cosine similarity. A cross encoder lets image and text tokens interact for each pair and may expose an explicit match/no-match logit, but normally costs much more. UForm is compact and shares some training layers, yet its deployed encoders remain separate; its cosine is not a match probability.

Calibration chooses a decision rule from development evidence. Abstention means returning no visual result when every score is below that rule. It can reduce false accepts, but the decision is useful only when it transfers to new sources without rejecting too many real matches.

ONNX is a portable inference graph format; ONNX Runtime executes that graph through CPU, GPU or providers such as OpenVINO. OpenVINO specializes execution for Intel hardware. Quantization stores or computes selected values at lower precision to reduce size and sometimes latency. UForm's native ONNX artifact already met resource gates, so extra conversion was stopped when quality failed. BLIP behaved differently because it has a joint ITM head, but it was about sixteen times slower here and its calibrated positive abstention was still 52.4%.
