# English AUTO router generalization

This bounded experiment changes no retrieval component. CLIP, Whisper, BM25, RRF, five-second sampling, Visual/Speech/Hybrid paths, result wording, and long-video processing stay fixed. It compares query-text routers only.

## Data and annotation

`dataset_v1.json` contains 450 independently authored English queries across 15 source scenarios spanning technical, software, scientific, creative, repair, and ordinary instructional domains. Nine source IDs belong to train, three to validation, and three to `frozen_test`; a source never crosses a split. Each source contributes ten Visual, ten Speech, and ten Hybrid queries. The test manifest's canonical SHA-256 is `ce1f2b0e0665c037fb4f914878e5209052160687775cfdfdafa5d406810380da` and is checked before evaluation.

The source scenarios are text-only annotation cards. No media, transcript, slide, returned result, timestamp, label, or phrasing from final acceptance was used. When the ignored protected manifest is locally available, the runner rejects normalized exact query overlap. The committed validator also rejects duplicate queries, source overlap, query/source split disagreement, missing classes, imbalance, and fewer than 300 rows.

Route labels describe evidence needed to answer:

- `VISUAL`: the requested moment is identified from visual evidence alone.
- `SPEECH`: spoken words or transcript alone identify it.
- `HYBRID`: the spoken point and displayed context both materially constrain the answer.

Words such as “show,” “say,” or “talk” do not determine the label. Queries include questions, imperatives, short fragments, indirect requests, technical and ordinary vocabulary, mixed evidence, and lexical traps. The scenarios were independently authored but were not verified against real videos. This is the decisive dataset limitation and prevents promotion.

## Candidates

The unchanged production baseline uses 18 readable lexical features and 54 multinomial weights. The candidates use word unigram/bigram TF-IDF, character 3–5 gram TF-IDF, or both, followed by a deterministic three-class softmax linear classifier implemented with NumPy. Vocabulary and IDF are fitted on train only. Architecture and confidence fallback are selected on validation only; frozen test labels never affect fitting or selection.

The selected character model uses at most 8,000 character features, 23,919 classifier parameters, and a validation-selected low-confidence Hybrid fallback. Inputs are non-empty English query strings. Outputs are Visual, Speech, or Hybrid plus class probabilities. Preprocessing case-folds text, retains ASCII alphanumeric/apostrophe word forms, creates boundary-aware character n-grams, applies smoothed IDF and L2 normalization, then computes a local linear softmax. It needs no service, paid API, model download, GPU, training corpus, or new runtime dependency.

Word features are smaller and faster but more sensitive to unseen forms. Character n-grams improve phrasing tolerance at a larger yet still sub-megabyte artifact size. A compact pretrained sentence embedder was not evaluated because the classical candidate passed the internal numeric gates. The candidate artifact is written only under ignored `data/`; production retains `query_router_v1.json`.

Run from the repository root:

```powershell
$env:PYTHONPATH = "backend;."
.venv\Scripts\python -m ml.experiments.english_router_generalization.run_experiment
```

The runner verifies the frozen hash and split/leakage invariants, evaluates production baseline, selects candidates on validation, evaluates the frozen test, repeats a passing configuration deterministically, writes the committed aggregate report, and leaves the experimental model artifact ignored.
