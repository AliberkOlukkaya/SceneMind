# Q&A Abstention Safety V1 protocol

## Scope

This milestone tests whether retrieved transcript evidence actually answers a question. It does not change Q&A chunking, BM25, Top-K, Whisper, CLIP, FAISS, RRF60, or any Find Moments behavior. Ask Video stays disabled with `SCENEMIND_QA_ENABLED=false`.

The previous Grounded Video Q&A V1 development and validation sources are historical only. Neither their questions nor the observed `val-09` and `val-12` failures contributed to design selection.

## Data

Development uses two English CC-licensed sources: *The Oceans* (CC BY 3.0) and *Design Students Experimenting with Free Software* (CC BY 4.0). Validation uses two different CC BY 4.0 sources: *ReWiring the Video Editor — Timeline as a Node* and *Exploring modern UI frameworks*. Source hashes, URLs, durations, labels, key facts, and evidence intervals are in `qa_abstention_safety_v1_manifest.json`.

Each split has 30 questions: 18 answerable and 12 unanswerable. Ten unanswerable questions per split are hard negatives involving a nearby topic, wrong entity/relation, unsupported count, unsupported temporal premise, or plausible outside knowledge. Development, validation, previous Q&A, and historical replay source hashes are disjoint.

## Fixed pipeline

1. Existing deterministic 45-second/900-character chunks with one-segment overlap.
2. Existing BM25 Top-5 retrieval.
3. Deterministic extraction of an explicit requested count, temporal word, and selected relation types.
4. One `gpt-5.4-mini-2026-03-17` Responses API call with `store=false` and a strict JSON schema.
5. The model returns `answerable`, a concise answer, claim-level evidence IDs, a compatibility evidence-ID list, and missing/unsupported requirements.
6. The backend validates every claim citation, rejects unknown IDs, derives displayed citations from the claim set, requires the requested number of separate claims for explicit counts, and converts any invalid contract to the standard abstention response.
7. Backend evidence objects remain authoritative for timestamps and text.

The model may combine supplied chunks and tolerate clear meaning in imperfect ASR. It must abstain when entity, relation, temporal anchor/order, requested count, or another required fact is absent. Outside knowledge is never evidence.

## Development selection

Only three meaningful designs were run. V1 used the strict contract and was too conservative. V2 allowed clear meaning across chunks despite imperfect ASR and was selected; with backend claim-derived citations it returned substantive answers for 16/18 answerable questions and correctly abstained on 12/12 unanswerable questions. V3 added another permissive ASR instruction, regressed to 15/18 answerable decisions, and was rejected. No retrieval setting, model, question, answerability label, key fact, or interval was selected from validation.

## Freeze and validation

Before validation, sources, questions, labels, facts, intervals, model, prompt hash, answerability rule, schema behavior, and generation parameters were frozen. Manifest SHA-256:

`26646d2bf977981950e68888898522e348e6019cc39acac417a8a28acca34255`

Prompt SHA-256:

`5d00c3f2a75ef745440de79ec6b3b640d670315f85661a61a835a9580a35c762`

Validation ran once. No tuning, relabeling, or rerun followed. Every substantive answer and backend-resolved citation was manually compared with the frozen key facts, expected interval, and retrieved transcript text without an LLM judge.

## Gates

- Evidence Recall@5 >= 90%
- Answer Correctness >= 90%
- Grounded Answer Rate >= 95%
- Citation Precision >= 95%
- Citation Recall >= 90%
- Unsupported Claim Rate <= 5%
- Correct Abstention >= 90%
- False Answer Rate <= 10%
- no catastrophic category failure; hard negatives and temporal questions must remain acceptable

Only decision A can make Ask Video eligible for a separate enablement milestone. This milestone never enables the feature automatically.
