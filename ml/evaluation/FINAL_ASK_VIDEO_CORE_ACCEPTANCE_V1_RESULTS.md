# Final Ask Video Core Acceptance V1 Results

## Decision

**B — CORE ANSWER QUALITY FAILED.** Ask Video remains disabled. The deterministic scope gate, retrieval, grounding, citations, and hard-negative abstention were safe, but the bounded core was not useful enough: Answer Correctness and Core User Success were **83.33%**, below the fixed 90% gates, and Citation Recall was **88.89%**, below 90%. Four directly answerable questions were falsely abstained and two answers omitted frozen material facts. No acceptance input, rule, prompt, or threshold was changed and the run was not repeated.

The current transcript-BM25 plus generation architecture has reached its practical V1 limit. Further incremental prompt, BM25, neighbor-expansion, or small-rule work is stopped. If Q&A resumes, the next milestone must choose and pre-register a different architecture, beginning with measured semantic transcript retrieval and hierarchical evidence representation.

## Data and freeze

The single frozen run used four new English spoken-content videos and 60 independently authored, human-grounded questions: 36 supported/answerable, 12 supported/unanswerable, and 12 out of scope. None of the sources appears in earlier Grounded Video Q&A, Abstention Safety, Structured Evidence, or historical Q&A replay sets.

| Source | Domain | License | Ingress | Duration | Frames | Whisper segments |
|---|---|---|---|---:|---:|---:|
| Creative Commons licences explained | Educational explainer | CC BY 3.0 NZ | YouTube | 332.76 s | 67 | 55 |
| WebM: A Video Codec for the Web | Technical explainer | CC BY 3.0 | Upload | 125.96 s | 26 | 24 |
| Humans as Software Extensions | Presentation/talk | CC BY 4.0 | Upload | 1,589.28 s | 318 | 273 |
| Demo Video Tutorial | Software tutorial | CC BY-SA 4.0 | Upload | 199.90 s | 40 | 54 |

The manifest SHA-256 is `6f11c5212db5a52fe8a5d41016952d81c4804a1bd47f018e2c27d6f8d337961c`. It freezes media and transcript hashes, questions, annotations, scope labels, expected evidence, implementation, prompt, model, generation behavior, BM25 Top-5, chunking, and pricing. The run used `gpt-5.4-mini-2026-03-17` once in manifest order.

## Metrics

| Area | Metric | Result | Gate | Pass? |
|---|---|---:|---:|:---:|
| Scope | Supported Question Recall | 100.00% (48/48) | >=90% | Yes |
| Scope | Supported Question Precision | 100.00% (48/48 allowed) | Report | Yes |
| Scope | False Scope Rejection | 0.00% (0/48) | <=10% | Yes |
| Scope | Out-of-Scope Rejection | 100.00% (12/12) | >=95% | Yes |
| Scope | Unsafe Generation | 0.00% (0/12) | <=5% | Yes |
| Core | Evidence Recall@5 | 100.00% (36/36) | >=90% | Yes |
| Core | Answer Correctness | 83.33% (30/36) | >=90% | **No** |
| Core | Grounded Answer Rate | 100.00% (32/32 answers) | >=95% | Yes |
| Core | Citation Precision | 100.00% (45/45) | >=95% | Yes |
| Core | Citation Recall | 88.89% (32/36) | >=90% | **No** |
| Core | Unsupported Claim Rate | 0.00% (0/62 claims) | <=5% | Yes |
| Core | Core User Success | 83.33% (30/36) | >=90% | **No** |
| Unanswerable | Correct Abstention | 100.00% (12/12) | >=95% | Yes |
| Unanswerable | False Answer Rate | 0.00% (0/12) | <=5% | Yes |

The false-abstention rate on supported/answerable questions was 11.11% (4/36). Per-video Core User Success was 77.78%, 88.89%, 88.89%, and 77.78% in table order. No video fell to or below 50%, so there was no catastrophic source failure.

## Performance and cost

The 48 scored provider calls used 40,277 input and 4,397 output tokens. At the frozen official standard rates of $0.75 and $4.50 per million input/output tokens, the scored run cost **$0.049994**, or **$0.001042 per generated call**. Generation median/p95 was **1,541.55/2,635.80 ms**. End-to-end median/p95 across all 60 rows, including zero-generation scope rejections, was **1,404.74/2,511.92 ms**.

Two separate real endpoint checks used 1,869 input and 155 output tokens and cost $0.002099. Full milestone API cost was **$0.052093**. Scope-rejected questions consumed no model tokens.

Pricing was frozen from the [official GPT-5.4 Mini model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini): $0.75/M input and $4.50/M output tokens.

## End-to-end product flow

The public YouTube URL `https://www.youtube.com/watch?v=4ZvJGV6YF6Y` was acquired by the production YouTube provider in 9.82 seconds, ingested in 15.86 seconds, transcribed in 17.04 seconds, and indexed in 26.60 seconds. A real Ask endpoint request returned HTTP 200, a supported answer, and two timestamp citations. The local upload flow also completed normal ingest, Whisper, and CLIP indexing; its real Ask endpoint request returned HTTP 200, a supported answer, and one timestamp citation.

The repository default and environment remained disabled. An isolated evaluation runtime enabled the endpoint only for these two product checks. Existing and expanded Playwright coverage verifies Ask rendering, abstention, bounded-scope messaging, citation click-to-seek, and the same workspace on desktop and mobile. This product flow passed, so Decision E does not apply.

## Product scope

The implemented candidate clearly describes its intended scope as facts and explanations spoken in the video. It allows factual, definition, direct-explanation, localized-topic, and localized-summary questions. It rejects explicit list/count, temporal-ordering, long-range compositional, visual-only, and OCR-dependent structures before generation. Unanswerable transcript questions remain in scope and must safely abstain.

Because Decision A did not occur, `SCENEMIND_QA_ENABLED` remains false, the version remains `v1.0.0-rc1`, and no v1.1 tag is prepared or pushed.

## Validation

- Backend: 248 passed, 1 opt-in real-network test skipped; two upstream deprecation warnings.
- Ruff: passed.
- Frontend ESLint, TypeScript, and Next.js production build: passed.
- Playwright: 16 desktop/mobile flows passed; two opt-in real-model flows skipped.
- Frozen retrieval regression selection: 75 passed.
