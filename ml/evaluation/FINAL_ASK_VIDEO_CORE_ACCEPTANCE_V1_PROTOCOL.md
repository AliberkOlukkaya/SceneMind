# Final Ask Video Core Acceptance V1 Protocol

## Purpose and decision rule

This is the final acceptance of SceneMind's bounded transcript-grounded Ask Video core. It is not a prompt or retrieval experiment. The run chooses exactly one decision: A (all gates pass), B (core answer quality fails), C (abstention safety fails), D (scope gate fails), or E (end-to-end product flow fails). Ask Video remains disabled unless every Decision A gate passes.

## Frozen data

The checksum-bound manifest is `final_ask_video_core_acceptance_v1_manifest.json`, SHA-256 `6f11c5212db5a52fe8a5d41016952d81c4804a1bd47f018e2c27d6f8d337961c`.

- Four licensed, English spoken-content sources are absent from all earlier Q&A development, validation, structured-evidence, and historical-replay sets.
- The domains are an educational explainer, technical explainer, art-and-technology presentation, and software tutorial.
- One source entered through the public YouTube URL provider. Three entered through local upload.
- The 60 human-grounded questions were authored from source viewing and transcripts before retrieval or answer generation: 36 supported and answerable, 12 supported but unanswerable, and 12 out of scope.
- Every row fixes the source, scope label, transcript answerability, key facts, expected interval or reason for no answer.

The frozen candidate uses deterministic scope classification, unchanged BM25 Top-5 over unchanged 45-second/900-character overlapping transcript chunks, `gpt-5.4-mini-2026-03-17`, Responses API defaults, strict JSON output, claim-level evidence, backend citation validation, and safe abstention. Failed structured neighbor expansion is outside the production candidate.

## Single-run procedure

1. Verify the manifest checksum plus frozen implementation and prompt hashes.
2. Keep the product default `SCENEMIND_QA_ENABLED=false`.
3. Run every question once in manifest order. Reject detected out-of-scope questions before retrieval/generation and record zero provider usage.
4. For allowed questions, run the unchanged BM25 Top-5 path, the frozen generator, and backend answer/citation resolution. Persist raw output after each call so an interrupted process can resume the same run without repeating completed questions.
5. Review answers, material claims, retrieved evidence, and citations against the frozen annotations. Do not edit questions, annotations, code, prompt, or rules and do not rerun after observing results.
6. Exercise the real URL/upload APIs, Ask UI, citation click-to-seek, and mobile layout. Product-flow checks do not change the frozen scoring rows.

## Metrics and fixed gates

For supported and answerable questions: Evidence Recall@5 >= 90%, Answer Correctness >= 90%, Grounded Answer Rate >= 95%, Citation Precision >= 95%, Citation Recall >= 90%, Unsupported Claim Rate <= 5%, and Core User Success Rate >= 90%.

For supported but unanswerable questions: Correct Abstention >= 95% and False Answer Rate <= 5%.

For out-of-scope questions: Correct Scope Rejection >= 95% and Unsafe Generation Rate <= 5%.

Scope usability requires Supported Question Recall >= 90% and False Scope Rejection Rate <= 10%. The report also includes supported-question precision. No source may suffer catastrophic failure.

Costs use the official standard API rates frozen in the manifest: $0.75 per million input tokens and $4.50 per million output tokens. Rejected questions must have zero model usage.

## Human review rules

An answer is correct only when it communicates all material frozen key facts without a contradictory claim. It is grounded only when every material claim is directly supported by the retrieved transcript evidence. Citation precision is the fraction of returned citations that support the answer; citation recall is measured per answer against the frozen expected interval(s). A core user success requires a correct answer, complete grounding, valid supporting citation, and no unsupported material claim. An abstention is correct only for a frozen unanswerable question. A capability response is correct only for a frozen out-of-scope question and must occur without generator use.

Failures use only: `SCOPE_FALSE_REJECTION`, `SCOPE_FALSE_ACCEPT`, `RETRIEVAL_MISS`, `TRANSCRIPT_ERROR`, `INSUFFICIENT_EVIDENCE`, `GENERATION_ERROR`, `UNSUPPORTED_CLAIM`, `BAD_CITATION`, `MISSED_ABSTENTION`, `FALSE_ABSTENTION`, or `AMBIGUOUS_QUESTION`.
