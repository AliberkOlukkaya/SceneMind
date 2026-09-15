# Final English long-video acceptance protocol

This is the historical v1 protocol used for the completed AUTO-based run. The next acceptance milestone uses [FINAL_ENGLISH_ACCEPTANCE_V2_PLAN.md](FINAL_ENGLISH_ACCEPTANCE_V2_PLAN.md), defaults directly to Smart Search / Hybrid, and does not score AUTO routing.

The final acceptance run requires at least one real, continuous 30–60 minute English-speaking lecture, tutorial, technical presentation, or software/project demonstration. It must be available for legal local testing and must not have contributed to SceneMind tuning or threshold selection. Synthetic repetition and concatenation are infrastructure fixtures only.

## Current result

After the initial inventory produced decision D, an eligible CC BY 4.0 real continuous 54:11 English technical presentation was supplied. The protocol was executed once with 30 frozen queries. Decision **C — FINAL ACCEPTANCE FAILED**: ingestion and conservative negative UX passed, while AUTO routing and positive Top-k failed. See [FINAL_ENGLISH_ACCEPTANCE_RESULTS.md](FINAL_ENGLISH_ACCEPTANCE_RESULTS.md). The failed media and labels remain held out and cannot be used for tuning; any later final acceptance requires a new independently reviewed source and manifest.

## Freeze the private manifest

After suitable media is available, copy `final_english_acceptance_template.json` to ignored `data/final-english-acceptance-v1.json`. Watch the complete video before using SceneMind search. Record the checksum, duration, rights, English language, inspection completion, and non-tuning status. Write 25–40 natural English queries spanning spoken concepts, explained topics, mentioned technical terms, visual objects, visual interface/scene states, spoken-plus-visual evidence, supported compositions, and plausible negatives.

Each query records its text, category, expected Visual/Speech/Hybrid route, reviewed half-open timestamp intervals, rationale, and negative status. Positives require intervals; negatives require none. Freeze the file before retrieval and validate it:

```powershell
$env:PYTHONPATH='backend;.'
.venv/Scripts/python -m ml.evaluation.final_english_acceptance `
  data/final-english-acceptance-v1.json
```

## Run the product and judge usefulness

Use the normal frontend upload and durable backend worker. Wait for ingestion, Whisper transcription, CLIP indexing, and Ready state. Record upload and total processing time, peak worker RAM, disk use, frame and ASR segment counts, failures, and retries. Run every frozen query in AUTO; use explicit modes only to diagnose failures.

Click and review returned moments. Record the selected route, Top-1/3/5 human usefulness, timestamp quality, PASS/PARTIAL/FAIL, failure categories, latency, and a concrete reason. For every negative, record whether the conservative wording makes the candidate-list behavior understandable. Returning candidates alone is not a failure, and interval overlap alone is not a PASS.

Bind the private observation file to the frozen manifest SHA-256, set `run_status` to `complete`, and summarize:

```powershell
.venv/Scripts/python -m ml.evaluation.final_english_acceptance `
  data/final-english-acceptance-v1.json `
  --observations data/final-english-acceptance-observations-v1.json
```

The fixed gates are AUTO route accuracy at least 90%, useful Top-3 at least 85%, useful Top-5 at least 90%, no catastrophic pipeline failure, and understandable conservative negative-query UX. Top-1 of 70% is preferred. Do not change labels or gates after seeing results.
