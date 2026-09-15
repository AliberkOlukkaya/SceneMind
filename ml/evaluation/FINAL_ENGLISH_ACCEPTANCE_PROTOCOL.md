# Final English long-video acceptance protocol

The final acceptance run requires at least one real, continuous 30–60 minute English-speaking lecture, tutorial, technical presentation, or software/project demonstration. It must be available for legal local testing and must not have contributed to SceneMind tuning or threshold selection. Synthetic repetition and concatenation are infrastructure fixtures only.

## Current readiness

The 2026-09-15 exact-SHA inventory found 41 unique media candidates; 33 decoded and eight were deliberately corrupt/unreadable test artifacts. The only decoded 30–60 minute file is the 2,700-second stress fixture, created by repeating a short Blender demo. The longest real content is 1,369.633 seconds (22:49) and silent; two differently encoded copies exist. Final acceptance therefore has decision **D — acceptance not run**. No quality values are inferred from these files.

The exact missing input is one real continuous video with English speech, duration from 1,800 through 3,600 seconds, documented local usage rights, and no previous tuning use.

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
