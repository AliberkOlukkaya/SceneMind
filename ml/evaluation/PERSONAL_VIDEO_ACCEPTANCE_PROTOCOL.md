# Personal video acceptance protocol

Copy `personal_video_acceptance_template.json` into ignored `data/`, then add only real videos Aliberk would search: ideally one 30–60 minute lecture, project demo, podcast/interview, and ordinary video. Record absolute local path, SHA-256, duration, language, source group, and whether a transcript and visual index are expected. Do not commit the populated manifest or media.

Review each complete video before retrieval. Write natural queries, exactly one target route (`VISUAL`, `SPEECH`, or `HYBRID`), and half-open relevant timestamp intervals. Include spoken concepts, exact mentions, visible objects/scenes, mixed narrated-screen moments, ambiguous wording, and explicit-mode corrections. Set `annotation_status` and `annotation_frozen_at` only after review.

Validate the frozen private manifest before running search:

```powershell
$env:PYTHONPATH='backend;.'
.venv/Scripts/python -m ml.evaluation.personal_acceptance `
  data/personal-video-acceptance-v1.json
```

Ingest each video through the normal product, build its visual index and transcript, then run every query in AUTO and its annotated explicit mode. Record requested/selected route, routing confidence, result timestamps, evidence modality, first/warm latency, transcript/index state, and user override. Compute routing accuracy and end-to-end R@1/R@3/R@5/MRR with the existing interval metric. Preserve the query-level failure evidence under ignored `data/`.

Acceptance should require AUTO R@5 >=93%, no category/domain loss above five points versus explicit mode, median routing below 5 ms, and no silent failure when the selected path is unavailable. A failure should direct the next bounded change; it must not trigger automatic model expansion or label editing.
