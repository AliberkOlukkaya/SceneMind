# Personal acceptance failures

Status: **collection blocked before execution** (2026-09-15).

## Blocking evidence

1. No real user-selected acceptance media exists under the workspace's ignored `data/` tree.
2. No populated `data/personal-video-acceptance-v1.json` exists.
3. The production duration limit is 1,800 seconds. Videos longer than 30 minutes fail ingestion, which conflicts with testing the full requested 30–60 minute range.

No query-level failure is assigned because no acceptance query was executed. In particular, ASR, Turkish routing, CLIP semantics, timestamp precision, hybrid fusion, unsupported temporal reasoning, OCR need, small-object behavior, no-match behavior, and UI friction have not been measured on personal videos.

## Next action

Place three private videos under ignored `data/` or provide their existing local paths, then create and freeze the private manifest using `PERSONAL_VIDEO_ACCEPTANCE_PROTOCOL.md`. The run must use the current product without label changes or tuning. A video above 30 minutes should first be attempted unchanged so the production rejection is captured; a separate in-limit lecture can supply query-level evidence without changing configuration.

Do not select another ML milestone from this blocked run. The missing evidence is user material and an ingestion product constraint, not demonstrated model failure.
