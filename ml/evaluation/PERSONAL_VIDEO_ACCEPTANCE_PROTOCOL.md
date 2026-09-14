# Personal video acceptance protocol

This is a private, human-judged product acceptance test. Benchmark and calibration videos cannot substitute for Aliberk's own daily-use material. Keep media, populated manifests, observations, transcripts, indexes, and databases under ignored `data/`.

## Freeze evidence before search

Copy `personal_video_acceptance_template.json` to `data/personal-video-acceptance-v1.json`. Add exactly these three scenarios:

- `lecture_tutorial`: one 30–60 minute lecture or tutorial
- `project_software_demo`: one 15–60 minute project or software demo
- `ordinary_real_world`: one ordinary real-world video

For each video record `video_id`, `scenario`, absolute `local_path`, SHA-256, `duration_seconds`, primary language, and natural English and Turkish queries. Include every category below for every video, with Turkish equivalents of the English search intent:

- `spoken_topic`
- `quoted_mentioned_phrase`
- `visual_object`
- `visual_scene`
- `compositional_visual`
- `mixed_visual_spoken`
- `difficult_negative_unsupported`

Each query needs `query_id`, `text`, `language` (`EN` or `TR`), `expected_best_route`, `expected_presence`, and manually reviewed half-open `relevant_intervals`. A positive needs at least one interval. A negative must use an empty list. Review the complete video, then set `annotation_status` to `frozen` and add `annotation_frozen_at` before running retrieval. Do not edit the frozen labels after seeing results.

Validate the frozen private manifest:

```powershell
$env:PYTHONPATH='backend;.'
.venv/Scripts/python -m ml.evaluation.personal_acceptance `
  data/personal-video-acceptance-v1.json
```

## Exercise the production product

Use the current frontend and backend. Upload each video normally, wait for ingestion, create its five-second CLIP index, and create its Whisper transcript. Record wall-clock processing time and transcript observations. The production maximum duration is currently 1,800 seconds; rejection of a longer test video is an acceptance result and must not be worked around during this run.

Run every frozen query in AUTO and then in Visual, Speech, and Hybrid modes. Judge the returned moments by clicking them. A result is useful only when a normal user would reasonably feel it found the request. Record:

- AUTO selected route and whether it equals the frozen expected route
- useful result in top 1, top 3, and top 5
- absolute timestamp error against the nearest reviewed interval
- `PASS`, `PARTIAL`, or `FAIL`
- one or more prescribed failure categories and a concrete reason
- end-to-end search latency

Top-k judgments must be monotonic: if top 1 is useful, top 3 and top 5 are useful. For an unsupported query, mark a rank useful only if the product response itself helps the user understand there is no supported match; unrelated nearest neighbors are not useful.

Store observations as a JSON object with the frozen `manifest_sha256`, `run_status: "complete"`, and one row for every video and query. Each video row needs `video_id`, `processing_time_seconds`, and non-empty `transcript_quality_observations`. Each query row uses `query_id`, `auto_selected_route`, `useful_top_1`, `useful_top_3`, `useful_top_5`, `timestamp_error_seconds`, `user_usefulness`, `failure_categories`, `failure_reason`, and `search_latency_ms`. It also contains `explicit_mode_results` with `VISUAL`, `SPEECH`, and `HYBRID` entries; every entry has the same usefulness, timestamp, failure, and latency fields except `auto_selected_route`.

Generate the aggregate JSON only after all human judgments are complete:

```powershell
.venv/Scripts/python -m ml.evaluation.personal_acceptance `
  data/personal-video-acceptance-v1.json `
  --observations data/personal-video-acceptance-observations.json `
  > data/personal-video-acceptance-summary.json
```

The summary reports AUTO routing accuracy, AUTO and explicit-mode useful Top-1/3/5, PASS/PARTIAL/FAIL, English/Turkish, scenario and category slices, processing time, median/p95 search latency, timestamp error, and failure counts. Copy only aggregate, non-private measurements into the committed result documents.
