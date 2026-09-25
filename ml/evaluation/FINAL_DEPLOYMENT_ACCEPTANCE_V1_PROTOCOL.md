# SceneMind v1.0 final deployment acceptance — frozen protocol

The evaluation set was frozen before invoking production search. Manifest SHA-256:
`d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196`.
The manifest contains source URLs, licenses, media hashes, 124 positive queries,
five absent-concept diagnostics, fixed modes/difficulties/intervals, production
configuration and SHA-256 hashes for the retrieval code and evaluator. The eight
videos were checked by source-title search against earlier tracked evaluation
material, with no prior-source match. Source media, publisher caption files,
contact sheets, model cache, database, embeddings and run log stay under ignored
`data/`.

The query annotations are based on independent source-video contact sheets at
10-second intervals (30 seconds for the 40-minute lecture) and publisher English
caption tracks for the three captioned sources. Caption timing is not the
SceneMind transcript. The other five videos have visual-only annotations.
Contact sheets provide a sampled review, so tiny or short events and precise
boundaries require caution; there was no separate human annotator sign-off or
inter-annotator agreement study. Do not describe this as an independently
human-audited benchmark. The final report must disclose these limitations.

There are 44 Speech, 55 Visual and 25 Smart/Hybrid positives. Every source has
at least eight positives. A result is relevant when its returned timestamp is
inside any closed annotated interval. A video with at least eight positives and
R@5 below 60% is catastrophic. Positive-query Recall@1/3/5 and MRR@5 use the
first relevant rank, with missing hits contributing zero. The Top-5 User Success
Rate is exactly interval-based R@5. Negative diagnostics are excluded from
positive recall and do not establish no-match reliability.

Predeclared promotion targets: overall R@1 >=75%, R@3 >=85%, R@5 >=90%; Speech
R@5 >=90%, Visual R@5 >=85%, Hybrid R@5 >=90%; long-video R@5 >=85%; and no
catastrophic source. Passing accuracy alone does not establish deployment
readiness; real user flow, release checks, security and resource constraints
also matter. A failure must be reported without retrieval tuning, label changes
or a second acceptance run.

Run the evaluator once from the repository root after verifying the manifest:

```powershell
.venv\Scripts\python scripts\final_deployment_acceptance.py --manifest-sha d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196
.venv\Scripts\python scripts\final_deployment_acceptance.py --manifest-sha d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196 --run
```

The script exercises upload, ingestion, transcript (for captioned sources),
visual indexing and the production `/videos/{id}/search` route. It saves
incremental details in ignored `data/final-deployment-v1/run.json` and refuses
to start if this output already exists. The separate browser journeys and
YouTube URL flow must be recorded independently; this script does not measure
them.
