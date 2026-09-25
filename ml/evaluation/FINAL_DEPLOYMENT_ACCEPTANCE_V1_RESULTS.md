# SceneMind v1.0 final deployment acceptance

**Decision C — not ready for deployment. Do not tag `v1.0.0`.** The one frozen
run missed every predeclared aggregate and category quality target. Three of
eight videos met the catastrophic-source definition, including the 40-minute
lecture. A post-run review also found under-annotated repeated visual moments,
so the interval score is a reproducible diagnostic rather than a defensible
public *user-success* claim. No retrieval parameter or ground-truth interval
was changed after the run.

## Set, freeze and limits

- Eight new licensed/public-domain Wikimedia Commons videos: one 40.38-minute
  UK Parliament lecture, a VOA studio talk, an OpenRefine screen tutorial, a
  CDC PPE tutorial, a 1901 Manchester street scene, an aikido demonstration,
  a CIA museum accessibility report, and a VOA heritage museum report.
- 124 positives: 44 Speech, 55 Visual, 25 Smart/Hybrid. All eight videos have
  at least eight positives. Five absent-concept diagnostics are separate.
- Source-title and media-hash searches found no match in earlier tracked
  evaluation material. The [frozen manifest](FINAL_DEPLOYMENT_ACCEPTANCE_V1_MANIFEST.json)
  includes source URLs, licenses, media SHA-256 hashes, queries, intervals,
  difficulty, code hashes, model/config settings and gates. Manifest SHA-256:
  `d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196`.
  The [protocol](FINAL_DEPLOYMENT_ACCEPTANCE_V1_PROTOCOL.md) was committed at
  `c85f1ac` before the single run.
- Queries were written from independent source-video contact sheets and three
  publisher caption tracks, never SceneMind search/ASR output. The annotation
  was agent-reviewed, with **no independent human sign-off** and no full
  frame-by-frame or second-annotator review. Thus the requested
  “human-grounded” standard is not fully established. Similar scenes in
  multiple positions were sometimes assigned only one interval; those labels
  stayed frozen and the affected hits remain scored as misses. This is an
  additional release blocker, not a reason to revise this holdout.

The evaluator used the real upload, ingest, Whisper, CLIP index and `/search`
routes through FastAPI TestClient; it did not monkeypatch inference. The
production modes were verified: Smart → Hybrid → uncapped RRF60 with up to 50
Speech and 50 Visual candidates, Spoken → BM25 Speech, Visual → CLIP/FAISS.
Frames remained at the existing five-second interval and CLIP revision and
Whisper tiny defaults remained unchanged. `k=5`, with a relevant hit defined
only by a returned timestamp inside a pre-annotated closed interval.

## Frozen interval metrics

| Mode | Positives | R@1 | R@3 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spoken Content | 44 | 68.2% | 81.8% | 81.8% | 74.2% |
| Visual Content | 55 | 30.9% | 56.4% | 63.6% | 44.1% |
| Smart Search | 25 | 36.0% | 60.0% | 68.0% | 47.1% |
| **Overall** | **124** | **45.2%** | **66.1%** | **71.0%** | **55.4%** |

The numerical “Top-5 User Success Rate” under the frozen interval rule is
88/124 = **71.0%**. Because repeated relevant moments were under-annotated and
there was no independent human assessment of clicked results, this number
must **not** be marketed as verified real-user success or generic accuracy.

| Difficulty | Positives | R@1 | R@3 | R@5 |
| --- | ---: | ---: | ---: | ---: |
| Easy | 21 | 47.6% | 61.9% | 71.4% |
| Medium | 61 | 44.3% | 67.2% | 67.2% |
| Hard | 42 | 45.2% | 66.7% | 76.2% |

Difficulty groups have different source/mode mixes; their non-monotone results
do not mean hard queries are intrinsically easier.

| Source | Positives | R@1 | R@3 | R@5 | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Aikido demonstration | 8 | 0.0% | 25.0% | 37.5% | Catastrophic |
| Heritage museum | 8 | 50.0% | 75.0% | 87.5% | Below overall target |
| Manchester street | 8 | 50.0% | 87.5% | 100.0% | Passes R@5 target |
| Museum accessibility | 26 | 65.4% | 88.5% | 96.2% | Passes R@5 target |
| OpenRefine tutorial | 26 | 53.8% | 76.9% | 76.9% | Below target |
| Parliament lecture | 32 | 37.5% | 50.0% | 50.0% | Catastrophic; long video |
| PPE tutorial | 8 | 50.0% | 62.5% | 75.0% | Below target |
| Social-media studio talk | 8 | 12.5% | 37.5% | 37.5% | Catastrophic |

Predeclared targets were overall R@1/3/5 >=75/85/90%, Speech/Visual/Smart
R@5 >=90/85/90%, long-video R@5 >=85%, and no source with at least eight
queries below 60% R@5. Every aggregate/category target and the long/no-
catastrophe gates failed.

## Long-video and resource profile

The Parliament source measured **2,422.8 seconds (40.38 minutes)**, 485
frames and 442 transcript segments. Upload/ingest took 35.8 seconds, local
Whisper transcription **551.5 seconds**, and CLIP indexing 29.4 seconds:
**616.7 seconds** total measured processing. It occupied **164.0 MiB** in its
UUID media directory and reached **3,338.4 MiB** peak process-tree RSS in this
single-process TestClient run. Its R@1/3/5 was 37.5/50.0/50.0% and MRR@5
43.2%. Its transcript language was incorrectly detected as **Welsh (`cy`)**
despite English publisher captions, corrupting its early English transcript.
This explains eight Speech misses and contributes to four Smart misses. No
language override was applied after this holdout observation.

Across the other seven sources, total per-video processing ranged 3.7–30.6
seconds and persistent media/index usage ranged 8.9–75.4 MiB. Their measured
single-process peak RSS ranged 859.9–1,365.5 MiB. Warm query latency over 124
searches was **23.9 ms median**, **45.5 ms p95** (nearest-rank) and 249.4 ms
maximum. These numbers include local synchronous TestClient operations and
must not be presented as separate API/worker deployment measurements. The
current cache directories for production CLIP and faster-whisper tiny occupy
about 1.292 GB combined; other experimental models in `data/models` are not
part of this figure. A prior durable-worker 45-minute stress fixture measured
3,018 MB peak worker RSS; see [long-video results](LONG_VIDEO_INGEST_RESULTS.md).

## Failure audit

There were **36 frozen Top-5 misses**. The per-query ranks, timestamps and
provisional cause assignments are in the [machine-readable report](reports/final-deployment-acceptance-v1.json).
The dominant audited cause is the long lecture's incorrect Whisper language
detection. The post-run diagnostic taxonomy is:

| Dominant provisional cause | Count | Representative evidence |
| --- | ---: | --- |
| Transcription error | 12 | Parliament Speech F001–F008 and Smart F105/F106/F108/F109; `cy` transcript against English source captions. |
| Multiple similar moments | 10 | Repeated aikido actions, recurring studio shots and PPE steps; some alternate returned timestamps may also expose missing intervals. |
| Annotation issue | 5 | Broad queries for the orange presenter, lecture audience/poster or renovated museum had only one of several plausible intervals. |
| Visual semantic miss | 5 | OpenRefine table/dialog and other scene searches surfaced unrelated visual frames. |
| Small object | 3 | Tiny numeric facet, spectacles and UI director field. |
| Hybrid ranking displacement | 1 | OpenRefine matching-status facet did not reach Top-5 under fusion; this is a diagnostic hypothesis, not a separately measured counterfactual. |

The taxonomy is **provisional** because clicked-result usefulness was not
independently human adjudicated. A top result for F045, for example, shows the
same orange-sweater presenter later in the video; the frozen interval covers
only the opening. This makes the interval R@5 overly strict for that query.
We neither widened the label nor reran acceptance. A future benchmark would
need fresh sources and independent interval review; this holdout is spent.

Five absent-concept diagnostics all returned HTTP 200 with five possible
moments. This is expected for a ranker without validated no-match detection.
The UI continues to say “Most relevant moments” and “Possible matches” and
does not claim absence or certainty.

## Product and release checks

Local upload, real processing, mode-specific search and indexed results were
exercised in the frozen API run. Desktop and mobile Playwright tests exercised
library, upload, URL-import UI contract, processing/errors, modes, thumbnails,
timestamps and click-to-seek; the opt-in real CLIP fixture search also sought
correctly in both viewports. Browser URL-import assertions use a mocked remote
response; they are not proof of a live YouTube download. The separate live
YouTube recheck succeeded independently: official Blender Foundation Big Buck
Bunny URL `aqz-KE-bpKQ` returned import HTTP 202, acquired and processed a
634.57-second local artifact in 108.6 seconds with 127 frames, built a ready
visual index in 13.9 seconds, returned HTTP 200 with five visual search
results, and served the local video with HTTP 206 byte-range response. This
was a separate flow check, **not** a second run of the frozen accuracy set.
The actual Next.js workspace was then opened against that real API/artifact in
desktop Chromium and a mobile iPhone 13-sized Chromium viewport. Selecting
Visual Content and searching `animated rabbit in a field` showed the first
result at 8:10; clicking it set the player's time to **490 seconds** in both
viewports, without horizontal overflow. This verifies real browser result
navigation for the URL video, while the frozen 40-minute lecture was not
requeried after acceptance.

Full validation after the release-only deployment changes: backend **282
passed, 1 skipped**; Ruff passed; frontend ESLint, TypeScript and Next.js
production build passed; Playwright **18 passed** including desktop/mobile
real-model click-to-seek. `docker compose config --quiet` passed. The Docker
daemon was unavailable, so the corrected image dependency installation and
container runtime were not build-tested. No production retrieval code changed.

The release stays at **`v1.0.0-rc1`**. Do not create `v1.0.0` or a GitHub
Release. Public-facing docs should report this frozen interval diagnostic,
its limitations, the catastrophic sources, and Decision C. A credible
deployment decision would require a newly designed, independently human-
reviewed acceptance set and an untested new-development remedy for the
observed ASR/retrieval failures, never retuning this holdout.
