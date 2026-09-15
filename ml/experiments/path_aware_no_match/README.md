# English path-aware no-match experiment

This experiment asks whether the existing ranked lists can distinguish an English query whose content exists from one whose content is absent. It does not change production search. AUTO still routes first; the experiment then applies a Visual, Speech, or Hybrid rule to that selected path.

The frozen manifest contains six source groups. Three Commons lecture/tutorial/demo groups provide 36 positives and 36 balanced hard negatives for calibration. Three different Natural V2 groups provide a single held-out evaluation with 24 positives and 24 negatives. Every path is balanced within each split. Negative taxonomy A–I covers unrelated content, in-domain omissions, wrong attributes/actions, absent scene objects, plausible visual omissions, absent related/technical speech, and compositional mismatch. Source URLs, licenses, attribution, media checksums, positive intervals, and the frozen personal-acceptance-manifest checksum are stored in `manifest_v1.json`. Media, frames, transcripts, and vectors remain ignored.

The candidate evidence is deliberately cheap. Visual features summarize the existing CLIP list: top score, score margins, top-five mean/variance/entropy, temporal support, and separated support. Speech features summarize production BM25 and the existing transcript: top score/margin, normalized term coverage, matching-segment count, concentration, phrase overlap, and technical-term coverage. Hybrid candidates use both feature families and five explicit evidence states. Candidate rules are one threshold or a conjunction of two thresholds; no classifier or new model is fitted.

Run from the repository root:

```powershell
$env:PYTHONPATH='backend;.'
.\.venv\Scripts\python.exe -m ml.experiments.path_aware_no_match.build_manifest
.\.venv\Scripts\python.exe -m ml.experiments.path_aware_no_match.prepare
.\.venv\Scripts\python.exe -m ml.experiments.path_aware_no_match.run_experiment
```

`prepare` checksum-verifies local ignored media and creates the production-equivalent five-second 480-pixel frames, Whisper-tiny transcript, and CLIP vectors needed by held-out evaluation. Rule selection sees calibration rows only. Held-out labels are consumed after the rules are fixed. A second identical held-out run and the frozen personal English subset are allowed only when every quality gate passes.

The selected rules failed. See [results](../../evaluation/PATH_AWARE_NO_MATCH_RESULTS.md), [failures](../../evaluation/PATH_AWARE_NO_MATCH_FAILURES.md), and the [machine report](../../evaluation/reports/path-aware-no-match-v1.json).
