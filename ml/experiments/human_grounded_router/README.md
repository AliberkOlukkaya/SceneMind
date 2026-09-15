# Human-grounded English AUTO router validation

This experiment tests the previously fixed character 3–5 gram TF-IDF plus linear-softmax router against natural English searches tied to evidence in real videos. It does not use the protected failed Final English Acceptance video, transcript, manifest, timestamps, failures, or queries. Large source media, extracted frames, contact sheets, and local Whisper transcripts stay under ignored `data/human-grounded-router/`.

## Data and inspection

Ten independent openly licensed videos contribute 360 unique queries, balanced at 120 Visual, 120 Speech, and 120 Hybrid. The split is by video: four train videos provide 180 queries, four validation videos provide 120, and two previously unseen frozen-test videos provide 60. Every event was checked against actual five-second JPEG contact sheets and a locally generated Whisper transcript before the query and route were written.

| Split | Source | Domain | License |
|---|---|---|---|
| Train | Explainer Video – Using good sources on Wikipedia | general explainer | CC BY-SA 4.0 |
| Train | The Oceans | science explainer | CC BY 3.0 |
| Train | WebM: A Video Codec for the Web | technical explainer | CC BY 3.0 |
| Train | Blended learning | education explainer | CC BY-SA 4.0 |
| Validation | Cooking with Laura: Homemade Pasta | ordinary instruction | CC BY 3.0 |
| Validation | Let's change the default language of the Internet | presentation | CC BY 4.0 |
| Validation | Design Students Experimenting with Free Software | technical presentation | CC BY 4.0 |
| Validation | Humans as Software Extensions | art/technology lecture | CC BY 4.0 |
| Frozen test | ReWiring the Video Editor – Timeline as a Node | software UX presentation | CC BY 4.0 |
| Frozen test | Exploring modern UI frameworks | technical presentation | CC BY 4.0 |

`sources_v1.json` records page URLs, download URLs, attribution, byte sizes, durations, and SHA-256 checksums. `build_annotations.py` retains the literal reviewed events, rationales, intervals, and queries; `annotations_v1.json` is the frozen evaluated manifest. Its SHA-256 over canonical JSON is `62e66456f00b9afc974c259c07a5512c05c60942052f961485fd4888ad3fca4d`.

## Annotation policy

Visual means visible evidence can answer the request. Speech means the answer depends on the spoken statement. Hybrid requires both a particular visible state and a spoken fact in the same reviewed interval. The audit asks whether either single modality could answer satisfactorily and flags borderline rows instead of silently changing them. Twenty of 360 rows are predeclared ambiguous. The final test deliberately mixes questions, commands, noun phrases, indirect requests, and technical wording across routes.

## Reproduction

With the repository virtual environment active and `PYTHONPATH=backend;.`:

```powershell
python -m ml.experiments.human_grounded_router.prepare
python -m ml.experiments.human_grounded_router.build_annotations
python -m ml.experiments.human_grounded_router.run_experiment
```

Preparation verifies local media before extracting review evidence. The experiment validates the frozen checksum, evaluates production first, trains only on train, reports validation, and opens the frozen test once. Character configuration is inherited unchanged from the prior experiment: character 3–5 grams, at most 8,000 features, minimum document frequency two, regularization 0.05, 450 iterations, and no confidence fallback. The experimental artifact remains ignored because it failed promotion gates.

The fixed character candidate reaches only 60.00% frozen accuracy. Production is unchanged; see [results](../../evaluation/HUMAN_GROUNDED_ROUTER_RESULTS.md) and [failures](../../evaluation/HUMAN_GROUNDED_ROUTER_FAILURES.md).
