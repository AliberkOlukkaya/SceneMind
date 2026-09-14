# Personal acceptance results

Status: **NOT EXECUTED — required personal media is absent** (2026-09-15).

The workspace inventory found test fixtures, public benchmark sources, and prior calibration media only. It found no Aliberk-selected lecture/tutorial, project/software demo, ordinary real-world video, or populated private acceptance manifest. Reusing those benchmark assets would violate the requirement to measure the product on untuned daily-use material, so no searches were run and no rates are reported.

| Required measurement | Result |
| --- | --- |
| AUTO routing accuracy | Not measured |
| Useful Top-1 / Top-3 / Top-5 | Not measured |
| English vs Turkish | Not measured |
| Lecture vs demo vs general video | Not measured |
| Processing time | Not measured |
| Transcript quality | Not observed |
| Search latency | Not measured |
| Timestamp error | Not measured |

One production constraint is established from current configuration rather than acceptance inference: `SCENEMIND_MAX_DURATION` defaults to 1,800 seconds. A video over 30 minutes is rejected during metadata validation, so the requested 30–60 minute lecture scenario cannot cover the upper half of that range with current defaults. The acceptance run must preserve this behavior and record the rejection as product friction if a selected video exceeds the limit.

The nine product questions therefore remain unanswered. There is no honest basis yet to claim that Aliberk would use SceneMind today, that Turkish is adequate, that long videos work smoothly, that AUTO is safe for personal use, or that model experimentation should continue or stop. Existing benchmark results remain engineering diagnostics and are not substituted here.

The frozen-manifest validator and human-usefulness summarizer are ready in `personal_acceptance.py`. They enforce all three scenarios, both languages, every requested query category, checksum-bound personal media, positive/negative interval rules, complete observations, and aggregate slices. Production search behavior was not changed.
