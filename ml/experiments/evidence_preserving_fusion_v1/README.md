# Evidence-Preserving Hybrid Fusion V1

This evaluation-only experiment tests whether deterministic lane preservation can prevent strong
Visual or Speech evidence from disappearing under production RRF60. It never imports into
`backend/app` and does not change Smart Search.

The finite family contains eight configurations: production RRF60, strict modality quotas 1/2,
Visual-first and Speech-first interleaving, and protected rank depths 1/2/3. Every method uses only
rank, existing exact-thumbnail grouping and deterministic ordering. Raw scores are recorded by the
production trace but are not used by a candidate.

Development selected `quota_1`: preserve each lane's first unique thumbnail bucket, then fill the
five-result capacity in production RRF order. The JSON specification was SHA-256 frozen before a
single source-disjoint validation run.

Validation improved Top-1/3/5 from 19/21/21 to 20/22/22 over 24 positives, recovered one Speech
failure, retained all 21 baseline successes and reduced measured displacement/irrelevant consensus
from one to zero. It still fails promotion because all aggregate gain is confined to Speech while
Multimodal MRR falls. Decision **B** leaves production RRF60 unchanged and preserves the protected
holdout.

Reproduce development selection and, only with the existing frozen local assets, validation:

```powershell
.venv/Scripts/python -m ml.experiments.evidence_preserving_fusion_v1.run_experiment development
.venv/Scripts/python -m ml.experiments.evidence_preserving_fusion_v1.run_experiment validation
```

The historical run consumed validation once. Do not rerun a modified candidate against it.
