# Evidence-Preserving Fusion V1 Frozen Specification

Frozen before source-disjoint validation. Production remains uncapped RRF60.

- Configuration: `quota_1`
- Strategy: `quota`
- Parameters: `{"config_id": "quota_1", "protected_depth": null, "quota": 1, "start_lane": null, "strategy": "quota", "temporal_grouping": "production exact-thumbnail grouping", "tie_rule": "declared lane order, then production RRF rank, timestamp, candidate_id"}`
- Grouping: existing exact-thumbnail production grouping
- Candidate input: existing Visual and Speech lists; rank information only
- Final capacity: 5 unique thumbnail buckets
- JSON specification SHA-256: `5e427a293a07968f5a650cf3aeef4ed974ad8a240974cf11a85de64d4d209b6f`
- Validation rule: one execution; no algorithm or parameter changes afterward
