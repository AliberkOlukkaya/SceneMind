# Next model recommendation

Evaluate one small-object/object-detector branch over current CLIP candidates as the next AI experiment. Preserve CLIP and BM25 as fast candidate generators and keep any detector isolated behind the same top-K boundary.

The completed lightweight experiment rules out another round of dual-encoder similarity thresholds. UForm3-small ONNX meets latency and memory gates and improves calibrated R@5 over scalar CLIP and BLIP, but still causes 47.6% held-out positive false abstention. The eight-parameter score/rank classifier worsens that to 71.4%. Raw CLIP finds all three small-object answers at K=5; scalar CLIP, UForm and the learned scorer accept none after calibration. Explicit object evidence is the narrowest next test supported by the failures.

The detector experiment should remain isolated from product code:

1. Define an explicit-object query subset and detector-supported object vocabulary using calibration/development evidence only.
2. Shortlist one permissive CPU detector with a pinned artifact; do not train on Natural V2 held-out.
3. Apply it only to CLIP's top candidates and retain CLIP fallback for unsupported queries.
4. Keep the same FAR, false-abstention, candidate-recall, latency and memory gates.
5. Promote only after two frozen runs pass.

Do not implement a temporal model from these results. UForm retains 100% R@5 on the three action cases, and all tested systems still score independent frames. Do not enlarge Whisper: speech remains the correct path for spoken topics, while visual scoring predictably fails on speech-worded queries.

Automatic query-path routing, OCR, relationship-specific reasoning and Video RAG remain later options. Current evidence selects the smaller object branch first.
