# Image-text verifier experiment

This isolated experiment tests whether a joint image-text model can verify and rerank the existing CLIP top five. It does not change production search. The flow is CLIP candidate generation, batched pair scoring, verifier-only reranking, then a calibration-only no-match threshold.

## Model shortlist

| Candidate | Architecture and scoring | Inputs and size | License and CPU assessment | Decision |
| --- | --- | --- | --- | --- |
| `Salesforce/blip-itm-base-coco` at `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c` | ViT-B image encoder, BERT text encoder, and joint multimodal image-text matching head; returns two ITM logits without generation | RGB resized to 384×384 and normalized by the pinned processor; BERT tokens capped at 35 here; 223,744,258 parameters; 895 MB PyTorch weights | BSD-3-Clause; supported on CPU. Measured batch-of-five median 3.35 s and isolated peak working set 1.49 GB | Tested because it is the smallest direct ITM checkpoint with first-class Transformers support among the serious candidates. Rejected after evaluation. |
| `BridgeTower/bridgetower-base-itm-mlm` at `6cfb318bb08588808368a9ab1fe26609bc148f1f` | ViT and RoBERTa encoders connected through bridge and cross-modal layers; Transformers exposes an image-text retrieval head; no generation required | 288×288 image configuration, text length 50 in its published config; approximately 1.6 GB PyTorch checkpoint | MIT; CPU execution is available through PyTorch, but expected memory and latency exceed BLIP because the checkpoint is substantially larger | Rejected by the initial resource filter after BLIP exceeded the latency ceiling by over 13×. |
| `google/siglip-base-patch16-224` at `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Independent vision/text encoders trained with sigmoid pair loss; exposes compatibility logits, but no joint cross-attention or dedicated ITM verifier head | 224×224 RGB, SentencePiece text; 203,155,970 parameters; 813 MB safetensors | Apache-2.0; CPU capable and likely similar to a replacement retriever | Rejected for this experiment because it does not test the proposed second-stage joint verifier architecture. |
| BLIP-2 / large generative VLMs | Multi-billion-parameter vision-to-language generation | Multi-GB weights and generation tokens | CPU use conflicts with the local latency/resource target | Rejected before download. A generative yes/no response is unnecessary for pair scoring. |

Model sizes are checkpoint sizes, not total runtime memory. Only BLIP was benchmarked locally. BridgeTower latency and memory are resource-filter inferences from its larger checkpoint, not measurements.

## Expanded calibration evidence

`calibration_v1.json` adds 30 frozen annotations over three source-disjoint Commons videos to the original 16 Natural V2 calibration annotations. The resulting fit pool contains 25 positive and 21 negative visual queries across five videos. New negatives probe food without eating, cooking without cutting or pouring, a walking couple without bicycles/dog/wedding/bench, and a dog without a cat/ball/pool/sofa/leash-walking relation.

The added sources are:

- `Food Preparation.webm`, CC BY-SA 4.0, Blvrsngh80540, SHA-256 `a2a2069e7fb5f27c26c6683c7ba2d831291b7b2b8d0e65ea115803301db93153`.
- `Couple holding hands.webm`, CC BY 3.0, YouTube user Editor, SHA-256 `726fa7efba143027486bddfbb2420f40284088cfa744df1993f17d7dbeb68b5f`.
- `Dog seen from a GoPro.webm`, CC BY 3.0, Pablo Fuentes, SHA-256 `66559a1c5d786a414202e417c39c86fbc7d6b2240d3dc06e20f2dd87b97fb87a`.

The manifest records direct downloads, source SHA-1 values, licenses, attribution, durations, and preparation. Media remains ignored.

## Reproduction

From the repository root with the visual dependencies installed:

```powershell
.venv/Scripts/python -m ml.experiments.image_text_verifier.prepare
.venv/Scripts/python -m ml.experiments.image_text_verifier.evaluate
.venv/Scripts/python -m ml.experiments.image_text_verifier.regression data/verifier-report.json ml/evaluation/reports/verifier-blip-v1.json
```

The default evaluation writes ignored artifacts under `data/`; it never overwrites the committed baseline. The model downloads into the existing ignored model cache. The runner verifies all eight video checksums and split disjointness, loads both models once, batches each query's five candidate images, fits only on calibration rows, and evaluates Natural V2 held-out afterward.

The threshold rule is fixed and interpretable: choose the lowest verifier probability that permits at most 10% calibration negative-query accepts. It does not combine CLIP and verifier scores because no calibration-only evidence justified a weight. See `VERIFIER_RESULTS.md` and `VERIFIER_FAILURE_ANALYSIS.md` for the rejection decision.
