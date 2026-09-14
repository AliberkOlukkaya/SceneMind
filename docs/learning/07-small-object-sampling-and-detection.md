# Small-object sampling and detection

## Why two measurements are required

A detector cannot recognize an object that the sampler omitted. This experiment first labels the actual stored JPEG, then evaluates a detector only on frames where a human can identify the target. The first metric is event-level visible-evidence recall. The second is detector recall conditional on a verified-visible target.

## Sampling and CLIP

Input is decoded video. FFmpeg selects the first frame and then the first frame at least 5, 2 or 1 seconds after the previous selection, scales it to 480 pixels wide and writes JPEG quality level 3. Output is a timestamped frame sequence. CLIP ViT-B/32 consumes RGB crops produced by its pinned processor and emits a normalized 512-float vector per frame. SceneMind stores those vectors in NumPy and performs exact inner-product search with FAISS.

Denser sampling increases the chance of retaining a brief object, but also increases JPEG and vector storage, image-encoding time and the number of distractors competing for a fixed top five. It does not guarantee higher top-five retrieval recall.

## YOLOX-Nano

Input is a BGR JPEG letterboxed with value 114 to 416, 640 or 768 square pixels. Pixel values remain on the 0–255 scale expected by the official export. The network outputs center, size, objectness and 80 COCO class scores at strides 8, 16 and 32. The experiment decodes boxes, multiplies objectness by class score, keeps evidence at 0.01 and applies class-aware NMS at 0.45. Output evidence contains class, confidence, source-pixel box and box-area ratio.

The source is the official YOLOX `0.1.1rc0` Nano checkpoint, Apache-2.0, 0.91M parameters. CPU uses ONNX Runtime with four intra-op threads; GPU-compatible ONNX providers are possible but unmeasured. Alternatives include NanoDet-Plus and RTMDet-tiny. They were not chosen because reproducing their pinned Windows CPU deployment adds a larger conversion stack.

## RT-DETR-R18

Input is RGB resized/padded by the pinned Transformers image processor to 640 square pixels. The 20M-parameter transformer detector produces 300 object queries with COCO labels, confidence and boxes; the experiment postprocesses them back to source pixels at a 0.01 evidence floor. The model and official implementation are Apache-2.0. The pinned Hugging Face artifact is `PekingU/rtdetr_r18vd` revision `6401be7fee8b49ee00b42fcc4e0064bba8061777`.

CPU inference uses PyTorch with four threads. CUDA is supported by the model family but was not measured. RTMDet-tiny is the most relevant lighter alternative at 640 input, while RT-DETR-R18 was selected here because a directly pinned official safetensors artifact and native Transformers runtime made the comparison reproducible.

## Reading the metrics

Raw class presence can be a false positive when another object of the same class appears. The final recall therefore counts a detection only after its best box is manually verified to cover the target. Confidence thresholds of 0.01, 0.10, 0.25 and 0.50 are reported without fitting on held-out data. Box-area ratio describes the detected target's scale, not segmentation area.

CPU latency is measured after one warm pass. p95 spans the verified-visible frames and repeated passes. Added peak RSS is sampled from an isolated process. These local measurements should not be substituted for vendor GPU throughput claims.
