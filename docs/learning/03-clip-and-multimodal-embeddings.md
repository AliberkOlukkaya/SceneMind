# CLIP embeddings

CLIP trains image and text encoders to place matching image/caption pairs near each other. SceneMind uses this shared space to compare a natural-language query against sampled video frames without generating captions or calling a hosted API.

Model: openai/clip-vit-base-patch32, pinned at revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. Source: https://huggingface.co/openai/clip-vit-base-patch32 and https://github.com/openai/CLIP. CLIP code/weights are released under MIT; read the model card's intended-use and deployment limitations as well. No identity recognition is implemented.

Implementation: backend/app/encoder.py separates image preprocessing, text preprocessing and inference. Transformers' CLIPProcessor converts RGB images to the model's 224×224 resized/center-cropped normalized tensor and tokenizes text up to 77 tokens. Both encoders output 512-dimensional vectors. We L2-normalize them explicitly; cosine similarity is then the dot product.

The 151M-parameter baseline requires roughly 600 MB of weights plus the PyTorch runtime and inference memory. It is the first substantial download, justified by cross-modal retrieval. CPU batches default to eight; four Torch threads limit oversubscription. CUDA is configurable but not yet verified. Model and processor are cached once. Index metadata records model/revision; changing them requires reindexing.

Alternatives: OpenCLIP checkpoints, MobileCLIP, or a captioning model. This baseline is well-understood and replaceable at the encoder boundary. It represents static frames, not motion, temporal causality or factual truth. A picture of an open door does not establish that someone opened it. Brief events and off-center objects can be missed by sampling/cropping. Similarity is not a calibrated confidence probability.
