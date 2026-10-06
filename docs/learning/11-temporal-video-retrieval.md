# Temporal video retrieval

SceneMind V1 embeds one JPEG every five seconds. A still frame can describe a
door or a person, but cannot reliably distinguish opening from closing the
door. More frequent still images improve coverage; they do not automatically
teach an encoder motion or event order. Object detection instead locates known
object classes. Neither operation is the same as comparing an entire short
video sequence to a natural-language description.

The isolated V2 experiment splits a video into overlapping 4- or 8-second
windows and samples eight ordered RGB frames inside each. A pretrained
X-CLIP encoder mixes information across those frames using cross-frame attention
and a multiframe integration transformer. It emits a 512-number video vector.
Its pretrained text encoder represents the query. The full model also adapts
the text vector using video patch features; removing that module changes the
model's score. We cache both the video vector and 49 projected patch vectors
per window, then apply the original text adaptation at query time. The score
is cosine similarity after normalization. Scores are ranking signals, not
probabilities that an answer is correct.

This candidate consequently needs more than one shared text embedding and a
plain FAISS lookup. It exhaustively scores the windows from one selected video.
Persisted NumPy archives contain no pickle and bind the model revision, source
hash, temporal settings and timestamps. None of these indexes is connected to
production Find Moments, Smart Search, speech, OCR or Ask Video.

Model: `microsoft/xclip-base-patch32`, revision
`a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`, 196,585,729 measured parameters,
MIT model-card license. Transformers 4.57.6 supplies the implementation under
Apache-2.0. Preprocessing uses the pinned VideoMAE image processor: RGB,
short-edge resize to 224, center crop 224, rescaling and ImageNet channel
normalization. Text uses the paired CLIP tokenizer, max77 tokens. The
checkpoint was trained on Kinetics-400, which makes domain transfer an open
question. We perform inference only, FP32 CPU and four Torch threads, video
batch one. PyTorch supports GPU placement, but this experiment's primary
deployment target is CPU; no GPU result has been measured.

The initial engineering smoke used fixed random pixels, not evaluation videos.
Its cached cosine exactly matched native model forward on that input. Model
load was 2.36 seconds, one eight-frame window took 0.47 seconds, and Windows
process peak working set was 1.03 GiB. Those numbers do not establish search
quality, long-video feasibility, or a minimum server specification.

Recall@5 asks whether at least one of the first five returned timestamps falls
inside any pre-retrieval frozen valid interval. The candidate's displayed
timestamp is its window midpoint. A window merely overlapping an event does
not earn credit. Repeated actions need all valid intervals, or a correct result
can be mislabeled as a miss. Ground truth must be audited before rankings are
seen. Sampling failure and representation failure need separate visual review:
an interval overlap alone does not prove the relevant pixels were sampled.

The protocol permits only two temporal settings and one model. Development
selects the setting; a wholly new frozen validation set decides whether the
architecture deserves integration. ViCLIP-B offers an attractive independent
dual encoder, but checkpoint license metadata was unresolved during research.
MobileViCLIP has a newer standalone implementation and base-weight licensing
questions. They are research alternatives, not additional evaluated models.

**Final finding: Decision C; reject this candidate for integration.**
On five new sources / 66 queries, X-CLIP R@5 is 25.76% versus single-frame
CLIP 60.61%. Action/interaction/temporal queries also regress (26.19% versus
52.38%). Actual sampled pixels show that most candidate misses are ranking
failures despite available evidence. Temporal input alone does not establish
useful video retrieval. Completed-stage CPU indexing is 40.38 seconds/minute
and prompt-feature storage is 3.09 MB/minute; full-attempt cost and whole-run
RSS are unavailable because of interrupted execution. No production integration
or further model experiment is authorized by these results. See [protocol](../../ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_PROTOCOL.md)
and [results](../../ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_RESULTS.md).

Sources: [model card](https://huggingface.co/microsoft/xclip-base-patch32),
[original implementation](https://github.com/microsoft/VideoX/tree/master/X-CLIP),
[Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/x_clip/modeling_x_clip.py).

The later historical replay was allowed only after Decision C was frozen. On
six short older videos, X-CLIP recovered 15/42 frozen intervals at Top-5 versus
CLIP's 31/42. Old repeated-moment annotations are incomplete and cannot be
used for tuning or precise user-success claims. This result supports the new
validation diagnosis but does not replace it.
