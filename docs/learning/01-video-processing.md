# Video processing

A video container holds encoded streams. Decoding reconstructs images; sampling keeps a small set of those images for browsing and later inference.

SceneMind accepts a raw video body at POST /videos?filename=sample.mp4. It streams bytes into a generated UUID directory, validates decodable video metadata with OpenCV, then runs FFmpeg to select frames and resize them to 480 pixels wide. Output: an atomic JSON manifest containing duration, FPS, dimensions, codec, status and timestamped JPEG paths.

Implementation: backend/app/video.py. Configuration: backend/app/config.py. Sampling defaults to five seconds. FFmpeg's showinfo filter records actual selected presentation timestamps, avoiding the assumption that every source has an exact integer frame rate. We select the first frame and then frames at least one interval apart. This is sampling, not scene or action detection; brief events can be missed. OpenCV duration is frame-count/FPS and is approximate for variable-frame-rate media.

Default configurable limits: 1 GiB, 60 minutes, 4K, one ingestion at a time, and a 30-minute FFmpeg timeout. Upload chunks write directly to disk with byte, free-space and processing-headroom guards. Frames are staged before promotion so failed extraction cannot appear ready. FFmpeg comes from imageio-ffmpeg wheels; IMAGEIO_FFMPEG_EXE can override its path. Some platforms may need a separately installed binary. Check FFmpeg build licensing before redistributing binaries; this repository does not vendor one.

Alternatives: ffprobe for richer authoritative stream metadata; PyAV for timestamp-aware decoding; scene-change sampling for fewer redundant images. The current approach is small and CPU-compatible. No AI model runs here. A junior engineer should distinguish container, codec, frame rate, presentation timestamp and sampling interval; none of these establishes semantic meaning.
