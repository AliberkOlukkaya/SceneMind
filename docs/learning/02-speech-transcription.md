# Timestamped speech

Automatic speech recognition maps audio to text. SceneMind extracts mono 16 kHz PCM audio with FFmpeg, then uses faster-whisper to produce segments with start/end seconds and text. This supports literal transcript search and seeking to speech in the source video.

Implementation: backend/app/speech.py. Storage: backend/app/database.py, with the initial schema in backend/migrations/versions/001_transcripts.py. GET /videos/{id}/transcript returns ordered segments; optional q performs case-insensitive literal substring matching, not semantic search. POST starts or retries transcription. Video processing remains usable without installing the speech extra.

Model: Systran/faster-whisper-tiny, a CTranslate2 conversion of Whisper tiny. Sources: https://github.com/SYSTRAN/faster-whisper and https://huggingface.co/Systran/faster-whisper-tiny. Both publish MIT licensing; retain required notices when redistributing. This uses local open weights, not the OpenAI API.

Input: decoded audio waveform, internally transformed into log-Mel features. Output: language estimate and timestamped text segments. There is no retrieval embedding dimension. Defaults: CPU, INT8, four CPU threads, beam size five, voice activity detection enabled. Model construction is cached once per process; first explicit transcription downloads model files into ignored data/models. Set SCENEMIND_SPEECH_MODEL to a local model directory for offline use. A CUDA device and compatible compute type/libraries can be configured; CPU is the tested development path.

Tiny favors cost and startup size over accuracy, especially for accents, technical terms and noisy audio. Larger Whisper variants and whisper.cpp were considered; defer larger downloads until evaluation justifies them. Voice activity detection reduces silence hallucinations but does not eliminate them. Timestamps and words are estimates, not ground truth. The inference iterator is lazy and must be consumed inside the job. Failed runs preserve explicit error status; retries replace segments transactionally.
