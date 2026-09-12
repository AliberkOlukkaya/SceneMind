"use client";
import { apiFetch } from "./api";
import Link from "next/link";
import Image from "next/image";
import Transcript from "./transcript";
import Search from "./search";
import { useEffect, useRef, useState } from "react";

type Video = {
  id: string;
  filename: string;
  status: string;
  error?: string;
  metadata?: { duration: number; width: number; height: number; fps: number };
  frames: { timestamp: number; thumbnail: string }[];
};
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const timestamp = (value: number) =>
  `${Math.floor(value / 60)}:${Math.floor(value % 60)
    .toString()
    .padStart(2, "0")}`;

export default function Home() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [error, setError] = useState("");
  const [serviceError, setServiceError] = useState("");
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const player = useRef<HTMLVideoElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const selected = videos.find((video) => video.id === selectedId);

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const response = await apiFetch(`${API}/videos`);
        if (!response.ok) throw new Error("Could not load the video library.");
        const data: Video[] = await response.json();
        if (active) {
          setVideos(data);
          setLoading(false);
          setServiceError("");
        }
      } catch {
        if (active) {
          setServiceError(
            "Cannot reach the video service. Check that the backend is running.",
          );
          setLoading(false);
        }
      }
    }
    void refresh();
    const timer = setInterval(refresh, 2500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  async function upload(file?: File) {
    if (!file) return;
    setError("");
    if (file.size > 250 * 1024 * 1024) {
      setError("Choose a video smaller than 250 MB.");
      return;
    }
    setUploading(true);
    try {
      const response = await apiFetch(
        `${API}/videos?filename=${encodeURIComponent(file.name)}`,
        { method: "POST", body: file },
      );
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Upload failed.",
        );
      setVideos((previous) => [
        data,
        ...previous.filter((item) => item.id !== data.id),
      ]);
      setSelectedId(data.id);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (input.current) input.current.value = "";
    }
  }

  return (
    <main>
      <header>
        <Link className="brand" href="/">
          SceneMind<span> / workspace</span>
        </Link>
        <span className="status">Local library</span>
      </header>
      <section className="heading toolbar">
        <div>
          <p className="eyebrow">YOUR WORKSPACE</p>
          <h1>Video library</h1>
          <p>
            {videos.length} {videos.length === 1 ? "video" : "videos"} · Browse
            sampled moments
          </p>
        </div>
        <div>
          <input
            ref={input}
            type="file"
            accept=".mp4,.mov,.webm,.mkv,.avi"
            aria-label="Choose video"
            className="sr-only"
            onChange={(event) => void upload(event.target.files?.[0])}
          />
          <button
            className="button"
            disabled={uploading}
            onClick={() => input.current?.click()}
          >
            {uploading ? "Uploading…" : "+ Upload video"}
          </button>
          <p className="hint">Up to 250 MB · 30 minutes · 4K</p>
        </div>
      </section>
      {serviceError && (
        <p role="alert" className="error">
          {serviceError}
        </p>
      )}
      {error && (
        <p role="alert" className="error">
          {error} <button onClick={() => setError("")}>Dismiss</button>
        </p>
      )}
      {loading ? (
        <section className="empty" aria-live="polite">
          Loading your library…
        </section>
      ) : videos.length === 0 ? (
        <section className="empty">
          <span className="film" aria-hidden="true">
            ?
          </span>
          <h2>Add your first video.</h2>
          <p>Upload a video to extract frames and browse its timeline.</p>
          <button
            className="button"
            disabled={uploading}
            onClick={() => input.current?.click()}
          >
            Choose a video
          </button>
        </section>
      ) : (
        <div className="workspace">
          <nav className="library" aria-label="Video library">
            {videos.map((video) => (
              <button
                className={`video-row ${selectedId === video.id ? "selected" : ""}`}
                key={video.id}
                onClick={() => setSelectedId(video.id)}
              >
                <span>{video.filename}</span>
                <small>
                  {video.status}{" "}
                  {video.metadata && `· ${timestamp(video.metadata.duration)}`}
                </small>
              </button>
            ))}
          </nav>
          <section className="viewer">
            {!selected ? (
              <div className="empty">
                <h2>Select a video</h2>
                <p>Open a video from your library to browse its moments.</p>
              </div>
            ) : (
              <>
                <h2 className="video-title">{selected.filename}</h2>
                <p aria-live="polite">
                  {selected.status === "ready"
                    ? `${selected.metadata?.width} × ${selected.metadata?.height} · ${selected.frames.length} sampled frames`
                    : selected.error || `Video ${selected.status}…`}
                </p>
                {selected.status === "ready" && (
                  <>
                    <video
                      key={selected.id}
                      ref={player}
                      controls
                      preload="metadata"
                      src={`${API}/videos/${selected.id}/media`}
                    />
                    <div className="timeline-heading">
                      <h3>Sampled moments</h3>
                      <span>Click a frame to seek</span>
                    </div>
                    <div className="frames">
                      {selected.frames.map((frame) => (
                        <button
                          key={frame.thumbnail}
                          className="frame"
                          onClick={() => {
                            if (player.current)
                              player.current.currentTime = frame.timestamp;
                          }}
                          aria-label={`Seek to ${timestamp(frame.timestamp)}`}
                        >
                          <Image
                            unoptimized
                            width={240}
                            height={135}
                            src={`${API}${frame.thumbnail}`}
                            alt={`Video frame at ${timestamp(frame.timestamp)}`}
                          />
                          <span>{timestamp(frame.timestamp)}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}
                {selected.status === "ready" && (
                  <>
                    <Search
                      key={`search-${selected.id}`}
                      videoId={selected.id}
                      api={API}
                      seek={(seconds) => {
                        if (player.current)
                          player.current.currentTime = seconds;
                      }}
                    />
                    <Transcript
                      key={selected.id}
                      videoId={selected.id}
                      api={API}
                      seek={(seconds) => {
                        if (player.current)
                          player.current.currentTime = seconds;
                      }}
                    />
                  </>
                )}
              </>
            )}
          </section>
        </div>
      )}
      <footer>
        SCENEMIND <span>Video processing runs locally.</span>
      </footer>
    </main>
  );
}
