"use client";
import { apiFetch } from "./api";

import { useEffect, useState } from "react";

type TranscriptData = {
  status: string;
  stage?: string;
  error?: string;
  segments: { start: number; end: number; text: string }[];
};

const TRANSCRIPT_STATUS: Record<string, string> = {
  loading: "Checking",
  not_started: "Not created",
  queued: "Waiting to transcribe",
  processing: "Transcribing speech",
  transcribing: "Transcribing speech",
  ready: "Ready",
  failed: "Failed",
};

export default function Transcript({
  videoId,
  api,
  seek,
}: {
  videoId: string;
  api: string;
  seek: (seconds: number) => void;
}) {
  const [data, setData] = useState<TranscriptData>({
    status: "loading",
    segments: [],
  });
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const response = await apiFetch(
          `${api}/videos/${videoId}/transcript?q=${encodeURIComponent(query)}`,
        );
        if (!response.ok) throw new Error("Could not retrieve transcript.");
        const transcript: TranscriptData = await response.json();
        if (active) setData(transcript);
      } catch (problem) {
        if (active)
          setError(
            problem instanceof Error
              ? problem.message
              : "Transcript unavailable.",
          );
      }
    }
    void refresh();
    const timer = setInterval(refresh, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [api, videoId, query]);

  async function start() {
    setStarting(true);
    setError("");
    try {
      const response = await apiFetch(`${api}/videos/${videoId}/transcript`, {
        method: "POST",
      });
      const result = await response.json();
      if (!response.ok)
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : "Could not start transcription.",
        );
      setData({ status: "processing", stage: "transcribing", segments: [] });
    } catch (problem) {
      setError(
        problem instanceof Error ? problem.message : "Transcription failed.",
      );
    } finally {
      setStarting(false);
    }
  }

  return (
    <section className="transcript">
      <div className="timeline-heading">
        <h3>Transcript</h3>
        <span aria-live="polite">
          {TRANSCRIPT_STATUS[data.stage || data.status] ||
            (data.stage || data.status).replaceAll("_", " ")}
        </span>
      </div>
      {(data.status === "not_started" || data.status === "failed") && (
        <>
          <p>
            Create a searchable transcript. The first run downloads a small
            local speech model.
          </p>
          <button
            className="button"
            disabled={starting}
            onClick={() => void start()}
          >
            {starting ? "Starting…" : "Transcribe video"}
          </button>
        </>
      )}
      {(error || data.error) && (
        <p role="alert" className="error">
          {error || data.error}
        </p>
      )}
      {data.status === "ready" && (
        <>
          <label className="search-label">
            Find spoken text
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search the transcript"
              maxLength={500}
            />
          </label>
          <div className="segments">
            {data.segments.length ? (
              data.segments.map((segment) => (
                <button
                  key={`${segment.start}-${segment.end}`}
                  onClick={() => seek(segment.start)}
                >
                  <time>
                    {Math.floor(segment.start / 60)}:
                    {Math.floor(segment.start % 60)
                      .toString()
                      .padStart(2, "0")}
                  </time>
                  <span>{segment.text}</span>
                </button>
              ))
            ) : (
              <p>
                {query
                  ? "No transcript excerpts returned."
                  : "No speech detected."}
              </p>
            )}
          </div>
        </>
      )}
    </section>
  );
}
