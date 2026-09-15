"use client";
import { apiFetch } from "./api";

import Image from "next/image";
import { FormEvent, useEffect, useState } from "react";

type Result = {
  timestamp: number;
  thumbnail: string;
  score: number;
  modality: string;
  text?: string;
};

export default function Search({
  videoId,
  api,
  seek,
}: {
  videoId: string;
  api: string;
  seek: (seconds: number) => void;
}) {
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [busy, setBusy] = useState(false);
  const [searched, setSearched] = useState(false);
  const [mode, setMode] = useState("auto");
  const [scoreType, setScoreType] = useState("");
  const [modalities, setModalities] = useState<string[]>([]);
  const [selectedRoute, setSelectedRoute] = useState("");
  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const response = await apiFetch(`${api}/videos/${videoId}/index`);
        if (!response.ok) throw new Error("Could not retrieve index status.");
        const data = await response.json();
        if (active) {
          setStatus(data.stage || data.status);
          if (data.error) setError(data.error);
        }
      } catch (problem) {
        if (active)
          setError(
            problem instanceof Error ? problem.message : "Index unavailable.",
          );
      }
    }
    void refresh();
    const timer = setInterval(refresh, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [api, videoId]);

  async function index() {
    setBusy(true);
    setError("");
    try {
      const response = await apiFetch(`${api}/videos/${videoId}/index`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Indexing failed.",
        );
      setStatus("indexing");
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Indexing failed.");
    } finally {
      setBusy(false);
    }
  }

  async function search(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResults([]);
    try {
      const response = await apiFetch(
        `${api}/videos/${videoId}/search?q=${encodeURIComponent(query)}&k=8&mode=${mode}`,
      );
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Search failed.",
        );
      setResults(data.results);
      setSearched(true);
      setScoreType(data.score_type);
      setModalities(data.modalities_used || [data.selected_route || mode]);
      setSelectedRoute(data.selected_route || mode);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Search failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="semantic-search">
      <div className="timeline-heading">
        <h3>Search moments</h3>
        <span aria-live="polite">Visual index: {status.replace("_", " ")}</span>
      </div>
      {(status === "not_started" || status === "failed") && (
        <>
          <p>
            Enable natural-language visual search. The first run downloads the
            local CLIP model (about 600 MB).
          </p>
          <button
            className="button"
            disabled={busy}
            onClick={() => void index()}
          >
            Build visual index
          </button>
        </>
      )}
      <label className="mode-label">
        Search in{" "}
        <select
          value={mode}
          onChange={(event) => {
            setMode(event.target.value);
            setResults([]);
            setSearched(false);
          }}
        >
          <option value="auto">Auto</option>
          <option value="visual">Visuals</option>
          <option value="speech">Speech</option>
          <option value="hybrid">Visuals + speech</option>
        </select>
      </label>
      <form onSubmit={(event) => void search(event)} className="search-form">
        <label className="search-label">
          Describe a moment
          <input
            required
            maxLength={500}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="A person sitting at a desk"
          />
        </label>
        <button className="button" disabled={busy || !query.trim()}>
          {busy ? "Searching…" : "Search"}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      {results.length > 0 && (
        <>
          <p className="hint">
            {scoreType.replaceAll("_", " ")} · Route: {selectedRoute} · Evidence:{" "}
            {modalities.join(" + ")} · Scores are not confidence.
          </p>
          <div className="results">
            {results.map((result) => (
              <button
                key={`${result.timestamp}-${result.modality}`}
                onClick={() => seek(result.timestamp)}
                className="result"
              >
                <Image
                  unoptimized
                  width={240}
                  height={135}
                  src={`${api}${result.thumbnail}`}
                  alt={`Match at ${result.timestamp.toFixed(1)} seconds`}
                />
                <div>
                  <span>
                    {Math.floor(result.timestamp / 60)}:
                    {Math.floor(result.timestamp % 60)
                      .toString()
                      .padStart(2, "0")}
                  </span>
                  <small>
                    {result.modality} ·{" "}
                    {result.score.toFixed(
                      scoreType === "reciprocal_rank_fusion" ? 5 : 3,
                    )}
                  </small>
                  {result.text && <p>{result.text}</p>}
                </div>
              </button>
            ))}
          </div>
        </>
      )}
      {searched && !results.length && !busy && !error && (
        <p>No matching moments found.</p>
      )}
    </section>
  );
}
