"use client";

import { FormEvent, useEffect, useState } from "react";
import { apiFetch } from "./api";

type Citation = {
  evidence_id: string;
  start_seconds: number;
  end_seconds: number;
  text: string;
};

type Answer = {
  answerable: boolean;
  answer: string;
  citations: Citation[];
};

const timestamp = (value: number) =>
  `${Math.floor(value / 60)}:${Math.floor(value % 60)
    .toString()
    .padStart(2, "0")}`;

export default function AskVideo({
  videoId,
  api,
  seek,
}: {
  videoId: string;
  api: string;
  seek: (seconds: number) => void;
}) {
  const [configured, setConfigured] = useState<boolean>();
  const [enabled, setEnabled] = useState<boolean>();
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    apiFetch(`${api}/videos/${videoId}/ask/status`)
      .then(async (response) => {
        if (!response.ok)
          throw new Error("Could not check Ask Video configuration.");
        const data = await response.json();
        if (active) {
          setEnabled(Boolean(data.enabled));
          setConfigured(Boolean(data.configured));
        }
      })
      .catch((problem) => {
        if (active)
          setError(
            problem instanceof Error
              ? problem.message
              : "Ask Video unavailable.",
          );
      });
    return () => {
      active = false;
    };
  }, [api, videoId]);

  async function ask(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setAnswer(undefined);
    try {
      const response = await apiFetch(`${api}/videos/${videoId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Ask Video failed.",
        );
      setAnswer(data);
    } catch (problem) {
      setError(
        problem instanceof Error ? problem.message : "Ask Video failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ask-video">
      <div className="timeline-heading">
        <div>
          <p className="eyebrow">ASK VIDEO</p>
          <h3>Answer from the transcript</h3>
        </div>
        <span>Transcript evidence only</span>
      </div>
      <p>
        Ask about what is said. Every substantive answer must cite the video
        moments that support it.
      </p>
      {enabled === false && (
        <p role="status" className="configuration-note">
          Ask Video remains evaluation-gated until real provider validation
          passes.
        </p>
      )}
      {enabled !== false && configured === false && (
        <p role="status" className="configuration-note">
          Ask Video is not configured. Add OPENAI_API_KEY to the local
          .env.local file and restart the backend.
        </p>
      )}
      <form className="ask-form" onSubmit={(event) => void ask(event)}>
        <label>
          Ask something about this video
          <textarea
            required
            maxLength={500}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What are the speaker's main recommendations?"
          />
        </label>
        <button
          className="button"
          disabled={busy || !question.trim() || configured !== true}
        >
          {busy ? "Finding evidence…" : "Ask"}
        </button>
      </form>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {answer && (
        <div className="qa-answer" aria-live="polite">
          <h4>{answer.answerable ? "Answer" : "Not enough evidence"}</h4>
          <p>{answer.answer}</p>
          {answer.citations.length > 0 && (
            <>
              <h4>Sources</h4>
              <div className="qa-sources">
                {answer.citations.map((citation, index) => (
                  <button
                    key={citation.evidence_id}
                    onClick={() => seek(citation.start_seconds)}
                    aria-label={`Source ${index + 1}, seek to ${timestamp(citation.start_seconds)}`}
                  >
                    <strong>
                      [{index + 1}] {timestamp(citation.start_seconds)}–
                      {timestamp(citation.end_seconds)}
                    </strong>
                    <span>{citation.text}</span>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
