"use client";

import { FormEvent, KeyboardEvent, useEffect, useState } from "react";
import Image from "next/image";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Stats = { total_reviews: number; sources: Record<string, number> };
type SourceReview = { review_id: string; text: string; source: string | null; rating: number | null };
type QueryResult = { answer: string; evidence_count: number; sources: SourceReview[] };

const examples = [
  "What information do users seek outside Myntra before purchasing?",
  "What unmet needs emerge consistently across user conversations?",
  "What prevents wishlisted products from eventually being purchased?",
];

const sourceNames: Record<string, string> = {
  play_store: "Play Store",
  app_store: "App Store",
  reddit: "Reddit",
  unknown: "Unknown",
};

const readableSource = (source: string) =>
  sourceNames[source] ?? source.replaceAll("_", " ");

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceReview[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/api/stats`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail ?? "Unable to load review counts.");
        return data as Stats;
      })
      .then(setStats)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  async function askQuestion(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setShowSources(false);
    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), top_k: 5 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Unable to analyse the reviews.");
      const result = data as QueryResult;
      setAnswer(result.answer);
      setSources(result.sources);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyboard(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !loading) {
      event.preventDefault();
      void askQuestion();
    }
  }

  return (
    <>
      <div className="brandRule" />
      <nav className="topNav" aria-label="Main navigation">
        <a className="wordmark" href="#top" aria-label="Myntra Review Analyser home">
          <Image className="logo" src="/myntra-logo.png" alt="Myntra" width={58} height={58} priority />
          <span className="wordmarkText">Myntra<br /><b>Insights</b></span>
        </a>
        <div className="navLinks">
          <a href="#sources">Sources</a>
          <a href="#questions">Questions</a>
          <a href="#analyser">Analyser</a>
        </div>
        <a className="navSearch" href="#analyser">
          <span aria-hidden="true">⌕</span>
          <span>Search customer reviews</span>
        </a>
        <div className="navStatus"><i /> AI analyser</div>
      </nav>

      <main className="page" id="top">
      <header className="hero">
        <span className="brandTag">Customer intelligence workspace</span>
        <h1>Myntra Review<br /><em>Analyser</em></h1>
        <p>Explore what customers need, value and struggle with across every review channel.</p>
        <a className="heroAction" href="#analyser">Ask a question <span>→</span></a>
      </header>

      <section className="contentSection" id="sources" aria-labelledby="sources-title">
        <div className="sectionHeading">
          <div><span className="sectionLabel">Data coverage</span><h2 id="sources-title">Review sources</h2></div>
          {stats && <span className="total">{stats.total_reviews.toLocaleString()} total reviews</span>}
        </div>
        <div className="sourceGrid">
          {stats ? Object.entries(stats.sources).map(([source, count]) => (
            <article className="sourceCard" key={source}>
              <span>{readableSource(source)}</span>
              <strong>{count.toLocaleString()}</strong>
              <small>reviews</small>
            </article>
          )) : <p className="muted">Loading review counts…</p>}
        </div>
      </section>

      <section className="contentSection tintedSection" id="questions" aria-labelledby="examples-title">
        <span className="sectionLabel">Try an example</span>
        <h2 id="examples-title">Questions you can explore</h2>
        <div className="exampleList">
          {examples.map((example, index) => (
            <button className="exampleButton" key={example} onClick={() => setQuestion(example)}>
              <span>{index + 1}</span><p>{example}</p><b aria-hidden="true">→</b>
            </button>
          ))}
        </div>
      </section>

      <section className="contentSection analyserSection" id="analyser" aria-labelledby="chat-title">
        <span className="sectionLabel">Ask the analyser</span>
        <h2 id="chat-title">What would you like to understand?</h2>
        <form className="chatForm" onSubmit={askQuestion}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyboard}
            placeholder="Ask a question about the customer reviews…"
            rows={4}
            maxLength={500}
            disabled={loading}
            aria-label="Question"
          />
          <div className="chatActions">
            <small>Enter to send · Shift + Enter for a new line</small>
            <button className="submitButton" disabled={loading || !question.trim()}>
              {loading ? "Analysing…" : "Analyse reviews"}
            </button>
          </div>
        </form>

        {error && <p className="error" role="alert">{error}</p>}
        {loading && <div className="answerCard loading"><span className="spinner" />Consolidating relevant reviews…</div>}
        {answer && !loading && (
          <article className="answerCard">
            <span className="sectionLabel">Consolidated insight</span>
            <p className="answerText">{answer}</p>
            {sources.length > 0 && (
              <div className="evidence">
                <button className="sourceLink" onClick={() => setShowSources((current) => !current)} aria-expanded={showSources}>
                  Sources ({sources.length}) {showSources ? "↑" : "↓"}
                </button>
                {showSources && <div className="reviewList">
                  <p className="muted">Reviews used to generate this consolidated answer.</p>
                  {sources.map((review) => (
                    <article className="reviewCard" key={review.review_id}>
                      <div><b>{readableSource(review.source ?? "unknown")}</b>{review.rating !== null && <span>{review.rating}/5</span>}</div>
                      <p>{review.text}</p>
                    </article>
                  ))}
                </div>}
              </div>
            )}
          </article>
        )}
      </section>
      </main>
      <footer><b>Myntra Review Analyser</b><span>Grounded insights from customer feedback</span></footer>
    </>
  );
}
