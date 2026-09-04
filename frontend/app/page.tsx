"use client";

import Image from "next/image";
import { FormEvent, KeyboardEvent, useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Stats = { total_reviews: number; sources: Record<string, number> };
type SentimentValue = { count: number; percent: number };
type Category = { category: string; label: string; review_count: number; review_share_percent: number; sentiments: Record<"positive" | "negative" | "neutral", SentimentValue>; dominant_sentiment: string; summary: string };
type Opportunity = { rank: number; barrier: string; mentions: number; wishlist_review_share_percent: number; negative_rate_percent: number; opportunity_score: number; suggested_improvement: string };
type QuestionSample = { review_id: string; source: string; sentiment: string; text: string };
type QuestionAnswer = { question: string; answer: string; dominant_sentiment: string; samples: QuestionSample[] };
type Analysis = { overview: { total_reviews: number; overall_dominant_sentiment: string }; categories: Category[]; wishlist_opportunities: Opportunity[]; questions_answered: QuestionAnswer[] };
type SourceReview = { review_id: string; text: string; source: string | null; rating: number | null };
type QueryResult = { answer: string; evidence_count: number; sources: SourceReview[] };

const examples = ["What information do users seek outside Myntra before purchasing?", "What unmet needs emerge consistently across user conversations?", "What prevents wishlisted products from eventually being purchased?"];
const sourceNames: Record<string, string> = { play_store: "Play Store", app_store: "App Store", reddit: "Reddit", unknown: "Other" };
const barrierNames: Record<string, string> = { wishlist_reliability: "Wishlist reliability", stock_and_size_availability: "Stock & size availability", price_and_discount_transparency: "Price & discount clarity", price_drop_communication: "Price-drop communication", product_confidence: "Product confidence", wishlist_organization: "Wishlist organisation", cart_and_checkout_friction: "Cart & checkout friction", decision_support: "Decision support", cross_device_continuity: "Cross-device continuity", purchase_reminders: "Purchase reminders" };
const readable = (value: string) => sourceNames[value] ?? barrierNames[value] ?? value.replaceAll("_", " ");

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [dataError, setDataError] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceReview[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [loading, setLoading] = useState(false);
  const [queryError, setQueryError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/api/stats`).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail ?? "Unable to load review counts."); return data as Stats; }),
      fetch(`${API_URL}/api/analysis`).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail ?? "Unable to load review analysis."); return data as Analysis; }),
    ]).then(([statsData, analysisData]) => { setStats(statsData); setAnalysis(analysisData); }).catch((error: Error) => setDataError(error.message));
  }, []);

  async function askQuestion(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true); setQueryError(""); setAnswer(""); setSources([]); setShowSources(false);
    try {
      const response = await fetch(`${API_URL}/api/query`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: question.trim(), top_k: 5 }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Unable to analyse the reviews.");
      const result = data as QueryResult; setAnswer(result.answer); setSources(result.sources);
    } catch (error) { setQueryError(error instanceof Error ? error.message : "An unexpected error occurred."); }
    finally { setLoading(false); }
  }

  function handleKeyboard(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !loading) { event.preventDefault(); void askQuestion(); }
  }

  return <>
    <div className="brandRule" />
    <nav className="topNav" aria-label="Dashboard navigation">
      <a className="wordmark" href="#top" aria-label="Myntra Review Analyser home"><Image className="logo" src="/myntra-logo.png" alt="Myntra" width={58} height={58} priority /><span>Myntra <b>Insights</b></span></a>
      <div className="navLinks"><a href="#sources">Sources</a><a href="#categories">Categories</a><a href="#questions-answered">Questions</a><a href="#opportunities">Wishlist</a><a href="#analyser">Analyser</a></div>
      <a className="navSearch" href="#analyser"><span>⌕</span> Search customer reviews</a><div className="navStatus"><i /> Analysis ready</div>
    </nav>
    <main id="top">
      <header className="dashboardHeader"><div><span className="eyebrow">Customer intelligence workspace</span><h1>Myntra Review Analyser</h1><p>Customer sentiment, recurring issues and wishlist opportunities across every review channel.</p></div>
        {analysis && <div className={`overallSentiment sentiment-${analysis.overview.overall_dominant_sentiment}`}><small>Overall sentiment</small><strong>{analysis.overview.overall_dominant_sentiment}</strong></div>}</header>
      {dataError && <p className="error dataError" role="alert">{dataError}</p>}

      <section className="contentSection" id="sources" aria-labelledby="sources-title"><div className="sectionHeading"><div><span className="eyebrow">Data coverage</span><h2 id="sources-title">Review sources</h2></div>{stats && <span className="total">{stats.total_reviews.toLocaleString()} unique reviews</span>}</div>
        <div className="sourceGrid">{stats ? Object.entries(stats.sources).map(([source, count], index) => <article className="sourceCard" key={source}><div className={`sourceIcon sourceIcon${index + 1}`}>{readable(source).charAt(0)}</div><div><span>{readable(source)}</span><strong>{count.toLocaleString()}</strong><small>reviews analysed</small></div></article>) : [1, 2, 3].map((item) => <div className="skeleton sourceSkeleton" key={item} />)}</div></section>

      <section className="contentSection categorySection" id="categories" aria-labelledby="categories-title"><div className="sectionHeading"><div><span className="eyebrow">Sentiment by theme</span><h2 id="categories-title">Review categories</h2></div><span className="legend"><i className="positiveDot" /> Positive <i className="negativeDot" /> Negative <i className="neutralDot" /> Neutral</span></div>
        <div className="categoryGrid">{analysis ? analysis.categories.map((category, index) => <article className="categoryCard" key={category.category}>
          <div className="categoryTop"><span className="categoryNumber">0{index + 1}</span><span className={`sentimentBadge sentiment-${category.dominant_sentiment}`}>{category.dominant_sentiment}</span></div><h3>{category.label}</h3>
          <div className="categoryCount"><strong>{category.review_count.toLocaleString()}</strong><span>{category.review_share_percent}% of reviews</span></div>
          <div className="sentimentBar" aria-label={`${category.label} sentiment distribution`}><i className="positiveBar" style={{ width: `${category.sentiments.positive.percent}%` }} /><i className="negativeBar" style={{ width: `${category.sentiments.negative.percent}%` }} /><i className="neutralBar" style={{ width: `${category.sentiments.neutral.percent}%` }} /></div>
          <div className="sentimentValues"><span><b>{category.sentiments.positive.percent}%</b> positive</span><span><b>{category.sentiments.negative.percent}%</b> negative</span><span><b>{category.sentiments.neutral.percent}%</b> neutral</span></div>
          <div className="summary"><small>Dominant sentiment summary</small><p>{category.summary}</p></div></article>) : [1, 2, 3, 4, 5, 6].map((item) => <div className="skeleton categorySkeleton" key={item} />)}</div></section>

      <section className="contentSection questionsSection" id="questions-answered" aria-labelledby="questions-answered-title">
        <div className="sectionHeading"><div><span className="eyebrow">Dominant-sentiment evidence</span><h2 id="questions-answered-title">Questions answered</h2></div><span className="total">Two supporting reviews per answer</span></div>
        <div className="answeredList">{analysis ? (analysis.questions_answered ?? []).map((item, index) => <article className="answeredCard" key={item.question}>
          <div className="questionIndex">{String(index + 1).padStart(2, "0")}</div><div className="answeredBody"><div className="answeredHeading"><h3>{item.question}</h3><span className={`sentimentBadge sentiment-${item.dominant_sentiment}`}>{item.dominant_sentiment}</span></div><p className="questionAnswer">{item.answer}</p>
          <details className="questionEvidence"><summary>View 2 supporting reviews</summary><div className="questionSamples">{item.samples.map((sample, sampleIndex) => <blockquote key={`${sample.review_id}-${sampleIndex}`}><div><b>{readable(sample.source)}</b><span>{sample.sentiment}</span></div><p>{sample.text}</p></blockquote>)}</div></details></div>
        </article>) : [1, 2, 3].map((item) => <div className="skeleton answeredSkeleton" key={item} />)}
        {analysis && !(analysis.questions_answered ?? []).length && <p className="empty">Run the updated analysis script to generate these answers.</p>}</div>
      </section>

      <section className="contentSection opportunitySection" id="opportunities" aria-labelledby="opportunities-title"><div className="sectionHeading"><div><span className="eyebrow">Conversion opportunities</span><h2 id="opportunities-title">Improve wishlist to purchase</h2></div><span className="total">Ranked from review evidence</span></div>
        <div className="opportunityList">{analysis ? analysis.wishlist_opportunities.slice(0, 7).map((opportunity) => <article className="opportunityCard" key={opportunity.barrier}><span className="rank">#{opportunity.rank}</span><div className="opportunityMain"><h3>{readable(opportunity.barrier)}</h3><p>{opportunity.suggested_improvement}</p></div><div className="opportunityMetrics"><span><b>{opportunity.mentions}</b> mentions</span><span><b>{opportunity.wishlist_review_share_percent}%</b> share</span><span><b>{opportunity.negative_rate_percent}%</b> negative</span></div><div className="score"><small>Score</small><strong>{opportunity.opportunity_score}</strong></div></article>) : [1, 2, 3].map((item) => <div className="skeleton opportunitySkeleton" key={item} />)}
          {analysis?.wishlist_opportunities.length === 0 && <p className="empty">No wishlist-related evidence was found.</p>}</div></section>

      <section className="contentSection analyserSection" id="analyser" aria-labelledby="chat-title"><div className="sectionHeading"><div><span className="eyebrow">Ask the analyser</span><h2 id="chat-title">Explore the customer voice</h2></div></div>
        <div className="exampleChips">{examples.map((example) => <button key={example} onClick={() => setQuestion(example)}>{example}</button>)}</div>
        <form className="chatForm" onSubmit={askQuestion}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleKeyboard} placeholder="Ask a question about the customer reviews…" rows={4} maxLength={500} disabled={loading} aria-label="Question" /><div className="chatActions"><small>Enter to send · Shift + Enter for a new line</small><button className="submitButton" disabled={loading || !question.trim()}>{loading ? "Analysing…" : "Analyse reviews"}</button></div></form>
        {queryError && <p className="error" role="alert">{queryError}</p>}{loading && <div className="answerCard loading"><span className="spinner" />Consolidating relevant reviews…</div>}
        {answer && !loading && <article className="answerCard"><span className="eyebrow">Consolidated insight</span><p className="answerText">{answer}</p>{sources.length > 0 && <div className="evidence"><button className="sourceLink" onClick={() => setShowSources(!showSources)} aria-expanded={showSources}>Sources ({sources.length}) {showSources ? "↑" : "↓"}</button>{showSources && <div className="reviewList">{sources.map((review) => <article className="reviewCard" key={review.review_id}><div><b>{readable(review.source ?? "unknown")}</b>{review.rating !== null && <span>{review.rating}/5</span>}</div><p>{review.text}</p></article>)}</div>}</div>}</article>}
      </section>
    </main>
    <footer><b>Myntra Review Analyser</b><span>Review-derived signals should be validated with behavioural data.</span></footer>
  </>;
}
