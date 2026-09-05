"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Stats = { total_reviews: number; sources: Record<string, number> };
type Category = { category: string; label: string; review_count: number; review_share_percent: number; issues_summary: string };
type EvidenceReview = { review_id: string; source: string; text: string };
type Opportunity = { rank: number; barrier: string; mentions: number; wishlist_review_share_percent: number; opportunity_score: number; suggested_improvement: string; samples: EvidenceReview[] };
type QuestionAnswer = { question: string; answer: string; samples: EvidenceReview[] };
type Analysis = { overview: { total_reviews: number }; categories: Category[]; wishlist_opportunities: Opportunity[]; questions_answered: QuestionAnswer[] };

const sourceNames: Record<string, string> = { play_store: "Play Store", app_store: "App Store", reddit: "Reddit", unknown: "Other" };
const barrierNames: Record<string, string> = { wishlist_reliability: "Wishlist reliability", stock_and_size_availability: "Stock & size availability", price_and_discount_transparency: "Price & discount clarity", price_drop_communication: "Price-drop communication", product_confidence: "Product confidence", wishlist_organization: "Wishlist organisation", cart_and_checkout_friction: "Cart & checkout friction", decision_support: "Decision support", cross_device_continuity: "Cross-device continuity", purchase_reminders: "Purchase reminders" };
const readable = (value: string) => sourceNames[value] ?? barrierNames[value] ?? value.replaceAll("_", " ");

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [dataError, setDataError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/api/stats`).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail ?? "Unable to load review counts."); return data as Stats; }),
      fetch(`${API_URL}/api/analysis`).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail ?? "Unable to load review analysis."); return data as Analysis; }),
    ]).then(([statsData, analysisData]) => { setStats(statsData); setAnalysis(analysisData); }).catch((error: Error) => setDataError(error.message));
  }, []);

  return <>
    <div className="brandRule" />
    <nav className="topNav" aria-label="Dashboard navigation">
      <a className="wordmark" href="#top" aria-label="Myntra Review Analyser home"><Image className="logo" src="/myntra-logo.png" alt="Myntra" width={58} height={58} priority /><span>Myntra <b>Insights</b></span></a>
      <div className="navLinks"><a href="#sources">Sources</a><a href="#categories">Categories</a><a href="#questions-answered">Questions</a><a href="#opportunities">Wishlist</a></div>
      <div className="navSearch"><span>⌕</span> Customer review intelligence</div><div className="navStatus"><i /> Analysis ready</div>
    </nav>
    <main id="top">
      <header className="dashboardHeader"><div><span className="eyebrow">Customer intelligence workspace</span><h1>Myntra Review Analyser</h1><p>Recurring customer issues, distinctive purchase behaviours and evidence-backed wishlist opportunities across every review channel.</p></div></header>
      {dataError && <p className="error dataError" role="alert">{dataError}</p>}

      <section className="contentSection" id="sources" aria-labelledby="sources-title"><div className="sectionHeading"><div><span className="eyebrow">Data coverage</span><h2 id="sources-title">Review sources</h2></div>{stats && <span className="total">{stats.total_reviews.toLocaleString()} unique reviews</span>}</div>
        <div className="sourceGrid">{stats ? Object.entries(stats.sources).map(([source, count], index) => <article className="sourceCard" key={source}><div className={`sourceIcon sourceIcon${index + 1}`}>{readable(source).charAt(0)}</div><div><span>{readable(source)}</span><strong>{count.toLocaleString()}</strong><small>reviews analysed</small></div></article>) : [1, 2, 3].map((item) => <div className="skeleton sourceSkeleton" key={item} />)}</div></section>

      <section className="contentSection categorySection" id="categories" aria-labelledby="categories-title"><div className="sectionHeading"><div><span className="eyebrow">Issues by theme</span><h2 id="categories-title">Review categories</h2></div><span className="total">Customer friction summarised by category</span></div>
        <div className="categoryGrid">{analysis ? analysis.categories.map((category, index) => <article className="categoryCard issueCategoryCard" key={category.category}>
          <div className="categoryTop"><span className="categoryNumber">0{index + 1}</span><span className="issueBadge">ISSUES</span></div><h3>{category.label}</h3>
          <div className="categoryCount"><strong>{category.review_count.toLocaleString()}</strong><span>{category.review_share_percent}% of reviews</span></div>
          <div className="summary"><small>Issues people face</small><p>{category.issues_summary}</p></div></article>) : [1, 2, 3, 4, 5, 6].map((item) => <div className="skeleton categorySkeleton" key={item} />)}</div></section>

      <section className="contentSection questionsSection" id="questions-answered" aria-labelledby="questions-answered-title"><div className="sectionHeading"><div><span className="eyebrow">Distinctive review evidence</span><h2 id="questions-answered-title">Questions answered</h2></div><span className="total">Unique, high-information reviews prioritised</span></div>
        <div className="answeredList">{analysis ? (analysis.questions_answered ?? []).map((item, index) => <article className="answeredCard" key={item.question}>
          <div className="questionIndex">{String(index + 1).padStart(2, "0")}</div><div className="answeredBody"><div className="answeredHeading"><h3>{item.question}</h3></div><p className="questionAnswer">{item.answer}</p>
          <details className="questionEvidence"><summary>View 2 distinctive supporting reviews</summary><div className="questionSamples">{item.samples.map((sample, sampleIndex) => <blockquote key={`${sample.review_id}-${sampleIndex}`}><div><b>{readable(sample.source)}</b></div><p>{sample.text}</p></blockquote>)}</div></details></div>
        </article>) : [1, 2, 3].map((item) => <div className="skeleton answeredSkeleton" key={item} />)}
        {analysis && !(analysis.questions_answered ?? []).length && <p className="empty">Run the updated analysis script to generate these answers.</p>}</div></section>

      <section className="contentSection opportunitySection" id="opportunities" aria-labelledby="opportunities-title"><div className="sectionHeading"><div><span className="eyebrow">Conversion opportunities</span><h2 id="opportunities-title">Improve wishlist to purchase</h2></div><span className="total">Ranked from review evidence</span></div>
        <div className="opportunityList">{analysis ? analysis.wishlist_opportunities.slice(0, 7).map((opportunity) => <article className="opportunityCard opportunityWithSources" key={opportunity.barrier}>
          <div className="opportunityOverview"><span className="rank">#{opportunity.rank}</span><div className="opportunityMain"><h3>{readable(opportunity.barrier)}</h3><p>{opportunity.suggested_improvement}</p></div>
          <div className="opportunityMetrics"><span><b>{opportunity.mentions}</b> mentions</span><span><b>{opportunity.wishlist_review_share_percent}%</b> share</span></div><div className="score"><small>Score</small><strong>{opportunity.opportunity_score}</strong></div></div>
          <div className="opportunitySources"><small>REVIEW SOURCES</small><div>{(opportunity.samples ?? []).map((sample, index) => <blockquote key={`${sample.review_id}-${index}`}><b>{readable(sample.source)}</b><p>{sample.text}</p></blockquote>)}</div></div>
        </article>) : [1, 2, 3].map((item) => <div className="skeleton opportunitySkeleton" key={item} />)}
        {analysis?.wishlist_opportunities.length === 0 && <p className="empty">No wishlist-related evidence was found.</p>}</div></section>
    </main>
    <footer><b>Myntra Review Analyser</b><span>Review-derived signals should be validated with behavioural data.</span></footer>
  </>;
}
