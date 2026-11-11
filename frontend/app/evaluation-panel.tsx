'use client';
// 평가 지표 패널: 세 검색 방식(Naive · Hybrid · Hybrid+Reranker)의 발제 필수 지표를 간단히 보여 준다.
// 데이터: GET /api/backend/evaluation/summary (evaluation/retrieval/api.py)
//   로컬 비교 실행이 있으면 그 결과, 없으면 저장소의 확정 결과(evaluation/retrieval/snapshots/*.json)
import { useEffect, useState } from 'react';
import { BarChart3, CircleCheck, Info, LoaderCircle, TriangleAlert, X } from 'lucide-react';
import styles from './evaluation-panel.module.css';

type Stat = { mean: number | null; n: number; na: number; unscorable: number; pending?: number };
type Config = Record<string, any> & { profile: string; mode: string; requests: number };
type Insight = { metric: string; tone: 'good' | 'warn' | 'info'; title: string; text: string };
type Summary = {
  run: { run_name: string; source?: string; question_count: number; model?: string; settings?: Record<string, number | null>;
         answer_scores_available?: boolean; scoring?: { evaluation_model?: string } };
  configs: Config[];
  insights: Record<string, Insight[]>;
  selection?: Record<string, { text: string }>;
};


// ── 표시 설정 ────────────────────────────────────────────────

const MODES = [
  { key: 'naive', label: 'Naive', color: '#8f98ad' },
  { key: 'hybrid', label: 'Hybrid · RRF', color: '#6fb3e0' },
  { key: 'hybrid_rerank', label: 'Hybrid + Reranker', color: '#a99bff' },
];

// 발제 필수 지표만 (추가 지표는 표시하지 않음)
const METRICS = [
  { key: 'hit_at_4', label: 'Hit@4', group: '검색', hint: '정답 문서를 하나라도 찾았나' },
  { key: 'recall_at_4', label: 'Recall@4', group: '검색', hint: '정답 문서를 얼마나 다 찾았나' },
  { key: 'faithfulness', label: '근거 충실도', group: '답변', hint: 'RAGAS Faithfulness · 답변이 받은 문맥에 근거하나' },
  { key: 'judge_facts', label: '사실', group: '답변', hint: 'LLM Judge · 핵심 사실이 맞나' },
  { key: 'judge_conditions', label: '조건', group: '답변', hint: 'LLM Judge · 날짜·예외 등 조건을 지켰나' },
  { key: 'judge_evidence', label: '근거', group: '답변', hint: 'LLM Judge · 인용이 주장을 뒷받침하나' },
  { key: 'judge_decision', label: '결정', group: '답변', hint: 'LLM Judge · 답변/보류/되묻기 판단이 맞나' },
];
const PROFILE_LABEL: Record<string, string> = { bge: 'BGE-M3', qwen: 'Qwen3 Embedding' };

const stat = (c: Config | undefined, key: string): Stat | null => (c && c[key] && typeof c[key] === 'object' && 'mean' in c[key] ? c[key] : null);
const num = (v: number | null | undefined) => (v == null ? '–' : v.toFixed(2));
const sec = (ms: number | null | undefined) => (ms == null ? '–' : `${(ms / 1000).toFixed(1)}초`);


// ── 지표별 막대 (방식 3개를 한 줄에) ──────────────────────────

function MetricBars({ configs }: { configs: Config[] }) {
  const byMode = Object.fromEntries(configs.map(c => [c.mode, c]));
  let lastGroup = '';
  return <div className={styles.bars}>
    {METRICS.filter(m => configs.some(c => stat(c, m.key)?.mean != null)).map(m => {
      const header = m.group !== lastGroup ? <div className={styles.group}>{m.group}</div> : null;
      lastGroup = m.group;
      return <div key={m.key}>{header}<div className={styles.barRow}>
        <div className={styles.barLabel}>{m.label}<small>{m.hint}</small></div>
        <div className={styles.barStack}>
          {MODES.map(mode => {
            const s = stat(byMode[mode.key], m.key);
            return <div className={styles.barLine} key={mode.key} title={`${mode.label} ${num(s?.mean)} (n=${s?.n ?? 0})`}>
              <span className={styles.bar} style={{ width: `${Math.max(0, Math.min(1, s?.mean ?? 0)) * 100}%`, background: mode.color }} />
              <em>{num(s?.mean)}</em>
            </div>;
          })}
        </div>
      </div></div>;
    })}
  </div>;
}


// ── 운영 지표 표: 평균 응답 시간 · 오류율 · 토큰 ─────────────────

function Operations({ configs }: { configs: Config[] }) {
  const maxLatency = Math.max(...configs.map(c => c.latency_mean_ms || 0), 1);
  return <div className={styles.ops}>
    {MODES.map(m => {
      const c = configs.find(x => x.mode === m.key);
      if (!c) return null;
      return <div className={styles.opRow} key={m.key}>
        <span className={styles.opName}><i style={{ background: m.color }} />{m.label}</span>
        <div className={styles.latency}><span style={{ width: `${(c.latency_mean_ms || 0) / maxLatency * 100}%`, background: m.color }} /></div>
        <span className={styles.opNums}>
          평균 {sec(c.latency_mean_ms)} · 오류 {c.errors}/{c.requests} ({(c.error_rate_pct ?? 0).toFixed(1)}%)<br />
          토큰 입력 {c.input_tokens_sum?.toLocaleString() ?? '–'} · 출력 {c.output_tokens_sum?.toLocaleString() ?? '–'}
        </span>
      </div>;
    })}
  </div>;
}

const ToneIcon = ({ tone }: { tone: Insight['tone'] }) => tone === 'good' ? <CircleCheck size={15} /> : tone === 'warn' ? <TriangleAlert size={15} /> : <Info size={15} />;


// ── 패널 ─────────────────────────────────────────────────────

export default function EvaluationPanel({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<Summary | null>(null), [error, setError] = useState(''), [loading, setLoading] = useState(true);
  const [profile, setProfile] = useState('bge');

  // 1. 요약 불러오기 (최근 실행 또는 스냅샷)
  useEffect(() => {
    fetch('/api/backend/evaluation/summary')
      .then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d?.detail?.message || '평가 결과를 불러오지 못했습니다.'); return d; })
      .then((d: Summary) => { setData(d); const profiles = [...new Set(d.configs.map(c => c.profile))]; if (!profiles.includes('bge')) setProfile(profiles[0]); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // 2. 선택한 임베딩의 3방식만 골라 표시
  const configs = data?.configs.filter(c => c.profile === profile) || [];
  const profiles = data ? [...new Set(data.configs.map(c => c.profile))] : [];
  const insights = data?.insights?.[profile] || [];
  const pick = data?.selection?.[profile]?.text;

  return <div className={styles.overlay} onClick={onClose}>
    <section className={styles.panel} onClick={e => e.stopPropagation()} aria-label="평가 지표">
      <header className={styles.head}>
        <div><span className={styles.eyebrow}>RETRIEVAL EVALUATION</span><h2><BarChart3 size={19} />검색 방식 성능 비교</h2></div>
        <button className="icon-button" aria-label="평가 지표 닫기" onClick={onClose}><X size={20} /></button>
      </header>

      <div className={styles.controls}>
        <div className={styles.seg}>{profiles.map(p => <button key={p} className={p === profile ? styles.on : ''} onClick={() => setProfile(p)}>{PROFILE_LABEL[p] || p}</button>)}</div>
        {data?.run.source === 'snapshot' && <span className={styles.badge}>저장된 확정 결과</span>}
      </div>

      {loading ? <p className={styles.empty}><LoaderCircle size={16} className="spin" /> 불러오는 중…</p>
        : error ? <p className={styles.empty}>{error}</p>
        : data && <>
          <p className={styles.meta}>
            {data.run.run_name} · 문항 {data.run.question_count}개 · 최종 청크 k={data.run.settings?.top_k ?? '–'} · 생성 {data.run.model ?? '–'}
            {data.run.scoring?.evaluation_model && ` · 채점 ${data.run.scoring.evaluation_model}`}
          </p>
          <div className={styles.legend}>{MODES.map(m => <span key={m.key}><i style={{ background: m.color }} />{m.label}</span>)}</div>

          {pick && <ul className={styles.insights}><li className={styles.good}><CircleCheck size={15} /><div><b>추천 구성</b><span>{pick}</span></div></li></ul>}

          <h3>지표 비교 (발제 필수)</h3>
          {!data.run.answer_scores_available && <p className={styles.note}>답변 채점 결과가 없어 검색 지표만 표시합니다.</p>}
          <MetricBars configs={configs} />
          <p className={styles.note} style={{ marginTop: 8 }}>답변 4항목은 항목별 충족률이며 합산하지 않습니다. 평가 불가 건은 0점으로 채우지 않고 평균에서 뺍니다.</p>

          <h3>속도 · 오류 · 토큰</h3>
          <p className={styles.note}>정상 요청의 평균 응답 시간. 로컬 추론이라 비용은 계산하지 않습니다.</p>
          <Operations configs={configs} />

          <h3>지표 해석</h3>
          <ul className={styles.insights}>
            {insights.length ? insights.map((i, n) => <li key={n} className={styles[i.tone]}><ToneIcon tone={i.tone} /><div><b>{i.title}</b><span>{i.text}</span></div></li>)
              : <li className={styles.info}><Info size={15} /><div><span>해석할 만한 차이가 없습니다.</span></div></li>}
          </ul>
        </>}
    </section>
  </div>;
}
