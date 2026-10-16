// Strict-TS port of agent_trace_evals/rag: BM25, cosine over precomputed embeddings, RRF, metrics, citation check.
// Not ported (Python-only): rerank, gate, pack, faithfulness.
export interface Doc { id: string; title: string; text: string; }
export type Hit = [id: string, score: number];
export interface Retriever { search(q: string, k?: number): Hit[]; }
const TOK = /[0-9a-zåäöéü]+/g;
const STOP = new Set(('the a an and or to of in on is i my me how do can what for with from at be by are you your this that ' +
  'och att det jag är på en ett av för med som har inte till om vad hur min mitt').split(' '));
export const tokens = (s: string): string[] => (s.toLowerCase().match(TOK) ?? []).filter((t) => !STOP.has(t));
export const docText = (d: Doc): string => `${d.title}. ${d.text}`;
const byScore = (a: Hit, b: Hit): number => b[1] - a[1] || a[0].localeCompare(b[0]);

export class BM25 implements Retriever {
  private tf: Map<string, number>[]; private len: number[]; private avg: number; private idf = new Map<string, number>();
  constructor(private docs: Doc[], private k1 = 1.5, private b = 0.75) {
    this.tf = docs.map((d) => { const m = new Map<string, number>(); for (const t of tokens(docText(d))) m.set(t, (m.get(t) ?? 0) + 1); return m; });
    this.len = this.tf.map((m) => [...m.values()].reduce((x, y) => x + y, 0));
    this.avg = this.len.reduce((x, y) => x + y, 0) / docs.length;
    const df = new Map<string, number>(); for (const m of this.tf) for (const t of m.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    for (const [t, v] of df) this.idf.set(t, Math.log(1 + (docs.length - v + 0.5) / (v + 0.5)));
  }
  search(q: string, k = 10): Hit[] {
    const qt = tokens(q);
    return this.docs.map((d, i): Hit => {
      let s = 0;
      for (const t of qt) { const f = this.tf[i].get(t) ?? 0; if (f) s += this.idf.get(t)! * f * (this.k1 + 1) / (f + this.k1 * (1 - this.b + this.b * this.len[i] / this.avg)); }
      return [d.id, s];
    }).filter((h) => h[1] > 0).sort(byScore).slice(0, k);
  }
}

const norm = (v: number[]): number[] => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
export const cosine = (a: number[], b: number[]): number => { const x = norm(a), y = norm(b); return x.reduce((s, v, i) => s + v * y[i], 0); };

export class Dense implements Retriever {
  private vecs: [string, number[]][];
  constructor(docs: Doc[], docVecs: Record<string, number[]>, private embedQuery: (q: string) => number[]) {
    this.vecs = docs.map((d): [string, number[]] => { const v = docVecs[d.id]; if (!v) throw new Error(`no vector for ${d.id}`); return [d.id, norm(v)]; });
  }
  search(q: string, k = 10): Hit[] { const v = norm(this.embedQuery(q)); return this.vecs.map(([id, d]): Hit => [id, d.reduce((s, x, i) => s + x * v[i], 0)]).sort(byScore).slice(0, k); }
}

export function rrf(rankings: Hit[][], k = 60, top = 10): Hit[] {
  const acc = new Map<string, number>();
  for (const r of rankings) r.forEach(([id], rank) => acc.set(id, (acc.get(id) ?? 0) + 1 / (k + rank + 1)));
  return [...acc.entries()].sort(byScore).slice(0, top);
}
export class Hybrid implements Retriever {
  constructor(private sparse: Retriever, private dense: Retriever, private depth = 20) {}
  search(q: string, k = 10): Hit[] { return rrf([this.sparse.search(q, this.depth), this.dense.search(q, this.depth)], 60, k); }
}

export const recallAtK = (ret: string[], gold: string[], k: number): number => {
  const g = new Set(gold);
  return g.size ? [...new Set(ret.slice(0, k))].filter((i) => g.has(i)).length / gold.length : 0;
};
export const mrr = (ret: string[], gold: string[]): number => { const i = ret.findIndex((x) => gold.includes(x)); return i < 0 ? 0 : 1 / (i + 1); };
export function ndcgAtK(ret: string[], gold: string[], k: number): number {
  const dcg = ret.slice(0, k).reduce((s, id, r) => s + (gold.includes(id) ? 1 / Math.log2(r + 2) : 0), 0);
  let idcg = 0; for (let r = 0; r < Math.min(gold.length, k); r++) idcg += 1 / Math.log2(r + 2);
  return idcg ? dcg / idcg : 0;
}
export interface Query { qid: string; query: string; gold: string[]; }
export function evaluateRetriever(r: Retriever, queries: Query[]): Record<string, number> {
  const qs = queries.filter((q) => q.gold.length); const m = { 'recall@1': 0, 'recall@3': 0, 'recall@5': 0, mrr: 0, 'ndcg@5': 0 };
  for (const q of qs) {
    const ids = r.search(q.query, 10).map((h) => h[0]);
    m['recall@1'] += recallAtK(ids, q.gold, 1); m['recall@3'] += recallAtK(ids, q.gold, 3); m['recall@5'] += recallAtK(ids, q.gold, 5);
    m.mrr += mrr(ids, q.gold); m['ndcg@5'] += ndcgAtK(ids, q.gold, 5);
  }
  return Object.fromEntries(Object.entries(m).map(([k, v]) => [k, Math.round((v / qs.length) * 1000) / 1000] /* Python uses round-half-even; differs only on an exact .xxx5, none occur here */));
}

const CITE = /\[([a-z]+-\d+)\]/g;
export interface CitationReport { citations: string[]; valid: boolean; unknown: string[]; refused: boolean; }
export const REFUSAL = "I don't know based on the available documents.";
export const REFUSAL_SV = 'Jag vet inte utifrån de tillgängliga dokumenten.';
const canon = (s: string): string => s.replace(/^[\s'"“”‘’]+/, '').replace(/[\s.!?'"“”‘’]+$/, '').toLowerCase();
/** Same rule as Python is_refusal: the whole answer is the refusal sentence, not a substring match. */
export const isRefusal = (a: string): boolean => a.trim() !== '' && [REFUSAL, REFUSAL_SV].map(canon).includes(canon(a));
export function checkCitations(answer: string, retrievedIds: string[]): CitationReport {
  const citations = [...answer.matchAll(CITE)].map((m) => m[1]); const unknown = citations.filter((c) => !retrievedIds.includes(c));
  const refused = isRefusal(answer);
  return { citations, unknown, refused, valid: unknown.length === 0 && (citations.length > 0 || refused) };
}
