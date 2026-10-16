import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BM25, Dense, Hybrid, rrf, recallAtK, mrr, ndcgAtK, evaluateRetriever, checkCitations, isRefusal, REFUSAL_SV, tokens, cosine, type Doc, type Query } from '../src/rag.js';

const data = (n: string): unknown => JSON.parse(readFileSync(new URL(`../../../rag_data/${n}`, import.meta.url), 'utf8'));
const docs = data('corpus.json') as Doc[];
const queries = data('queries.json') as Query[];
const emb = data('embeddings.json') as { docs: Record<string, number[]>; queries: Record<string, number[]> };
const qv = new Map(queries.map((q) => [q.query, emb.queries[q.qid]]));
const dense = new Dense(docs, emb.docs, (q) => { const v = qv.get(q); if (!v) throw new Error('no precomputed vector'); return v; });

test('tokens drop stopwords, keep Swedish letters', () => assert.deepEqual(tokens('How do I reset my lösenord?'), ['reset', 'lösenord']));
test('metrics', () => {
  assert.equal(recallAtK(['a', 'b'], ['b'], 1), 0); assert.equal(mrr(['a', 'b'], ['b']), 0.5);
  assert.equal(ndcgAtK(['g'], ['g'], 5), 1); assert.ok(ndcgAtK(['x', 'g'], ['g'], 5) < 1);
  assert.equal(recallAtK(['a', 'a'], ['a'], 5), 1); // duplicate ids count once (Python set parity)
});
test('cosine', () => { assert.ok(Math.abs(cosine([1, 0], [2, 0]) - 1) < 1e-9); assert.ok(Math.abs(cosine([1, 0], [0, 3])) < 1e-9); });
test('bm25 exact match', () => assert.equal(new BM25(docs).search('Vasterby-Gast guest network')[0][0], 'wifi-01'));
test('rrf fuses', () => assert.equal(rrf([[['a', 9], ['b', 1]], [['b', 5], ['c', 1]]])[0][0], 'b'));
test('retriever metrics match Python numbers', () => {
  // values from `python -m agent_trace_evals.rag.bench` (same corpus, same embeddings)
  assert.deepEqual(evaluateRetriever(new BM25(docs), queries), { 'recall@1': 0.615, 'recall@3': 0.769, 'recall@5': 0.846, mrr: 0.699, 'ndcg@5': 0.736 });
  assert.deepEqual(evaluateRetriever(dense, queries), { 'recall@1': 0.885, 'recall@3': 1, 'recall@5': 1, mrr: 0.936, 'ndcg@5': 0.952 });
  assert.deepEqual(evaluateRetriever(new Hybrid(new BM25(docs), dense), queries), { 'recall@1': 0.769, 'recall@3': 0.923, 'recall@5': 0.962, mrr: 0.862, 'ndcg@5': 0.883 });
});
test('citation check', () => {
  assert.equal(checkCitations('Use cable [vpn-02].', ['vpn-02']).valid, true);
  assert.deepEqual(checkCitations('Use cable [sec-05].', ['vpn-02']).unknown, ['sec-05']);
  assert.equal(checkCitations('No citation here.', ['vpn-02']).valid, false);
  assert.equal(checkCitations("I don't know based on the available documents.", []).valid, true);
  assert.equal(checkCitations("I don't know the menu, but use GlobalConnect.", []).valid, false); // partial refusal is not a refusal
});
test('isRefusal', () => {
  assert.equal(isRefusal(REFUSAL_SV), true);
  assert.equal(isRefusal('"I don\'t know based on the available documents"'), true);
  assert.equal(checkCitations(REFUSAL_SV, []).valid, true);
});
