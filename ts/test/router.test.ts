import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rankTools, route, type Tool } from '../src/router.js';

const ok = (name: string, keywords: string[], costUsd = 0.01, latencyMs = 100, out = name + ':ok'): Tool =>
  ({ name, keywords, costUsd, latencyMs, run: async (i: string) => out + ':' + i.length });
const bad = (name: string, keywords: string[], costUsd = 0.01, latencyMs = 100): Tool =>
  ({ name, keywords, costUsd, latencyMs, run: async () => { throw new Error(name + ' down'); } });

test('rankTools: match count desc, then cost asc, then name', () => {
  const t = [ok('search', ['find', 'doc']), ok('calc', ['sum']), ok('docs2', ['doc'], 0.001), ok('alpha', ['doc'], 0.001)];
  assert.deepEqual(rankTools('Find the DOC', t).map((x) => x.name), ['search', 'alpha', 'docs2']);
  assert.deepEqual(rankTools('nothing here', t), []);
});

test('route: first ranked tool succeeds', async () => {
  const r = await route('find doc', [ok('search', ['find'])], { maxCostUsd: 1, maxLatencyMs: 1000, fallback: 'FB' });
  assert.equal(r.tool, 'search'); assert.equal(r.output, 'search:ok:8'); assert.equal(r.degraded, false);
  assert.deepEqual(r.attempts, ['search']); assert.equal(r.costUsd, 0.01); assert.equal(r.latencyMs, 100);
});

test('route: falls through a failing tool, degraded', async () => {
  const r = await route('find doc', [bad('a', ['find', 'doc']), ok('b', ['doc'])], { maxCostUsd: 1, maxLatencyMs: 1000, fallback: 'FB' });
  assert.equal(r.tool, 'b'); assert.equal(r.degraded, true); assert.deepEqual(r.attempts, ['a', 'b']);
  assert.ok(Math.abs(r.costUsd - 0.02) < 1e-9); assert.equal(r.latencyMs, 200);
});

test('route: budget skips expensive tools', async () => {
  const r = await route('find', [ok('pricey', ['find'], 5), ok('cheap', ['find'], 0.5)], { maxCostUsd: 1, maxLatencyMs: 1000, fallback: 'FB' });
  assert.equal(r.tool, 'cheap'); assert.deepEqual(r.attempts, ['cheap']); assert.equal(r.degraded, false);
});

test('route: all fail -> fallback with last error', async () => {
  const r = await route('find', [bad('x', ['find'])], { maxCostUsd: 1, maxLatencyMs: 1000, fallback: 'FB' });
  assert.equal(r.tool, null); assert.equal(r.output, 'FB'); assert.equal(r.degraded, true); assert.equal(r.error, 'x down');
});

test('route: no matching tool -> fallback', async () => {
  const r = await route('hello', [ok('x', ['find'])], { maxCostUsd: 1, maxLatencyMs: 1000, fallback: 'FB' });
  assert.equal(r.tool, null); assert.equal(r.output, 'FB'); assert.equal(r.error, 'no matching tool'); assert.deepEqual(r.attempts, []);
});

test('route: latency budget', async () => {
  const r = await route('find', [ok('slow', ['find'], 0.01, 5000), ok('fast', ['find'], 0.02, 50)], { maxCostUsd: 1, maxLatencyMs: 1000, fallback: 'FB' });
  assert.equal(r.tool, 'fast');
});
