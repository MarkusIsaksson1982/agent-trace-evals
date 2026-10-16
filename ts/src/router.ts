export interface Tool { name: string; keywords: string[]; costUsd: number; latencyMs: number; run: (input: string) => Promise<string>; }
export interface RouteOptions { maxCostUsd: number; maxLatencyMs: number; fallback: string; }
export interface RouteResult { tool: string | null; output: string; degraded: boolean; attempts: string[]; costUsd: number; latencyMs: number; error?: string; }
const hits = (input: string, t: Tool): number => { const s = input.toLowerCase(); return t.keywords.filter((k) => s.includes(k.toLowerCase())).length; };
export function rankTools(input: string, tools: Tool[]): Tool[] {
  return tools.map((t) => ({ t, h: hits(input, t) })).filter((x) => x.h > 0)
    .sort((a, b) => b.h - a.h || a.t.costUsd - b.t.costUsd || a.t.name.localeCompare(b.t.name)).map((x) => x.t);
}
export async function route(input: string, tools: Tool[], opts: RouteOptions): Promise<RouteResult> {
  const attempts: string[] = []; let cost = 0, lat = 0; let error: string | undefined;
  for (const t of rankTools(input, tools)) {
    if (cost + t.costUsd > opts.maxCostUsd || lat + t.latencyMs > opts.maxLatencyMs) continue;
    attempts.push(t.name); cost += t.costUsd; lat += t.latencyMs;
    try { const output = await t.run(input); return { tool: t.name, output, degraded: attempts.length > 1, attempts, costUsd: cost, latencyMs: lat }; }
    catch (e) { error = e instanceof Error ? e.message : String(e); }
  }
  return { tool: null, output: opts.fallback, degraded: true, attempts, costUsd: cost, latencyMs: lat, error: error ?? 'no matching tool' };
}
