// free-model-aggregator-01 entry for @ice (Node 22, standard library only).
//
// A free-model aggregator: one server, two client wires, ten providers. The server reads the
// trial's brief (MAKEFUN_PROMPT_FILE) — ten providers on three wires (OpenAI, Anthropic, Gemini),
// a sealed quality table rating their free models, and a twenty-request workload — then serves
// each request by routing it to the best free model that can answer it, failing over when a
// provider refuses, and answering in exactly the wire the client used.
//
// The whole program is a function of the brief and the requests it is given. Every choice —
// which provider to try, how to translate a request, when to give up — is derived from the
// brief's quality table and the providers' own answers. There is no wall clock and no
// unseeded randomness anywhere in a decision, so a replay of a trial sees the same brief, the
// same probes and the same answers, and produces the same result.
//
// Two files sit beside this one: lib/canon.ts (the canonical form every wire translates
// through) and lib/upstream.ts (one call at a time per provider, with retries and failover).
// lib/providers.ts reads the brief and probes the catalog; lib/wire.ts translates each wire.

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { fromAnthropic, fromGemini, toAnthropic, toGemini, toOpenAI } from './lib/wire.ts';
import { callUpstream, type CallResult, type Refusal } from './lib/upstream.ts';
import { canonicalToOpenAI, openAIToCanonical, type CanonicalRequest } from './lib/canon.ts';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PORT = process.env.PORT ? Number(process.env.PORT) : 0;

interface Env {
  url: string;
  promptFile: string;
}

const env: Env = { url: ENV_URL, promptFile: PROMPT_FILE };

function logLine(line: string): void {
  process.stdout.write(`${line}\n`);
}

function requestIdOf(req: { headers: { [key: string]: string | string[] | undefined } }): string {
  const header = req.headers['x-request-id'];
  if (typeof header === 'string' && header !== '') return header;
  return 'unknown';
}

/** A served model's answer: the model that answered, in the client's wire. */
interface Served {
  model: string;
  body: Json;
}

function errorJson(message: string, status: number): { status: number; body: Json } {
  return { status, body: { error: { message, type: 'invalid_request_error', code: status } } };
}

/**
 * The candidates for a request, in the order they are tried: every free model whose
 * capabilities meet the request's requirements, ranked by its quality rating, best first.
 * Ties break on provider id, then model id — a fixed order, so the choice is deterministic.
 */
export function candidatesFor(
  probes: Probe[],
  ratings: Parameters<typeof ratingOfModel>[0],
  requires: Requirement,
): { provider: Probe['provider']; model: ProbedModel; rating: number }[] {
  const out: { provider: Probe['provider']; model: ProbedModel; rating: number }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      out.push({ provider: probe.provider, model, rating: row ? row.rating_milli : 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/**
 * Serve one request: try the candidates in order, one upstream call at a time, until one
 * answers. The best candidate is tried twice (one retry); every other candidate is tried
 * once. A provider that refuses three times in a row is exhausted and skipped until it
 * answers again (a success resets its count). A candidate's upstream call is made in its
 * provider's wire; the answer is translated back through the canonical form into the
 * client's wire, and names the model that answered.
 */
async function serveRequest(
  probes: Probe[],
  ratings: Parameters<typeof ratingOfModel>[0],
  canonical: CanonicalRequest,
  clientWire: 'openai' | 'anthropic',
  upstream: ReturnType<typeof callUpstream>,
  exhausted: Map<string, number>,
): Promise<Served | { error: { status: number; body: Json } }> {
  const requires: Requirement = {
    tools: canonical.tools.length > 0,
    vision: canonical.images.length > 0,
    json: canonical.jsonMode,
    minContext: canonical.minContext,
  };
  const candidates = candidatesFor(probes, ratings, requires);
  if (candidates.length === 0) {
    return { error: errorJson('no free model can serve this request', 400) };
  }
  let lastRefusal: Refusal | null = null;
  for (let index = 0; index < candidates.length; index += 1) {
    const candidate = candidates[index];
    const providerId = candidate.provider.id;
    if ((exhausted.get(providerId) ?? 0) >= 3) continue;
    const attempts = index === 0 ? 2 : 1;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const base = joinBase(env.url, candidate.provider.base);
      const headers = authHeaders(candidate.provider.auth, candidate.provider.key, candidate.provider.wire);
      const upstreamBody = buildUpstreamBody(candidate.provider.wire, canonical, candidate.model.id);
      const result = await upstream(base, candidate.provider.wire, headers, upstreamBody, canonical.stream);
      if (result.ok) {
        exhausted.set(providerId, 0);
        const body = translateAnswer(candidate.provider.wire, clientWire, result, canonical);
        return { model: candidate.model.id, body };
      }
      lastRefusal = result.refusal;
      exhausted.set(providerId, (exhausted.get(providerId) ?? 0) + 1);
      if ((exhausted.get(providerId) ?? 0) >= 3) break;
    }
  }
  const status = lastRefusal ? lastRefusal.status : 502;
  return { error: errorJson(lastRefusal ? lastRefusal.message : 'all providers refused', status) };
}

function buildUpstreamBody(wire: string, canonical: CanonicalRequest, modelId: string): Json {
  if (wire === 'openai') return canonicalToOpenAI(canonical, modelId);
  if (wire === 'anthropic') return toAnthropic(canonical, modelId);
  return toGemini(canonical, modelId);
}

function translateAnswer(
  wire: string,
  clientWire: 'openai' | 'anthropic',
  result: Extract<CallResult, { ok: true }>,
  canonical: CanonicalRequest,
): Json {
  let canonicalAnswer: ReturnType<typeof fromAnthropic>;
  if (wire === 'openai') canonicalAnswer = openAIToCanonical(result.body, canonical);
  else if (wire === 'anthropic') canonicalAnswer = fromAnthropic(result.body, canonical);
  else canonicalAnswer = fromGemini(result.body, canonical);
  if (clientWire === 'openai') return canonicalToOpenAI(canonicalAnswer, canonicalAnswer.model);
  return toAnthropic(canonicalAnswer, canonicalAnswer.model);
}

/** Read the request body as JSON; a malformed body is a 400. */
async function readJsonBody(req: { [Symbol.asyncIterator](): AsyncIterator<Buffer> }): Promise<{ ok: true; value: Json } | { ok: false; error: { status: number; body: Json } }> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString('utf8');
  try {
    const value = JSON.parse(text) as unknown;
    if (!isRecord(value)) return { ok: false, error: errorJson('request body is not a JSON object', 400) };
    return { ok: true, value };
  } catch {
    return { ok: false, error: errorJson('request body is not valid JSON', 400) };
  }
}

/** The wire the auth header names: an x-api-key names Anthropic, anything else OpenAI. */
function clientWireOf(headers: { [key: string]: string | string[] | undefined }): 'openai' | 'anthropic' {
  const key = headers['x-api-key'];
  return typeof key === 'string' && key !== '' ? 'anthropic' : 'openai';
}

/** The catalog the trial registers: every answered provider and its free models. */
function catalogOf(probes: Probe[], ratings: Parameters<typeof ratingOfModel>[0]): CatalogReport {
  return buildCatalog(env.url, probes, ratings);
}

/** Write a JSON answer, or an SSE stream when the client asked for one. */
function writeAnswer(
  res: { writeHead(status: number, headers: { [key: string]: string }): void; write(chunk: string): void; end(chunk?: string): void },
  wire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): void {
  if (!stream) {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(body));
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (wire === 'openai') {
    const chunk = {
      id: typeof body.id === 'string' ? body.id : 'chatcmpl-1',
      object: 'chat.completion.chunk',
      created: 0,
      model: typeof body.model === 'string' ? body.model : '',
      choices: [{ index: 0, delta: { content: textOf(body) }, finish_reason: 'stop' }],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    res.write('data: [DONE]\n\n');
  } else {
    const text = textOf(body);
    res.write('event: message_start\n');
    res.write(`data: ${JSON.stringify({ type: 'message_start', message: { id: 'msg_1', type: 'message', role: 'assistant', content: [], model: typeof body.model === 'string' ? body.model : '', stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } })}\n\n`);
    res.write('event: content_block_start\n');
    res.write(`data: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n\n`);
    res.write('event: content_block_delta\n');
    res.write(`data: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } })}\n\n`);
    res.write('event: content_block_stop\n');
    res.write(`data: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`);
    res.write('event: message_delta\n');
    res.write(`data: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 0 } })}\n\n`);
    res.write('event: message_stop\n');
    res.write(`data: ${JSON.stringify({ type: 'message_stop' })}\n\n`);
  }
  res.end();
}

function textOf(body: Json): string {
  if (isRecord(body) && Array.isArray(body.choices)) {
    const choice = body.choices[0];
    if (isRecord(choice) && isRecord(choice.message) && typeof choice.message.content === 'string') return choice.message.content;
  }
  if (isRecord(body) && Array.isArray(body.content)) {
    const block = body.content[0];
    if (isRecord(block) && typeof block.text === 'string') return block.text;
  }
  return '';
}

async function main(): Promise<void> {
  if (env.url === '') {
    process.stderr.write('[entry] MAKEFUN_ENV_URL is not set\n');
    process.exit(1);
  }
  const brief = readBrief(env.promptFile);
  if (brief.error) {
    process.stderr.write(`[entry] ${brief.error}\n`);
    process.exit(1);
  }
  const probes: Probe[] = [];
  for (const provider of brief.directory.providers) {
    probes.push(await probeProvider(env.url, provider, 10_000));
  }
  const catalog = catalogOf(probes, brief.ratings);
  const exhausted = new Map<string, number>();
  const upstream = callUpstream;

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    const wire = clientWireOf(req.headers);
    try {
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        const body = wire === 'anthropic'
          ? { object: 'list', data: catalog.providers.filter((p) => p.wire === 'anthropic').flatMap((p) => p.models.map((m) => ({ id: m.id, object: 'model', display_name: m.display_name }))) }
          : { object: 'list', data: catalog.providers.filter((p) => p.wire !== 'anthropic').flatMap((p) => p.models.map((m) => ({ id: m.id, object: 'model', display_name: m.display_name }))) };
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(body));
        return;
      }
      if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
        const parsed = await readJsonBody(req as unknown as { [Symbol.asyncIterator](): AsyncIterator<Buffer> });
        if (!parsed.ok) {
          res.writeHead(parsed.error.status, { 'content-type': 'application/json' });
          res.end(JSON.stringify(parsed.error.body));
          return;
        }
        const canonical = wire === 'openai' ? openAIToCanonical(parsed.value, null) : fromAnthropic(parsed.value, null);
        const served = await serveRequest(probes, brief.ratings, canonical, wire, upstream, exhausted);
        if ('error' in served) {
          res.writeHead(served.error.status, { 'content-type': 'application/json' });
          res.end(JSON.stringify(served.error.body));
          return;
        }
        writeAnswer(res, wire, served.body, canonical.stream);
        return;
      }
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'not found', type: 'invalid_request_error', code: 404 } }));
    } catch (err) {
      res.writeHead(500, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: (err as Error).message, type: 'server_error', code: 500 } }));
    }
  });

  await new Promise<void>((resolve) => server.listen(PORT, '127.0.0.1', () => resolve()));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : PORT;
  const registerRes = await fetch(`${env.url}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ port, catalog }),
  });
  if (!registerRes.ok) {
    process.stderr.write(`[entry] register answered HTTP ${registerRes.status}\n`);
    process.exit(1);
  }
  logLine(`[entry] listening on port ${port}; catalog registered`);
  for (;;) {
    const workRes = await fetch(`${env.url}/work`);
    if (!workRes.ok) {
      process.stderr.write(`[entry] work answered HTTP ${workRes.status}\n`);
      process.exit(1);
    }
    const work = (await workRes.json()) as { done?: unknown };
    if (work.done === true) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  server.close();
  logLine('[entry] done');
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`[entry] ${(err as Error).message}\n`);
  process.exit(1);
});
