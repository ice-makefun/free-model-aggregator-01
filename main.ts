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
// Routing, in one paragraph. A client request (OpenAI Chat Completions or Anthropic Messages,
// JSON or server-sent events) is read into one canonical form. From that form the entry works
// out what the request needs — tools, vision, JSON mode, and an approximate context length —
// then ranks every free model that meets those needs by the quality table's rating, best first.
// It calls the best candidate on its own wire, translating the canonical request into that
// wire and the provider's answer back into canonical form. A refusal (402, 429, or any 5xx)
// retires that provider for the rest of the trial after three consecutive refusals, and the
// strictly-best candidate is retried once before the entry fails over down the ranked list.
// An OpenAI JSON-mode request is never routed to an Anthropic-wire provider, because that wire
// cannot answer it in JSON mode. Only free models are ever routed; a paid model's answer
// earns nothing, so a paid model is never tried.
//
// The trial lifecycle is the kit's own: probe every provider's catalog route, build the
// catalog, listen on an ephemeral port, register that port and the catalog with the env,
// then poll the env's work endpoint until it says the workload is done, and exit zero.
// The catalog the entry registers is the one the probes and the quality table make, so a
// replay of the same trial rebuilds the same catalog byte for byte.
//
// Files beside this one: lib/canon.ts (the canonical form every wire translates through),
// lib/upstream.ts (one call at a time per provider, with retries and failover), lib/providers.ts
// (reading the brief, probing the catalog, the quality table), lib/wire.ts (the client wires'
// shapes: model lists, errors, and the two directions of translation).

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
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
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  canonFromAnthropic,
  canonFromOpenAI,
  type CanonReq,
  type CanonResp,
} from './lib/canon.ts';
import {
  callUpstream,
  canonToAnthropicResp,
  canonToAnthropicStream,
  canonToOpenAIResp,
  canonToOpenAIStream,
  respToCanon,
  streamToCanon,
  type CanonDelta,
} from './lib/upstream.ts';

// The protocol version the env expects at registration. A mismatch is a 400.
const PROTOCOL_VERSION = 1;
// How often the entry asks the env whether the workload is done, in milliseconds.
const POLL_INTERVAL_MS = 100;
// How long a catalog probe may take before it counts as a failure, in milliseconds.
const PROBE_TIMEOUT_MS = 5000;
// The largest request body the entry will read, in bytes. The env caps at the same size.
const MAX_BODY_BYTES = 32 * 1024 * 1024;
// Consecutive refusals after which a provider is exhausted for the rest of the trial.
const EXHAUST_AFTER = 3;
// A fixed id for streamed OpenAI chunks. The upstream id is not known until a stream ends,
// and a replay must see the same bytes, so the id is a constant, never the clock.
const STREAM_ID = 'chatcmpl-free-model-aggregator';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// The brief: the provider directory, the sealed quality table, and the workload manifest.
// Read once at startup; every later choice is a function of this document and the requests.
const brief = readBrief(PROMPT_FILE);

// Per-provider consecutive refusals, by provider id. A success resets the count to zero.
const rejections = new Map<string, number>();
// Providers that have refused three times in a row. They are skipped for the rest of the trial.
const exhausted = new Set<string>();
// The catalog probes, filled in by load() before the server starts serving.
let probes: Probe[] = [];
// The catalog the probes and the quality table make, registered with the env at startup.
let catalog: CatalogReport | null = null;
// The HTTP server, created by listen() and closed by start() once the workload is done.
let server: ReturnType<typeof createServer> | null = null;

// One log line to stderr. Stdout stays clean for the trial harness.
function log(message: string): void {
  console.error(`[aggregator] ${message}`);
}

// A deterministic pause: the poll loop's wait between work checks.
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// The text of every message in a client body, as one string. Used to estimate how much
// context a request needs: a long request needs a long-context model, a short one does not.
function allTextOf(body: Json): string {
  const parts: string[] = [];
  const system = body.system;
  if (typeof system === 'string') parts.push(system);
  if (Array.isArray(system)) {
    for (const block of system) {
      if (isRecord(block) && typeof block.text === 'string') parts.push(block.text);
    }
  }
  const messages = body.messages;
  if (Array.isArray(messages)) {
    for (const message of messages) {
      if (!isRecord(message)) continue;
      const content = message.content;
      if (typeof content === 'string') {
        parts.push(content);
        continue;
      }
      if (!Array.isArray(content)) continue;
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (typeof block.text === 'string') parts.push(block.text);
        if (typeof block.input === 'string') parts.push(block.input);
        if (isRecord(block.input)) parts.push(JSON.stringify(block.input));
      }
    }
  }
  return parts.join('\n');
}

// Whether any message in a client body carries an image, on either client wire.
// OpenAI spells it image_url; Anthropic spells it image.
function hasImage(body: Json): boolean {
  const messages = body.messages;
  if (!Array.isArray(messages)) return false;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image_url') return true;
      if (block.type === 'image') return true;
      if (isRecord(block.source) && block.source.type === 'base64') return true;
    }
  }
  return false;
}

// What one client request needs, read off its own body. Tools and vision are read from the
// request itself; JSON mode is an OpenAI response_format; the context a request needs is
// estimated from its text at roughly four characters per token, rounded up. A request that
// needs more context than a model offers is never routed to that model.
function requirementsFromBody(body: Json, clientWire: 'openai' | 'anthropic'): Requirement {
  const requires: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (hasImage(body)) requires.vision = true;
  if (clientWire === 'openai' && isRecord(body.response_format) && body.response_format.type === 'json_object') {
    requires.json = true;
  }
  const text = allTextOf(body);
  if (text.length > 0) requires.min_context = Math.ceil(text.length / 4);
  return requires;
}

// The models this entry serves, as GET /v1/models lists them: every free model the catalog
// holds, deduplicated by id and sorted by id so the list is the same on every replay.
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  if (catalog === null) return out;
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return out;
}

// One ranked candidate: a provider and one of its models, with the quality table's rating.
interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
}

// Every free model that can answer a request, best first. The ranking is the quality table's
// rating, highest first; ties break on provider id, then model id, so the order is the same
// on every replay. An OpenAI JSON-mode request never goes to an Anthropic-wire provider.
function candidatesFor(
  requires: Requirement,
  jsonMode: boolean,
  clientWire: 'openai' | 'anthropic',
): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.directory.providers) {
    const probe = probes.find((p) => p.provider.id === provider.id);
    if (probe === undefined || !probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      if (jsonMode && clientWire === 'openai' && provider.wire === 'anthropic') continue;
      out.push({ provider, model, rating: row === null ? 0 : row.rating_milli });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

// Record one refusal from a provider. Three in a row exhausts it for the rest of the trial.
function noteRejection(providerId: string): void {
  const count = (rejections.get(providerId) ?? 0) + 1;
  rejections.set(providerId, count);
  if (count >= EXHAUST_AFTER) exhausted.add(providerId);
}

// Record one success from a provider. The consecutive-refusal count starts over.
function noteSuccess(providerId: string): void {
  rejections.set(providerId, 0);
}

// Read a request body as text, refusing anything over the cap. The env caps at the same size,
// so a body this large is a client error, not a trial to serve.
function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let failed = false;
    req.on('data', (chunk: Buffer) => {
      if (failed) return;
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        failed = true;
        reject(new Error('request body is too large'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (failed) return;
      resolve(Buffer.concat(chunks).toString('utf8'));
    });
    req.on('error', (err) => {
      if (failed) return;
      failed = true;
      reject(err);
    });
  });
}

// Send a JSON document with a status code. Every reply is JSON, never HTML.
function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const text = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

// Send an error in the client's own wire shape: OpenAI's { error: { message, type, code } }
// or Anthropic's { type: 'error', error: { type, message } }.
function sendError(res: ServerResponse, status: number, message: string, clientWire: 'openai' | 'anthropic'): void {
  if (clientWire === 'anthropic') {
    sendJson(res, status, anthropicError(status === 404 ? 'not_found_error' : 'api_error', message));
  } else {
    sendJson(res, status, openAIError(message, status === 404 ? 'not_found' : 'server_error', null));
  }
}

// The answer to one request: either a complete canonical response, or a stream of canonical
// deltas with the model that produced them. Null means no free model could serve the request.
type ServedAnswer =
  | { kind: 'json'; resp: CanonResp; model: string }
  | { kind: 'stream'; deltas: AsyncGenerator<CanonDelta>; model: string };

// Route one canonical request to the best free model that can answer it, failing over when
// a provider refuses. The strictly-best candidate is retried once before the entry moves
// down the ranked list; a provider that refuses three times in a row is exhausted for the
// rest of the trial and skipped. A refusal is a 402, a 429, any 5xx, any other non-2xx, a
// network failure, or an answer that cannot be parsed. Only a 2xx with a usable answer
// counts as a success, and only then is the provider's refusal count reset.
async function serveRequest(
  canon: CanonReq,
  requires: Requirement,
  jsonMode: boolean,
  clientWire: 'openai' | 'anthropic',
): Promise<ServedAnswer | null> {
  const list = candidatesFor(requires, jsonMode, clientWire);
  if (list.length === 0) return null;
  let firstTried = -1;
  let retriedFirst = false;
  for (let i = 0; i < list.length; i++) {
    const candidate = list[i];
    if (exhausted.has(candidate.provider.id)) continue;
    if (firstTried === -1) firstTried = i;
    const base = joinBase(ENV_URL, candidate.provider.base);
    let upstream: Response;
    try {
      upstream = await callUpstream(
        candidate.provider.wire,
        base,
        candidate.provider.auth,
        candidate.provider.key,
        canon,
        candidate.model.id,
      );
    } catch {
      noteRejection(candidate.provider.id);
      if (i === firstTried && !retriedFirst) {
        retriedFirst = true;
        i--;
      }
      continue;
    }
    if (!upstream.ok) {
      noteRejection(candidate.provider.id);
      if (i === firstTried && !retriedFirst) {
        retriedFirst = true;
        i--;
      }
      continue;
    }
    noteSuccess(candidate.provider.id);
    if (canon.stream) {
      if (upstream.body === null) {
        noteRejection(candidate.provider.id);
        continue;
      }
      return { kind: 'stream', deltas: streamToCanon(candidate.provider.wire, upstream.body), model: candidate.model.id };
    }
    let body: Json;
    try {
      const parsed: unknown = await upstream.json();
      if (!isRecord(parsed)) throw new Error('upstream answer is not a JSON object');
      body = parsed;
    } catch {
      noteRejection(candidate.provider.id);
      if (i === firstTried && !retriedFirst) {
        retriedFirst = true;
        i--;
      }
      continue;
    }
    const resp = respToCanon(candidate.provider.wire, body, candidate.model.id);
    return { kind: 'json', resp, model: candidate.model.id };
  }
  return null;
}

// GET /v1/models. The wire is decided by the auth header the client sent: an x-api-key
// header means the Anthropic shape, anything else the OpenAI shape. That is how the env's
// warm-up asks twice, once per wire, and how the catalog's free models are listed.
async function handleModels(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const models = servedModels();
  if (req.headers['x-api-key'] !== undefined) {
    sendJson(res, 200, anthropicModelList(models));
  } else {
    sendJson(res, 200, openAIModelList(models));
  }
}

// POST /v1/chat/completions, JSON or SSE. The body is read into canonical form, served,
// and answered in the OpenAI shape the client used.
async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: Json;
  try {
    const text = await readBody(req);
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) throw new Error('body is not a JSON object');
    body = parsed;
  } catch {
    sendError(res, 400, 'the request body is not valid JSON', 'openai');
    return;
  }
  let canon: CanonReq;
  try {
    canon = canonFromOpenAI(body);
  } catch {
    sendError(res, 400, 'the request body is not a valid chat completion', 'openai');
    return;
  }
  const requires = requirementsFromBody(body, 'openai');
  const answer = await serveRequest(canon, requires, canon.json, 'openai');
  if (answer === null) {
    sendError(res, 502, 'no free model could serve this request', 'openai');
    return;
  }
  if (answer.kind === 'json') {
    sendJson(res, 200, canonToOpenAIResp(answer.resp, answer.model));
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const chunk of canonToOpenAIStream(answer.deltas, answer.model, STREAM_ID)) {
    res.write(chunk);
  }
  res.end();
}

// POST /v1/messages, JSON or SSE. The body is read into canonical form, served, and
// answered in the Anthropic shape the client used.
async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: Json;
  try {
    const text = await readBody(req);
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) throw new Error('body is not a JSON object');
    body = parsed;
  } catch {
    sendError(res, 400, 'the request body is not valid JSON', 'anthropic');
    return;
  }
  let canon: CanonReq;
  try {
    canon = canonFromAnthropic(body);
  } catch {
    sendError(res, 400, 'the request body is not a valid message', 'anthropic');
    return;
  }
  const requires = requirementsFromBody(body, 'anthropic');
  const answer = await serveRequest(canon, requires, canon.json, 'anthropic');
  if (answer === null) {
    sendError(res, 502, 'no free model could serve this request', 'anthropic');
    return;
  }
  if (answer.kind === 'json') {
    sendJson(res, 200, canonToAnthropicResp(answer.resp, answer.model));
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const chunk of canonToAnthropicStream(answer.deltas, answer.model)) {
    res.write(chunk);
  }
  res.end();
}

// One request: route by method and path. Unknown routes are a JSON 404, never HTML, and
// never a 403: the wire shape follows the path's wire where the path names one.
function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    void handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && path === '/v1/chat/completions') {
    void handleChat(req, res);
    return;
  }
  if (req.method === 'POST' && path === '/v1/messages') {
    void handleMessages(req, res);
    return;
  }
  const wire = path === '/v1/messages' ? 'anthropic' : 'openai';
  sendError(res, 404, `no such route: ${path}`, wire);
}

// Probe every provider's catalog route, then build the catalog from the probes and the
// quality table. A provider that does not answer is left out of the catalog; the trial
// goes on with the providers that did.
async function load(): Promise<void> {
  const next: Probe[] = [];
  for (const provider of brief.directory.providers) {
    const probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    if (!probe.ok) log(`probe ${provider.id} failed: ${probe.error ?? 'unknown error'}`);
    next.push(probe);
  }
  probes = next;
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  log(`catalog: ${catalog.providers.length} providers, ${catalog.providers.reduce((n, p) => n + p.models.length, 0)} models`);
}

// Listen on an ephemeral port on loopback and return the port number. The env reaches the
// entry on this port; nothing else needs to know it.
function listen(): Promise<number> {
  return new Promise((resolve, reject) => {
    server = createServer(handle);
    server.listen(0, '127.0.0.1', () => {
      const address = server === null ? null : server.address();
      if (address !== null && typeof address === 'object') {
        log(`listening on 127.0.0.1:${address.port}`);
        resolve(address.port);
      } else {
        reject(new Error('could not bind a port'));
      }
    });
    server.on('error', reject);
  });
}

// Register the port and the catalog with the env. The env answers 200 with the number of
// requests it will send; anything else is a failure and the trial cannot start.
async function register(port: number): Promise<void> {
  if (catalog === null) throw new Error('no catalog to register');
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ port, protocol_version: PROTOCOL_VERSION, catalog }),
  });
  if (!res.ok) throw new Error(`register answered HTTP ${res.status}`);
  const body: unknown = await res.json();
  if (!isRecord(body) || body.ok !== true) throw new Error('register did not answer ok');
  log(`registered, ${String(body.requests)} requests to serve`);
}

// Ask the env whether the workload is done. True means every request has been sent and the
// entry may exit; false means the env is still sending, so the entry keeps serving.
async function pollWork(): Promise<boolean> {
  const res = await fetch(`${ENV_URL}/work`);
  if (!res.ok) throw new Error(`work answered HTTP ${res.status}`);
  const body: unknown = await res.json();
  return isRecord(body) && body.done === true;
}

// The trial: load the catalog, listen, register, then serve until the env says the workload
// is done, and exit zero. Any failure before the workload is done exits non-zero.
async function start(): Promise<void> {
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (PROMPT_FILE === '') throw new Error('MAKEFUN_PROMPT_FILE is not set');
  await load();
  const port = await listen();
  await register(port);
  for (;;) {
    if (await pollWork()) break;
    await sleep(POLL_INTERVAL_MS);
  }
  log('workload done, exiting');
  if (server !== null) {
    await new Promise<void>((resolve) => server?.close(() => resolve()));
  }
  process.exit(0);
}

start().catch((err: unknown) => {
  log(`failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
