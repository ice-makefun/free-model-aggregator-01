// The upstream call: one call at a time per provider, with retries and failover.
//
// A provider answers one request at a time. A second call while one is in flight is refused
// with 429 `concurrent_requests`, so the entry serialises its calls per provider: a queue
// per provider id, drained one call at a time, in the order the calls were made. The queue
// is the only shared state here, and it is the only place a call waits.
//
// A call is a POST to the provider's base with the provider's auth headers and the body the
// caller built in the provider's wire. The answer is read in full (a stream is collected
// into one body — the trial's providers answer a stream as a single SSE document, and the
// entry re-emits it as a stream to its own client). The call reports one of three outcomes:
//
//   ok       the provider answered 200 with a body: the body, parsed as JSON when it parses.
//   refused  the provider answered 402, 429 or 5xx: the status and a short message. A 402 is
//            a refusal to serve (the model is not free here); a 429 is a rate limit; a 5xx
//            is the provider failing. All three mean "try the next candidate".
//   error    the network failed or the answer could not be read: a message, tried as a
//            refusal with status 502.
//
// A refusal is not retried by this module: the caller decides whether the candidate gets
// another attempt. The module's job is only to make the call, read the answer, and say
// plainly what happened — and to never have two calls in flight to the same provider.

import type { Json } from './providers.ts';

export type Refusal = { kind: 'refused'; status: number; message: string };
export type CallResult =
  | { ok: true; body: Json; status: number }
  | { ok: false; refusal: Refusal };

const queues = new Map<string, Promise<unknown>>();

/** Run `task` after every earlier task queued under `key` has finished. */
function enqueue<T>(key: string, task: () => Promise<T>): Promise<T> {
  const previous = queues.get(key) ?? Promise.resolve();
  const run = previous.then(task, task);
  queues.set(key, run.catch(() => undefined));
  return run;
}

function refusal(status: number, message: string): Refusal {
  return { kind: 'refused', status, message };
}

/**
 * POST `body` to `base` + `/chat/completions` (OpenAI wire) or `/messages` (Anthropic
 * wire) or the Gemini generate endpoint, with `headers`, and read the answer. The path
 * depends on the wire: OpenAI and Anthropic take their usual paths under the base, and
 * Gemini takes `/v1beta/models/<model>:generateContent` — the model is in the body.
 */
export async function callUpstreamOnce(
  base: string,
  wire: string,
  headers: Record<string, string>,
  body: Json,
  _stream: boolean,
): Promise<CallResult> {
  let path = '/v1/chat/completions';
  if (wire === 'anthropic') path = '/v1/messages';
  if (wire === 'gemini') {
    const model = typeof body.model === 'string' ? body.model : '';
    path = `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  }
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    });
  } catch (err) {
    return { ok: false, refusal: refusal(502, `upstream request failed: ${(err as Error).message}`) };
  }
  const text = await res.text().catch(() => '');
  if (res.status === 200) {
    let parsed: Json = {};
    try {
      parsed = JSON.parse(text) as Json;
    } catch {
      parsed = { text };
    }
    return { ok: true, body: parsed, status: 200 };
  }
  if (res.status === 402 || res.status === 429 || res.status >= 500) {
    return { ok: false, refusal: refusal(res.status, `upstream answered HTTP ${res.status}: ${text.slice(0, 300)}`) };
  }
  return { ok: false, refusal: refusal(res.status, `upstream answered HTTP ${res.status}: ${text.slice(0, 300)}`) };
}

/**
 * Call a provider, one call at a time: the call is queued behind any earlier call to the
 * same provider base, so two calls to one provider never overlap. The queue key is the
 * provider's base URL, which names the provider.
 */
export function callUpstream(
  base: string,
  wire: string,
  headers: Record<string, string>,
  body: Json,
  stream: boolean,
): Promise<CallResult> {
  return enqueue(base, () => callUpstreamOnce(base, wire, headers, body, stream));
}
