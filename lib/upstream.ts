// The three upstream wires, spoken from the canonical form, and the answers read back into it.
//
// Every provider is called the way the real service is called (its route, its auth header, its
// body shape), and every answer — JSON or server-sent events — is reduced to the same canonical
// response or the same stream of canonical deltas. That is what lets one request be retried on a
// provider of another wire without being rebuilt, and what keeps a replay byte-identical: the
// translation is a pure function of the request and the provider's answer.
//
// Streaming: each wire's SSE is parsed into canonical deltas (text, tool-call starts and argument
// pieces, the stop), and the client's wire is written from those deltas. The client's event order
// is the wire's own: OpenAI's chunk stream, Anthropic's message_start / content_block_* /
// message_delta / message_stop. Tool-call argument pieces are held and emitted in order.

import { isRecord, type Json } from './providers.ts';
import type { CanonReq, CanonResp } from './canon.ts';
import { canonToAnthropic, canonToGemini, canonToOpenAI } from './canon.ts';

export type Wire = 'openai' | 'anthropic' | 'gemini';

/** The auth headers a provider wants its key in. */
export function upstreamHeaders(auth: string, key: string, wire: Wire): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (key === '') return headers;
  if (auth === 'bearer') headers.authorization = `Bearer ${key}`;
  else if (auth === 'x-api-key') headers['x-api-key'] = key;
  else if (auth === 'x-goog-api-key') headers['x-goog-api-key'] = key;
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** POST one canonical request to a provider on its own wire. Returns the raw Response. */
export async function callUpstream(
  wire: Wire,
  base: string,
  auth: string,
  key: string,
  canon: CanonReq,
  model: string,
): Promise<Response> {
  const headers = upstreamHeaders(auth, key, wire);
  if (canon.stream) headers.accept = 'text/event-stream';
  if (wire === 'openai') {
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(canonToOpenAI(canon, model)) });
  }
  if (wire === 'anthropic') {
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(canonToAnthropic(canon, model)) });
  }
  const path = canon.stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
  return fetch(`${base}/v1beta/models/${encodeURIComponent(model)}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(canonToGemini(canon)),
  });
}

const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** An OpenAI JSON answer as a canonical response. */
export function openAIRespToCanon(body: Json, model: string): CanonResp {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : '';
  const toolCalls: CanonResp['toolCalls'] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let args: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: an empty input.
        }
      } else if (isRecord(call.function.arguments)) args = call.function.arguments;
      toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', args });
    }
  }
  const finish = choice.finish_reason;
  const stopReason: CanonResp['stopReason'] = finish === 'length' ? 'max_tokens' : finish === 'tool_calls' || finish === 'function_call' ? 'tool_use' : 'end_turn';
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stopReason,
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    id: typeof body.id === 'string' ? body.id : `chatcmpl-${model}`,
  };
}

/** An Anthropic JSON answer as a canonical response. */
export function anthropicRespToCanon(body: Json, model: string): CanonResp {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: CanonResp['toolCalls'] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        args: isRecord(block.input) ? block.input : {},
      });
    }
  }
  const stop = body.stop_reason;
  const stopReason: CanonResp['stopReason'] = stop === 'max_tokens' ? 'max_tokens' : stop === 'tool_use' ? 'tool_use' : 'end_turn';
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stopReason,
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    id: typeof body.id === 'string' ? body.id : `msg-${model}`,
  };
}

/** A Gemini JSON answer as a canonical response. */
export function geminiRespToCanon(body: Json, model: string): CanonResp {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: CanonResp['toolCalls'] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      const fc = part.functionCall;
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: typeof fc.name === 'string' ? fc.name : '',
        args: isRecord(fc.args) ? fc.args : {},
      });
    }
  }
  const finish = cand.finishReason;
  const stopReason: CanonResp['stopReason'] = finish === 'MAX_TOKENS' ? 'max_tokens' : toolCalls.length > 0 ? 'tool_use' : 'end_turn';
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    stopReason,
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    id: `gen-${model}`,
  };
}

/** One JSON answer, in whichever wire it arrived, as a canonical response. */
export function respToCanon(wire: Wire, body: Json, model: string): CanonResp {
  if (wire === 'anthropic') return anthropicRespToCanon(body, model);
  if (wire === 'gemini') return geminiRespToCanon(body, model);
  return openAIRespToCanon(body, model);
}

/** A canonical response as an OpenAI Chat Completions JSON body. */
export function canonToOpenAIResp(resp: CanonResp, model: string): Json {
  const message: Json = { role: 'assistant', content: resp.text !== '' ? resp.text : null };
  if (resp.toolCalls.length > 0) {
    message.tool_calls = resp.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
  }
  const finish = resp.stopReason === 'max_tokens' ? 'length' : resp.stopReason === 'tool_use' ? 'tool_calls' : 'stop';
  return {
    id: resp.id,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage: { prompt_tokens: resp.inputTokens, completion_tokens: resp.outputTokens, total_tokens: resp.inputTokens + resp.outputTokens },
  };
}

/** A canonical response as an Anthropic Messages JSON body. */
export function canonToAnthropicResp(resp: CanonResp, model: string): Json {
  const content: Json[] = [];
  if (resp.text !== '') content.push({ type: 'text', text: resp.text });
  for (const c of resp.toolCalls) content.push({ type: 'tool_use', id: c.id, name: c.name, input: c.args });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${resp.id}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: resp.stopReason,
    stop_sequence: null,
    usage: { input_tokens: resp.inputTokens, output_tokens: resp.outputTokens },
  };
}

/** One piece of a streamed answer, in wire-neutral form. */
export type CanonDelta =
  | { kind: 'text'; text: string }
  | { kind: 'tool_start'; index: number; id: string; name: string }
  | { kind: 'tool_args'; index: number; text: string }
  | { kind: 'stop'; stopReason: CanonResp['stopReason']; inputTokens: number; outputTokens: number };

/** Split an SSE byte stream into its frames (blank-line separated). */
async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** The `data:` payloads of a frame, in order. */
function dataLines(frame: string): string[] {
  const out: string[] = [];
  for (const line of frame.split('\n')) {
    const t = line.trim();
    if (t.startsWith('data:')) out.push(t.slice(5).trim());
  }
  return out;
}

/** An OpenAI SSE stream as canonical deltas. */
export async function* openAIStreamToCanon(stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonDelta> {
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: CanonResp['stopReason'] | null = null;
  const toolIndex = new Map<number, number>();
  let nextTool = 0;
  for await (const frame of sseFrames(stream)) {
    for (const payload of dataLines(frame)) {
      if (payload === '[DONE]') {
        yield { kind: 'stop', stopReason: pendingStop ?? 'end_turn', inputTokens, outputTokens };
        return;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numOrZero(chunk.usage.prompt_tokens);
        outputTokens = numOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content !== '') yield { kind: 'text', text: delta.content };
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (!isRecord(call)) continue;
          const upstreamIndex = typeof call.index === 'number' ? call.index : 0;
          let index = toolIndex.get(upstreamIndex);
          if (index === undefined) {
            index = nextTool;
            nextTool += 1;
            toolIndex.set(upstreamIndex, index);
            const fn = isRecord(call.function) ? call.function : {};
            yield { kind: 'tool_start', index, id: typeof call.id === 'string' ? call.id : `call_${index}`, name: typeof fn.name === 'string' ? fn.name : '' };
          }
          const fn = isRecord(call.function) ? call.function : {};
          if (typeof fn.arguments === 'string' && fn.arguments !== '') yield { kind: 'tool_args', index, text: fn.arguments };
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = choice.finish_reason === 'length' ? 'max_tokens' : choice.finish_reason === 'tool_calls' || choice.finish_reason === 'function_call' ? 'tool_use' : 'end_turn';
      }
    }
  }
  yield { kind: 'stop', stopReason: pendingStop ?? 'end_turn', inputTokens, outputTokens };
}

/** An Anthropic SSE stream as canonical deltas. */
export async function* anthropicStreamToCanon(stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonDelta> {
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: CanonResp['stopReason'] | null = null;
  const toolIndex = new Map<number, number>();
  let nextTool = 0;
  let currentText = false;
  for await (const frame of sseFrames(stream)) {
    let eventName: string | null = null;
    for (const line of frame.split('\n')) {
      const t = line.trim();
      if (t.startsWith('event:')) eventName = t.slice(6).trim();
    }
    for (const payload of dataLines(frame)) {
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
        inputTokens = numOrZero(data.message.usage.input_tokens);
      } else if (eventName === 'content_block_start' && isRecord(data.content_block)) {
        const block = data.content_block;
        const index = typeof data.index === 'number' ? data.index : 0;
        if (block.type === 'tool_use') {
          toolIndex.set(index, nextTool);
          yield { kind: 'tool_start', index: nextTool, id: typeof block.id === 'string' ? block.id : `call_${nextTool}`, name: typeof block.name === 'string' ? block.name : '' };
          nextTool += 1;
        } else if (block.type === 'text') currentText = true;
      } else if (eventName === 'content_block_delta' && isRecord(data.delta)) {
        const delta = data.delta;
        const index = typeof data.index === 'number' ? data.index : 0;
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') yield { kind: 'text', text: delta.text };
        else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          const mapped = toolIndex.get(index);
          if (mapped !== undefined) yield { kind: 'tool_args', index: mapped, text: delta.partial_json };
        }
      } else if (eventName === 'content_block_stop') {
        currentText = false;
      } else if (eventName === 'message_delta' && isRecord(data.delta)) {
        const delta = data.delta;
        if (typeof delta.stop_reason === 'string') {
          pendingStop = delta.stop_reason === 'max_tokens' ? 'max_tokens' : delta.stop_reason === 'tool_use' ? 'tool_use' : 'end_turn';
        }
        if (isRecord(data.usage)) outputTokens = numOrZero(data.usage.output_tokens);
      }
    }
  }
  yield { kind: 'stop', stopReason: pendingStop ?? 'end_turn', inputTokens, outputTokens };
}

/** A Gemini SSE stream (streamGenerateContent) as canonical deltas. */
export async function* geminiStreamToCanon(stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonDelta> {
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: CanonResp['stopReason'] | null = null;
  const toolIndex = new Map<string, number>();
  let nextTool = 0;
  for await (const frame of sseFrames(stream)) {
    for (const payload of dataLines(frame)) {
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string' && part.text !== '') yield { kind: 'text', text: part.text };
        else if (isRecord(part.functionCall)) {
          const fc = part.functionCall;
          const name = typeof fc.name === 'string' ? fc.name : '';
          let index = toolIndex.get(name);
          if (index === undefined) {
            index = nextTool;
            nextTool += 1;
            toolIndex.set(name, index);
            yield { kind: 'tool_start', index, id: `call_${index}`, name };
          }
          if (isRecord(fc.args)) yield { kind: 'tool_args', index, text: JSON.stringify(fc.args) };
        }
      }
      if (typeof cand.finishReason === 'string') {
        pendingStop = cand.finishReason === 'MAX_TOKENS' ? 'max_tokens' : 'end_turn';
      }
      if (isRecord(data.usageMetadata)) {
        inputTokens = numOrZero(data.usageMetadata.promptTokenCount);
        outputTokens = numOrZero(data.usageMetadata.candidatesTokenCount);
      }
    }
  }
  yield { kind: 'stop', stopReason: pendingStop ?? 'end_turn', inputTokens, outputTokens };
}

/** One upstream SSE stream, in its wire, as canonical deltas. */
export function streamToCanon(wire: Wire, stream: ReadableStream<Uint8Array>): AsyncGenerator<CanonDelta> {
  if (wire === 'anthropic') return anthropicStreamToCanon(stream);
  if (wire === 'gemini') return geminiStreamToCanon(stream);
  return openAIStreamToCanon(stream);
}

const sse = (event: string | null, data: unknown): string =>
  `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

/** Canonical deltas as an OpenAI SSE stream. `id` is the completion id the client sees. */
export async function* canonToOpenAIStream(
  deltas: AsyncGenerator<CanonDelta>,
  model: string,
  id: string,
): AsyncGenerator<string> {
  let finished = false;
  for await (const d of deltas) {
    if (d.kind === 'text') {
      yield sse(null, { id, object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { role: 'assistant', content: d.text }, finish_reason: null }] });
    } else if (d.kind === 'tool_start') {
      yield sse(null, {
        id,
        object: 'chat.completion.chunk',
        created: 1767225600,
        model,
        choices: [{ index: 0, delta: { tool_calls: [{ index: d.index, id: d.id, type: 'function', function: { name: d.name, arguments: '' } }] }, finish_reason: null }],
      });
    } else if (d.kind === 'tool_args') {
      yield sse(null, {
        id,
        object: 'chat.completion.chunk',
        created: 1767225600,
        model,
        choices: [{ index: 0, delta: { tool_calls: [{ index: d.index, function: { arguments: d.text } }] }, finish_reason: null }],
      });
    } else if (d.kind === 'stop' && !finished) {
      finished = true;
      const finish = d.stopReason === 'max_tokens' ? 'length' : d.stopReason === 'tool_use' ? 'tool_calls' : 'stop';
      yield sse(null, { id, object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: {}, finish_reason: finish }] });
      yield sse(null, {
        id,
        object: 'chat.completion.chunk',
        created: 1767225600,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: null }],
        usage: { prompt_tokens: d.inputTokens, completion_tokens: d.outputTokens, total_tokens: d.inputTokens + d.outputTokens },
      });
      yield 'data: [DONE]\n\n';
    }
  }
  if (!finished) yield 'data: [DONE]\n\n';
}

/** Canonical deltas as an Anthropic SSE stream. */
export async function* canonToAnthropicStream(deltas: AsyncGenerator<CanonDelta>, model: string): AsyncGenerator<string> {
  let started = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let finished = false;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sse('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
    ];
  };
  const close = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sse('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  for await (const d of deltas) {
    if (d.kind === 'text') {
      const out = [...start()];
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sse('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sse('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: d.text } }));
      for (const event of out) yield event;
    } else if (d.kind === 'tool_start') {
      const out = [...start()];
      if (textIndex !== -1) {
        out.push(sse('content_block_stop', { type: 'content_block_stop', index: textIndex }));
        textIndex = -1;
      }
      const index = nextIndex;
      nextIndex += 1;
      toolIndexes.set(d.index, index);
      out.push(sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: d.id, name: d.name, input: {} } }));
      for (const event of out) yield event;
    } else if (d.kind === 'tool_args') {
      const index = toolIndexes.get(d.index);
      if (index === undefined) continue;
      yield sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: d.text } });
    } else if (d.kind === 'stop' && !finished) {
      finished = true;
      const out = [...start(), ...close()];
      out.push(
        sse('message_delta', { type: 'message_delta', delta: { stop_reason: d.stopReason, stop_sequence: null }, usage: { output_tokens: d.outputTokens } }),
        sse('message_stop', { type: 'message_stop' }),
      );
      for (const event of out) yield event;
    }
  }
  if (!finished) {
    for (const event of [...start(), ...close()]) yield event;
    yield sse('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 0 } });
    yield sse('message_stop', { type: 'message_stop' });
  }
}
