// The canonical request and response: one shape every client wire is read into and every
// provider wire is written from. Keeping one form means the three upstream wires (OpenAI,
// Anthropic, Gemini) share the same routing, failover and budget logic, and a request that
// fails on one provider can be retried on another wire without being rebuilt.
//
// A canonical request is a system prompt, a list of turns, tool definitions, a tool choice,
// generation knobs and the JSON-mode flag. A turn is a role with an ordered list of blocks, so
// text, images, tool calls and tool results keep their order and their pairing.
//
// Everything here is a pure function of its arguments: no clock, no randomness, so a replay
// sees the same bytes.

import { isRecord, type Json } from './providers.ts';

export type CanonRole = 'user' | 'assistant';

export type CanonBlock =
  | { kind: 'text'; text: string }
  | { kind: 'image'; data: string; media: string }
  | { kind: 'tool_call'; id: string; name: string; args: Json }
  | { kind: 'tool_result'; id: string; text: string };

export interface CanonMsg {
  role: CanonRole;
  blocks: CanonBlock[];
}

export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

export type CanonToolChoice = 'auto' | 'required' | { name: string } | null;

export interface CanonReq {
  system: string;
  messages: CanonMsg[];
  tools: CanonTool[];
  toolChoice: CanonToolChoice;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string[];
  json: boolean;
  stream: boolean;
}

/** A canonical answer: the text, the tool calls, the stop reason and the token counts. */
export interface CanonResp {
  text: string;
  toolCalls: { id: string; name: string; args: Json }[];
  stopReason: 'end_turn' | 'max_tokens' | 'tool_use' | 'stop';
  inputTokens: number;
  outputTokens: number;
  id: string;
}

const textOf = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
};

const argsOf = (value: unknown): Json => {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
};

const numOr = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

/** An OpenAI Chat Completions body as a canonical request. */
export function canonFromOpenAI(body: Json): CanonReq {
  const messages: CanonMsg[] = [];
  let system = '';
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const r of raw) {
    if (!isRecord(r)) continue;
    const role = r.role;
    if (role === 'system') {
      const t = textOf(r.content);
      system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (role === 'tool') {
      // A tool answer joins the previous user turn (or opens one) as a tool_result block.
      const last = messages[messages.length - 1];
      const block: CanonBlock = { kind: 'tool_result', id: typeof r.tool_call_id === 'string' ? r.tool_call_id : '', text: textOf(r.content) };
      if (last !== undefined && last.role === 'user') last.blocks.push(block);
      else messages.push({ role: 'user', blocks: [block] });
      continue;
    }
    if (role !== 'user' && role !== 'assistant') continue;
    const blocks: CanonBlock[] = [];
    const content = r.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const m = /^data:([^;]+);base64,(.*)$/.exec(url);
          if (m !== null) blocks.push({ kind: 'image', data: m[2] as string, media: m[1] as string });
          else blocks.push({ kind: 'text', text: url });
        }
      }
    }
    if (role === 'assistant' && Array.isArray(r.tool_calls)) {
      for (const call of r.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          kind: 'tool_call',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argsOf(call.function.arguments),
        });
      }
    }
    messages.push({ role, blocks });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonToolChoice = null;
  const tc = body.tool_choice;
  if (tc === 'auto' || tc === 'required') toolChoice = tc;
  else if (isRecord(tc) && tc.type === 'function' && isRecord(tc.function) && typeof tc.function.name === 'string') toolChoice = { name: tc.function.name };
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numOr(body.max_tokens),
    temperature: numOr(body.temperature),
    topP: numOr(body.top_p),
    stop: Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : undefined,
    json: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a canonical request. */
export function canonFromAnthropic(body: Json): CanonReq {
  const messages: CanonMsg[] = [];
  const system = textOf(body.system);
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const r of raw) {
    if (!isRecord(r)) continue;
    const role = r.role === 'assistant' ? 'assistant' : 'user';
    const blocks: CanonBlock[] = [];
    const content = r.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            blocks.push({ kind: 'image', data: src.data, media: typeof src.media_type === 'string' ? src.media_type : 'image/png' });
          } else if (src.type === 'url' && typeof src.url === 'string') blocks.push({ kind: 'text', text: src.url });
        } else if (block.type === 'tool_use') {
          blocks.push({
            kind: 'tool_call',
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          });
        } else if (block.type === 'tool_result') {
          const c = block.content;
          const text = typeof c === 'string' ? c : Array.isArray(c) ? textOf(c) : c === undefined ? '' : JSON.stringify(c);
          blocks.push({ kind: 'tool_result', id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text });
        }
      }
    }
    messages.push({ role, blocks });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonToolChoice = null;
  const tc = body.tool_choice;
  if (isRecord(tc)) {
    if (tc.type === 'auto') toolChoice = 'auto';
    else if (tc.type === 'any') toolChoice = 'required';
    else if (tc.type === 'tool' && typeof tc.name === 'string') toolChoice = { name: tc.name };
  }
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numOr(body.max_tokens),
    temperature: numOr(body.temperature),
    topP: numOr(body.top_p),
    stop: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : undefined,
    json: false,
    stream: body.stream === true,
  };
}

/** A canonical request as an OpenAI Chat Completions body naming `model`. */
export function canonToOpenAI(canon: CanonReq, model: string): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const msg of canon.messages) {
    if (msg.role === 'assistant') {
      const toolCalls: Json[] = [];
      const text = msg.blocks.flatMap((b) => (b.kind === 'text' ? [b.text] : [])).join('\n');
      for (const b of msg.blocks) {
        if (b.kind === 'tool_call') {
          toolCalls.push({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.args) } });
        }
      }
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const b of msg.blocks) {
      if (b.kind === 'text') parts.push({ type: 'text', text: b.text });
      else if (b.kind === 'image') parts.push({ type: 'image_url', image_url: { url: `data:${b.media};base64,${b.data}` } });
      else if (b.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: b.id, content: b.text });
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (canon.maxTokens !== undefined) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== undefined) out.temperature = canon.temperature;
  if (canon.topP !== undefined) out.top_p = canon.topP;
  if (canon.stop !== undefined && canon.stop.length > 0) out.stop = canon.stop;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (canon.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (canon.toolChoice === 'required') out.tool_choice = 'required';
  else if (canon.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: canon.toolChoice.name } };
  if (canon.json) out.response_format = { type: 'json_object' };
  if (canon.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A canonical request as an Anthropic Messages body naming `model`. */
export function canonToAnthropic(canon: CanonReq, model: string): Json {
  const messages: Json[] = [];
  for (const msg of canon.messages) {
    const blocks: Json[] = [];
    for (const b of msg.blocks) {
      if (b.kind === 'text') blocks.push({ type: 'text', text: b.text });
      else if (b.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: b.media, data: b.data } });
      else if (b.kind === 'tool_call') blocks.push({ type: 'tool_use', id: b.id, name: b.name, input: b.args });
      else if (b.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: b.id, content: b.text });
    }
    if (blocks.length > 0) messages.push({ role: msg.role, content: blocks });
  }
  const out: Json = { model, messages, max_tokens: canon.maxTokens ?? 1024 };
  if (canon.system !== '') out.system = canon.system;
  if (canon.temperature !== undefined) out.temperature = canon.temperature;
  if (canon.topP !== undefined) out.top_p = canon.topP;
  if (canon.stop !== undefined && canon.stop.length > 0) out.stop_sequences = canon.stop;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (canon.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (canon.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (canon.toolChoice !== null) out.tool_choice = { type: 'tool', name: canon.toolChoice.name };
  if (canon.stream) out.stream = true;
  return out;
}

/** A canonical request as a Gemini generateContent body (the model goes in the URL). */
export function canonToGemini(canon: CanonReq): Json {
  const contents: Json[] = [];
  for (const msg of canon.messages) {
    const parts: Json[] = [];
    for (const b of msg.blocks) {
      if (b.kind === 'text') parts.push({ text: b.text });
      else if (b.kind === 'image') parts.push({ inlineData: { mimeType: b.media, data: b.data } });
      else if (b.kind === 'tool_call') parts.push({ functionCall: { name: b.name, args: b.args } });
      else if (b.kind === 'tool_result') parts.push({ functionResponse: { name: '', response: { result: b.text } } });
    }
    if (parts.length > 0) contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  // Gemini wants a user turn first; a leading model turn is merged into the first user turn.
  const out: Json = { contents };
  if (canon.system !== '') out.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
    if (canon.toolChoice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (canon.toolChoice !== null && typeof canon.toolChoice === 'object') {
      out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [canon.toolChoice.name] } };
    }
  }
  const gen: Json = {};
  if (canon.maxTokens !== undefined) gen.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== undefined) gen.temperature = canon.temperature;
  if (canon.topP !== undefined) gen.topP = canon.topP;
  if (canon.stop !== undefined && canon.stop.length > 0) gen.stopSequences = canon.stop;
  if (canon.json) gen.responseMimeType = 'application/json';
  if (Object.keys(gen).length > 0) out.generationConfig = gen;
  return out;
}
