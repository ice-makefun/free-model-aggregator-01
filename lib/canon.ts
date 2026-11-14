// The canonical form: one shape every wire translates through.
//
// A request arrives in one of two client wires (OpenAI Chat Completions or Anthropic
// Messages) and is answered by a provider on one of three wires (OpenAI, Anthropic, Gemini).
// Rather than translating every pair of wires directly — six translators, each a chance to
// lose a field — every wire translates to and from a single canonical request and a single
// canonical answer. A field that survives the round trip through the canonical form is a
// field the entry understands; a field that does not is a field the entry drops on purpose.
//
// The canonical request carries: the model the client asked for (or null, when the client
// lets the entry choose), the system prompt, the messages (text, images and tool calls, in
// order), the tools the client offered, the stop sequences, the temperature, the maximum
// tokens, whether the client wants JSON output, whether it wants a stream, and the minimum
// context the model must have. The canonical answer carries: the model that answered, the
// text, the tool calls the model made, the stop reason, and the usage.
//
// Nothing here touches the network or the clock. The functions are pure: the same input
// always gives the same output, so a replay translates identically.

import { isRecord, type Json } from './providers.ts';

export interface CanonicalMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string;
  images: { mime: string; data: string }[];
  toolCalls: { id: string; name: string; arguments: string }[];
  toolCallId: string;
}

export interface CanonicalTool {
  name: string;
  description: string;
  schema: Json;
}

export interface CanonicalRequest {
  model: string | null;
  system: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  stop: string[];
  temperature: number | null;
  maxTokens: number | null;
  jsonMode: boolean;
  stream: boolean;
  minContext: number | null;
  images: { mime: string; data: string }[];
}

export interface CanonicalAnswer {
  model: string;
  text: string;
  toolCalls: { id: string; name: string; arguments: string }[];
  stopReason: string | null;
  usage: { inputTokens: number; outputTokens: number };
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberOf(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** The messages of an OpenAI request, in canonical form. */
function openAIMessages(body: Json): { system: string; messages: CanonicalMessage[]; images: { mime: string; data: string }[] } {
  let system = '';
  const messages: CanonicalMessage[] = [];
  const images: { mime: string; data: string }[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    const role = entry.role === 'system' || entry.role === 'user' || entry.role === 'assistant' || entry.role === 'tool' ? entry.role : 'user';
    if (role === 'system') {
      system += (system === '' ? '' : '\n') + openAIText(entry.content, images);
      continue;
    }
    const text = openAIText(entry.content, images);
    const toolCalls: CanonicalMessage['toolCalls'] = [];
    if (Array.isArray(entry.tool_calls)) {
      for (const call of entry.tool_calls) {
        if (!isRecord(call)) continue;
        const fn = isRecord(call.function) ? call.function : {};
        toolCalls.push({ id: textOf(call.id), name: textOf(fn.name), arguments: textOf(fn.arguments) });
      }
    }
    messages.push({ role, text, images: [], toolCalls, toolCallId: textOf(entry.tool_call_id) });
  }
  return { system, messages, images };
}

function openAIText(content: unknown, images: { mime: string; data: string }[]): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let text = '';
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') text += part.text;
    if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const match = /^data:([^;]+);base64,(.*)$/.exec(url);
      if (match) images.push({ mime: match[1], data: match[2] });
    }
  }
  return text;
}

/** An OpenAI Chat Completions request, in canonical form. */
export function openAIToCanonical(body: Json, _model: string | null): CanonicalRequest {
  const { system, messages, images } = openAIMessages(body);
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const entry of body.tools) {
      if (!isRecord(entry) || entry.type !== 'function' || !isRecord(entry.function)) continue;
      tools.push({ name: textOf(entry.function.name), description: textOf(entry.function.description), schema: isRecord(entry.function.parameters) ? entry.function.parameters : {} });
    }
  }
  const stop = Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  const responseFormat = isRecord(body.response_format) ? body.response_format : {};
  return {
    model: textOf(body.model) || null,
    system,
    messages,
    tools,
    stop,
    temperature: numberOf(body.temperature),
    maxTokens: numberOf(body.max_tokens),
    jsonMode: responseFormat.type === 'json_object' || responseFormat.type === 'json_schema',
    stream: body.stream === true,
    minContext: null,
    images,
  };
}

/** A canonical request, as an OpenAI Chat Completions body for the given model. */
export function canonicalToOpenAI(canonical: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (canonical.system !== '') messages.push({ role: 'system', content: canonical.system });
  for (const message of canonical.messages) {
    const content: Json[] = [];
    if (message.text !== '') content.push({ type: 'text', text: message.text });
    for (const image of message.images) {
      content.push({ type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } });
    }
    const entry: Json = { role: message.role, content: content.length === 1 && typeof content[0] === 'object' && (content[0] as Json).type === 'text' ? message.text : content };
    if (message.toolCalls.length > 0) {
      entry.tool_calls = message.toolCalls.map((call) => ({ id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }));
    }
    if (message.toolCallId !== '') entry.tool_call_id = message.toolCallId;
    messages.push(entry);
  }
  const body: Json = { model, messages };
  if (canonical.tools.length > 0) {
    body.tools = canonical.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.schema } }));
  }
  if (canonical.stop.length > 0) body.stop = canonical.stop;
  if (canonical.temperature !== null) body.temperature = canonical.temperature;
  if (canonical.maxTokens !== null) body.max_tokens = canonical.maxTokens;
  if (canonical.jsonMode) body.response_format = { type: 'json_object' };
  if (canonical.stream) body.stream = true;
  return body;
}

/** An OpenAI answer body, in canonical form (the request's model names the answer's). */
export function openAIAnswerToCanonical(body: Json, request: CanonicalRequest): CanonicalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = textOf(message.content);
  const toolCalls: CanonicalAnswer['toolCalls'] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call)) continue;
      const fn = isRecord(call.function) ? call.function : {};
      toolCalls.push({ id: textOf(call.id), name: textOf(fn.name), arguments: textOf(fn.arguments) });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    model: textOf(body.model) || request.model || '',
    text,
    toolCalls,
    stopReason: textOf(choice.finish_reason) || null,
    usage: { inputTokens: numberOf(usage.prompt_tokens) ?? 0, outputTokens: numberOf(usage.completion_tokens) ?? 0 },
  };
}
