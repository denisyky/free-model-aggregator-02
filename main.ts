// free-model-aggregator-02 — "LIVE ROUTER": one server, two client wires,
// one real free OpenRouter model behind every answer.
//
// What a trial gives this server:
//   A trial runs `node /bot/main.ts` from the repository root, with Node 22's
//   standard library alone (nothing installed). The env is at
//   $MAKEFUN_ENV_URL; model calls go through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on
//   the key pushed to this competition. The brief at $MAKEFUN_PROMPT_FILE
//   carries two documents: the free model list (gateway id, published rating
//   in milli-points, tools/vision/JSON capabilities, context length) and the
//   six prompts (id, api, stream, weight, requirements).
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//        "catalog": <catalog>};
//     3. serves GET /v1/models (shape by auth header: x-api-key is Anthropic,
//        Authorization is OpenAI), POST /v1/chat/completions and
//        POST /v1/messages, JSON or server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway
//        (OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//        entrant's own pushed key; the real key never enters the sandbox);
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//        says {"done": true}.
//   `node main.ts --list` prints the same catalog JSON and exits 0.
//
// ROUTING (quality is 60% of the score): the requirements are read off each
// arriving request body (JSON mode from response_format, tools from a
// non-empty tools list, vision from image content) — never from prompt order,
// so the final trial's different manifest still routes right. Among the free
// models that meet them, the highest rating_milli serves; ties keep the
// brief's order (a stable sort, so replay is byte-identical).
//
// FAILOVER (10%): a 429/5xx or a broken call is retried on the next eligible
// model, up to 4 attempts a prompt. Every refusal is reported on the answer
// in `x-makefun-upstream-failures`, a JSON array of at most 8 rows
// {"status","code","model"} in refusal order — without it, a recovered
// refusal is invisible to the failover column. A refused model cools down
// for the next 2 prompts (counted in prompts, never in wall-clock time, so
// a replay makes the same choices); a model already tried for this prompt
// is never tried twice for it. At most 60 gateway calls a trial.
//
// THE MODEL FIELD: the house records every gateway call and scores the model
// that really answered, so every answer and every streamed chunk carries the
// id this entry dialled. The provider's own answer spells it without the
// gateway's `openrouter/` prefix; this server puts the dialled id back.
//
// DETERMINISM: every route is a pure function of the requests and the
// gateway answers seen so far. No Date.now(), no Math.random() — the model
// listings use fixed timestamps, and MAKEFUN_TRIAL_SEED is not needed
// because nothing here is random.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway attempt's deadline, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 30000;
/** The most gateway attempts one prompt costs. */
const MAX_ATTEMPTS = 4;
/** How many later prompts a refused model sits out (prompt-counted, not timed). */
const COOLDOWN_PROMPTS = 2;
/** The trial's ceiling on gateway calls. */
const MAX_CALLS = 60;
/** The most refusal rows the failover header carries. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
/** The public base URL of the provider the free models belong to (never the gateway's trial-local address). */
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under. */
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** One model in the catalog document. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
interface CatalogReport {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: CatalogModel[];
  }[];
}

/** One model the entry serves, as GET /v1/models lists it. */
interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One refused gateway call, as the failover header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/**
 * The sealed free-model list, exactly as the brief spells it, as a fallback
 * when MAKEFUN_PROMPT_FILE is missing or unreadable: `--list` and the
 * routing still print and serve the same fourteen ids. The brief file always
 * wins when it reads — this copy only keeps a missing brief from emptying
 * the catalog.
 */
const FALLBACK_MODELS: ModelRow[] = [
  { id: 'openrouter/baidu/ernie-5.5-flash-preview:free', provider: 'openrouter', display_name: 'Baidu: ERNIE-5.5-Flash-Preview (free)', rating_milli: 1150, context_length: 1024000, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/cohere/north-mini-code:free', provider: 'openrouter', display_name: 'Cohere: North Mini Code (free)', rating_milli: 1200, context_length: 256000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/dots-studio/dots-3-note-preview:free', provider: 'openrouter', display_name: 'Dots Studio: Dots3-Note Preview (free)', rating_milli: 1150, context_length: 512000, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/google/gemma-4-26b-a4b-it:free', provider: 'openrouter', display_name: 'Google: Gemma 4 26B A4B (free)', rating_milli: 1150, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/google/gemma-4-31b-it:free', provider: 'openrouter', display_name: 'Google: Gemma 4 31B (free)', rating_milli: 1210, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/inclusionai/ling-3.2-flash:free', provider: 'openrouter', display_name: 'inclusionAI: Ling 3.2 Flash (free)', rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/liquid/lfm-3-1.7b:free', provider: 'openrouter', display_name: 'LiquidAI: LFM3-1.7B (free)', rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3 Nano Omni (free)', rating_milli: 1150, context_length: 256000, tools: true, vision: true, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3 Ultra (free)', rating_milli: 1285, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3.5-lightning:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3.5 Lightning (free)', rating_milli: 1221, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3.5-super-vl:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3.5 Super VL (free)', rating_milli: 1303, context_length: 1000000, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/particle-ai/particle-1:free', provider: 'openrouter', display_name: 'Particle AI: Particle-1 (free)', rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/recursive/flint-v1.0:free', provider: 'openrouter', display_name: 'Recursive: Flint (free)', rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/thinkingmachines/inkling-2-small:free', provider: 'openrouter', display_name: 'Thinking Machines: Inkling 2 Small (free)', rating_milli: 1475, context_length: 1048576, tools: true, vision: true, json: false, free: true },
];

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/* ------------------------------------------------------------------ brief -- */

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** The first document under this contest's prefix that carries one of `keys` as an array. */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes('free-model-aggregator-02/')) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`: its model list and its prompt list, or the built-in list when it cannot be read. */
function readBrief(path: string): { models: ModelRow[]; prompts: PromptRow[]; error: string | null } {
  if (path === '') return { models: [...FALLBACK_MODELS], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set (using the built-in model list)' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [...FALLBACK_MODELS], prompts: [], error: `cannot read ${path}: ${(err as Error).message} (using the built-in model list)` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [...FALLBACK_MODELS], prompts: [], error: `the brief at ${path} holds no free-model-aggregator-02/models/v1 document (using the built-in model list)` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  const prompts = promptsDoc === null ? [] : (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

const brief = readBrief(PROMPT_FILE);
/** The free models the brief lists, in the brief's order. */
const FREE_MODELS: ModelRow[] = brief.models.filter((m) => m.free);

/** The catalog the free model list makes: one provider, the models it serves. */
function buildCatalog(): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of FREE_MODELS) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

const catalog: CatalogReport = buildCatalog();

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return FREE_MODELS.map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/* -------------------------------------------------------------- routing -- */

/** What a model can do, as the brief's row says it. */
function capabilitiesOf(model: ModelRow): { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return { context_length: model.context_length, tools: model.tools, vision: model.vision, json: model.json };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The requirements of one arriving request, read off its own body: JSON mode
 * from response_format (OpenAI only), tools from a non-empty tools list,
 * vision from image content. Never from prompt order, so a final trial whose
 * manifest differs still routes right.
 */
function requirementsFromBody(body: Json, wire: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = wire === 'openai' && isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** How many later prompts each refused model still sits out (prompt-counted, never timed). */
const cooldowns = new Map<string, number>();
/** Gateway calls made this trial (the trial's ceiling is 60). */
let totalCalls = 0;

/** One prompt's failover chain, best first: eligible models by rating, ties in brief order, skipping cooled-down and already-tried models. */
function candidatesFor(requires: Requirement, tried: ReadonlySet<string>): ModelRow[] {
  return FREE_MODELS.map((model, index) => ({ model, index }))
    .filter(({ model }) => meetsRequirements(requires, capabilitiesOf(model)))
    .filter(({ model }) => !tried.has(model.id) && (cooldowns.get(model.id) ?? 0) <= 0)
    .sort((a, b) => (b.model.rating_milli - a.model.rating_milli) || (a.index - b.index))
    .map(({ model }) => model);
}

/** Age every cooldown by one arriving prompt (called once per prompt, before routing). */
function ageCooldowns(): void {
  for (const [id, left] of [...cooldowns.entries()]) {
    if (left <= 1) cooldowns.delete(id);
    else cooldowns.set(id, left - 1);
  }
}

/* ----------------------------------------------------------------- wire -- */

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
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
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them; a frame cut across reads is held
 * until its end arrives, and a body that ends without one is relayed as it is.
 */
async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield frameWithModel(frame, model);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithModel(buffer, model);
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
      // OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
      // output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* -------------------------------------------------------------- gateway -- */

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The error code a refused gateway body names, or a fallback from its status. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      // OpenRouter names some refusals with a numeric code (e.g. 429): keep it as a string.
      if (typeof parsed.error.code === 'number') return String(parsed.error.code);
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
  } catch {
    // Not JSON: fall back to the status.
  }
  return `http_${status}`;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  const upstream = { ...body, model };
  if (typeof upstream.max_tokens !== 'number') upstream.max_tokens = 1024;
  else if (upstream.max_tokens > 8192) upstream.max_tokens = 8192;
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(upstream),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

/**
 * Dial the failover chain for one prompt: the best eligible model first, then
 * the next, until one answers or the chain runs out. Returns the answering
 * model and response, or null when every attempt was refused. Refusals are
 * recorded in `failures` (in order) and each refused model cools down.
 */
async function dialChain(requires: Requirement, upstreamBody: Json): Promise<{ model: string; response: Response; failures: FailureRow[] } | null> {
  const failures: FailureRow[] = [];
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    if (totalCalls >= MAX_CALLS) break;
    const next = candidatesFor(requires, tried)[0];
    if (next === undefined) break;
    tried.add(next.id);
    totalCalls += 1;
    let upstream: Response;
    try {
      upstream = await dial(next.id, upstreamBody);
    } catch (err) {
      failures.push({ status: 0, code: 'network_error', model: next.id });
      cooldowns.set(next.id, COOLDOWN_PROMPTS);
      log(`gateway unreachable for ${next.id}: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: next.id, response: upstream, failures };
    const text = await upstream.text();
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: next.id });
    cooldowns.set(next.id, COOLDOWN_PROMPTS);
    log(`gateway refused ${next.id} with HTTP ${upstream.status}`);
  }
  return failures.length > 0 || totalCalls >= MAX_CALLS ? { model: '', response: null as unknown as Response, failures } : null;
}

/** The common tail of both completion routes: relay a successful upstream answer, or a 502 after the chain ran out. */
async function relay(
  req: IncomingMessage,
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  body: Json,
  outcome: { model: string; response: Response; failures: FailureRow[] } | null,
): Promise<void> {
  if (outcome === null || outcome.model === '') {
    const last = outcome?.failures[outcome.failures.length - 1];
    const status = last !== undefined && last.status > 0 ? last.status : 502;
    const detail = outcome !== null && outcome.failures.length > 0 ? `after ${outcome.failures.length} refused upstream call(s)` : 'no free model could serve this request';
    return sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', `the gateway answered HTTP ${status} ${detail}`, outcome?.failures ?? []);
  }
  const { model, response, failures } = outcome;
  if (body.stream === true) {
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
    res.writeHead(200, headers);
    try {
      if (wire === 'anthropic') {
        for await (const event of anthropicEventsFromOpenAI(response.body as ReadableStream<Uint8Array>, model)) res.write(event);
      } else {
        for await (const frame of openAIStreamWithModel(response.body as ReadableStream<Uint8Array>, model)) res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await response.json();
  if (wire === 'anthropic') sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), failures);
  else sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), failures);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  ageCooldowns();
  const requires = requirementsFromBody(body, 'openai');
  const outcome = await dialChain(requires, { ...body, stream: body.stream === true });
  await relay(req, res, 'openai', body, outcome);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  ageCooldowns();
  const requires = requirementsFromBody(body, 'anthropic');
  // One chain only: dialChainTranslated translates per candidate inside the loop, so no
  // attempt is ever spent on a placeholder model.
  await relay(req, res, 'anthropic', body, await dialChainTranslated(body, requires));
}

/** The Anthropic chain: translate once, then dial each candidate with its own id. */
async function dialChainTranslated(body: Json, requires: Requirement): Promise<{ model: string; response: Response; failures: FailureRow[] } | null> {
  const failures: FailureRow[] = [];
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    if (totalCalls >= MAX_CALLS) break;
    const next = candidatesFor(requires, tried)[0];
    if (next === undefined) break;
    tried.add(next.id);
    totalCalls += 1;
    let upstream: Response;
    try {
      upstream = await dial(next.id, anthropicToOpenAI(body, next.id));
    } catch (err) {
      failures.push({ status: 0, code: 'network_error', model: next.id });
      cooldowns.set(next.id, COOLDOWN_PROMPTS);
      log(`gateway unreachable for ${next.id}: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: next.id, response: upstream, failures };
    const text = await upstream.text();
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: next.id });
    cooldowns.set(next.id, COOLDOWN_PROMPTS);
    log(`gateway refused ${next.id} with HTTP ${upstream.status}`);
  }
  return failures.length > 0 || totalCalls >= MAX_CALLS ? { model: '', response: null as unknown as Response, failures } : null;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
