import { buildPromptFromMessages } from "./prompt-builder.js";
import { findAvailablePort } from "./server.js";
import { LineBuffer } from "../streaming/line-buffer.js";
import { StreamToSseConverter, formatSseDone } from "../streaming/openai-sse.js";
import { StreamToAnthropicConverter } from "../streaming/anthropic-sse.js";
import { parseStreamJsonLine } from "../streaming/parser.js";
import {
  extractText,
  extractThinking,
  inferToolName,
  isAssistantText,
  isThinking,
  isToolCall,
  type StreamJsonEvent,
  type StreamJsonToolCallEvent,
} from "../streaming/types.js";
import { createLogger } from "../utils/logger.js";
import { parseAgentError, formatErrorForUser, stripAnsi } from "../utils/errors.js";
import { parseAnthropicRequest, formatAnthropicResponse, formatAnthropicError } from "./protocols/anthropic-adapter.js";

const log = createLogger("proxy:daemon");

export type StandaloneProxyOptions = {
  host?: string;
  port?: number;
  logLevel?: string;
  idleTimeoutSeconds?: number;
  sseHeartbeatMs?: number;
};

export type StandaloneProxyInstance = {
  host: string;
  port: number;
  url: string;
  baseURL: string;
  stop: () => Promise<void>;
};

type OpenAiToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 32124;
const DEFAULT_IDLE_TIMEOUT_SECONDS = 120;
const DEFAULT_SSE_HEARTBEAT_MS = 5000;

function mergePartialChunk(current: string, chunk: string): { next: string; delta: string } {
  if (!chunk) return { next: current, delta: "" };
  if (chunk.startsWith(current)) {
    return { next: chunk, delta: chunk.slice(current.length) };
  }
  if (current.endsWith(chunk)) {
    return { next: current, delta: "" };
  }
  return { next: current + chunk, delta: chunk };
}

function parseNonNegativeInt(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.floor(value);
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed) && parsed >= 0) return parsed;
  }
  return fallback;
}

function getSseHeartbeatMs(): number {
  return parseNonNegativeInt(process.env.PROXY_SSE_HEARTBEAT_MS, DEFAULT_SSE_HEARTBEAT_MS);
}

function printRequestMetrics(path: string, model: string, requestedTokens: number | null): void {
  const timestamp = new Date().toISOString();
  const tokenText = requestedTokens === null ? "n/a" : String(requestedTokens);
  console.log(`[${timestamp}] path=${path} requested_tokens=${tokenText} model=${model}`);
}

function normalizeModel(model: unknown): string {
  const raw = typeof model === "string" ? model.trim() : "";
  if (raw.length === 0) return "auto";
  const prefix = "cursor-acp/";
  if (raw.startsWith(prefix)) {
    const stripped = raw.slice(prefix.length).trim();
    return stripped.length > 0 ? stripped : "auto";
  }
  return raw;
}

function createChatCompletionResponse(model: string, content: string, reasoningContent?: string) {
  const message: { role: "assistant"; content: string; reasoning_content?: string } = {
    role: "assistant",
    content,
  };
  if (reasoningContent && reasoningContent.length > 0) {
    message.reasoning_content = reasoningContent;
  }
  return {
    id: `cursor-acp-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message, finish_reason: "stop" }],
  };
}

function createToolCallResponse(meta: { id: string; created: number; model: string }, toolCall: OpenAiToolCall) {
  return {
    id: meta.id,
    object: "chat.completion",
    created: meta.created,
    model: meta.model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: null, tool_calls: [toolCall] },
      finish_reason: "tool_calls",
    }],
  };
}

function createToolCallStreamChunks(meta: { id: string; created: number; model: string }, toolCall: OpenAiToolCall) {
  return [
    {
      id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model,
      choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{ index: 0, ...toolCall }] }, finish_reason: null }],
    },
    {
      id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model,
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    },
  ];
}

function createChunk(id: string, created: number, model: string, content: string, done = false) {
  return {
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta: content ? { content } : {}, finish_reason: done ? "stop" : null }],
  };
}

function toolCallFromEvent(event: StreamJsonToolCallEvent): OpenAiToolCall {
  const callId = event.call_id || `call_${Date.now()}`;
  const toolName = inferToolName(event) || "tool";
  const toolKey = Object.keys(event.tool_call ?? {})[0];
  const payload = toolKey ? event.tool_call[toolKey] : undefined;
  let args: unknown = {};
  if (payload && typeof payload === "object" && "args" in payload) {
    args = (payload as any).args;
  } else if (payload && typeof payload === "object") {
    const { result: _r, ...rest } = payload as any;
    if (Object.keys(rest).length > 0) args = rest;
  }
  return {
    id: callId,
    type: "function",
    function: { name: toolName, arguments: args === undefined ? "{}" : JSON.stringify(args) },
  };
}

function extractFirstToolCall(output: string): OpenAiToolCall | null {
  for (const line of output.split("\n")) {
    const event = parseStreamJsonLine(line);
    if (event && isToolCall(event)) return toolCallFromEvent(event);
  }
  return null;
}

function extractCompletion(output: string): { assistantText: string; reasoningText: string } {
  let assistantText = "";
  let reasoningText = "";
  let sawPartials = false;
  let sawThinkingPartials = false;
  for (const line of output.split("\n")) {
    const event = parseStreamJsonLine(line);
    if (!event) continue;
    if (isAssistantText(event)) {
      const text = extractText(event);
      if (!text) continue;
      if (typeof (event as any).timestamp_ms === "number") {
        const merged = mergePartialChunk(assistantText, text);
        assistantText = merged.next;
        sawPartials = true;
      } else if (!sawPartials) {
        assistantText = text;
      }
    }
    if (isThinking(event)) {
      const t = extractThinking(event as any);
      if (!t) continue;
      if (typeof (event as any).timestamp_ms === "number") {
        const merged = mergePartialChunk(reasoningText, t);
        reasoningText = merged.next;
        sawThinkingPartials = true;
      } else if (!sawThinkingPartials) {
        reasoningText = t;
      }
    }
  }
  return { assistantText, reasoningText };
}

function spawnCursorAgent(model: string, prompt: string): any {
  const bunAny = globalThis as any;
  const workspace = process.env.PROXY_WORKSPACE || process.cwd();
  const child = bunAny.Bun.spawn({
    cmd: [
      "cursor-agent", "--print", "--output-format", "stream-json",
      "--stream-partial-output", "--workspace", workspace, "--model", model, "--force",
    ],
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: bunAny.Bun.env,
  });
  try {
    child.stdin.write(prompt);
    child.stdin.end();
  } catch {
    try { child.kill(); } catch { /* */ }
    throw new Error("Failed to write prompt to cursor-agent stdin");
  }
  return child;
}

async function handleModels(): Promise<Response> {
  const bunAny = globalThis as any;
  const proc = bunAny.Bun.spawn({ cmd: ["cursor-agent", "models"], stdout: "pipe", stderr: "pipe", env: bunAny.Bun.env });
  const [stdoutText, stderrText] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const detail = stripAnsi(stderrText || stdoutText || "").trim() || `exit code ${exitCode}`;
    return Response.json({ error: "Failed to fetch models", detail }, { status: 500 });
  }
  const output = stripAnsi(stdoutText || "");
  const models: Array<{ id: string; object: string; created: number; owned_by: string }> = [];
  const created = Math.floor(Date.now() / 1000);
  for (const line of output.split("\n")) {
    const match = line.match(/^([a-z0-9.-]+)\s+-\s+(.+?)(?:\s+\((current|default)\))*\s*$/i);
    if (match) models.push({ id: match[1], object: "model", created, owned_by: "cursor" });
  }
  return Response.json({ object: "list", data: models });
}

async function handleChatCompletions(req: Request): Promise<Response> {
  const body: any = await req.json().catch(() => ({}));
  const messages: any[] = Array.isArray(body?.messages) ? body.messages : [];
  const tools: any[] = Array.isArray(body?.tools) ? body.tools : [];
  const stream = body?.stream === true;
  const model = normalizeModel(body?.model);
  const requestedTokens = typeof body?.max_tokens === "number"
    ? body.max_tokens
    : typeof body?.max_completion_tokens === "number"
      ? body.max_completion_tokens
      : null;
  printRequestMetrics("/v1/chat/completions", model, requestedTokens);
  const prompt = buildPromptFromMessages(messages, tools);

  log.debug("chat request", { stream, model, messages: messages.length, tools: tools.length });

  const child = spawnCursorAgent(model, prompt);

  if (!stream) {
    const [stdoutText, stderrText] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ]);
    const exitCode = await child.exited;
    const stdout = (stdoutText || "").trim();
    const stderr = (stderrText || "").trim();
    const meta = { id: `cursor-acp-${Date.now()}`, created: Math.floor(Date.now() / 1000), model };

    const toolCall = extractFirstToolCall(stdout);
    if (toolCall) return Response.json(createToolCallResponse(meta, toolCall));

    if (exitCode !== 0) {
      const parsed = parseAgentError(stderr || stdout || `cursor-agent exited with code ${exitCode}`);
      return Response.json(createChatCompletionResponse(model, formatErrorForUser(parsed)));
    }

    const c = extractCompletion(stdout);
    return Response.json(createChatCompletionResponse(model, c.assistantText || stdout || stderr, c.reasoningText || undefined));
  }

  // Streaming
  const encoder = new TextEncoder();
  const id = `cursor-acp-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);
  const meta = { id, created, model };

  const sse = new ReadableStream<Uint8Array>({
    async start(controller) {
      let terminated = false;
      const emit = (data: string) => { if (!terminated) controller.enqueue(encoder.encode(data)); };
      const terminate = () => { terminated = true; try { child.kill(); } catch { /* */ } };
      const heartbeatMs = getSseHeartbeatMs();
      const heartbeatTimer = heartbeatMs > 0
        ? setInterval(() => emit(": keep-alive\n\n"), heartbeatMs)
        : null;

      try {
        const reader = (child.stdout as ReadableStream<Uint8Array>).getReader();
        const converter = new StreamToSseConverter(model, { id, created });
        const lineBuffer = new LineBuffer();

        const processLine = (line: string) => {
          if (terminated) return;
          const event = parseStreamJsonLine(line);
          if (!event) return;
          if (isToolCall(event)) {
            for (const chunk of createToolCallStreamChunks(meta, toolCallFromEvent(event))) {
              emit(`data: ${JSON.stringify(chunk)}\n\n`);
            }
            emit(formatSseDone());
            terminate();
            return;
          }
          for (const s of converter.handleEvent(event as StreamJsonEvent)) emit(s);
        };

        while (!terminated) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value || value.length === 0) continue;
          for (const line of lineBuffer.push(value)) processLine(line);
        }
        if (!terminated) for (const line of lineBuffer.flush()) processLine(line);

        if (!terminated) {
          const exitCode = await child.exited;
          if (exitCode !== 0) {
            const stderrText = await new Response(child.stderr).text();
            const parsed = parseAgentError(stripAnsi(stderrText || "").trim() || `exit code ${exitCode}`);
            emit(`data: ${JSON.stringify(createChunk(id, created, model, formatErrorForUser(parsed), true))}\n\n`);
          } else {
            emit(`data: ${JSON.stringify(createChunk(id, created, model, "", true))}\n\n`);
          }
          emit(formatSseDone());
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        emit(`data: ${JSON.stringify(createChunk(id, created, model, msg, true))}\n\n`);
        emit(formatSseDone());
      } finally {
        if (heartbeatTimer) clearInterval(heartbeatTimer);
        controller.close();
      }
    },
    cancel() { try { child.kill(); } catch { /* */ } },
  });

  return new Response(sse, {
    status: 200,
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" },
  });
}

function extractToolCalls(output: string): Array<{ id: string; name: string; args: string }> {
  const calls: Array<{ id: string; name: string; args: string }> = [];
  for (const line of output.split("\n")) {
    const event = parseStreamJsonLine(line);
    if (event && isToolCall(event)) {
      const tc = toolCallFromEvent(event);
      calls.push({ id: tc.id, name: tc.function.name, args: tc.function.arguments });
    }
  }
  return calls;
}

async function handleAnthropicMessages(req: Request): Promise<Response> {
  const body: any = await req.json().catch(() => null);
  if (!body) {
    return Response.json(formatAnthropicError("invalid_request_error", "Invalid JSON body"), { status: 400 });
  }

  let parsed: ReturnType<typeof parseAnthropicRequest>;
  try {
    parsed = parseAnthropicRequest(body);
  } catch (err) {
    return Response.json(
      formatAnthropicError("invalid_request_error", err instanceof Error ? err.message : String(err)),
      { status: 400 },
    );
  }

  const { model, messages, tools, stream, maxTokens } = parsed;
  printRequestMetrics("/v1/messages", model, maxTokens);
  const prompt = buildPromptFromMessages(messages, tools);

  log.debug("anthropic request", { stream, model, messages: messages.length, tools: tools.length });

  const child = spawnCursorAgent(model, prompt);

  if (!stream) {
    const [stdoutText, stderrText] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ]);
    const exitCode = await child.exited;
    const stdout = (stdoutText || "").trim();
    const stderr = (stderrText || "").trim();

    if (exitCode !== 0) {
      const agentErr = parseAgentError(stderr || stdout || `cursor-agent exited with code ${exitCode}`);
      return Response.json(
        formatAnthropicError("api_error", formatErrorForUser(agentErr)),
        { status: 500 },
      );
    }

    const completion = extractCompletion(stdout);
    const toolCalls = extractToolCalls(stdout);
    return Response.json(
      formatAnthropicResponse(model, completion.assistantText || stdout, completion.reasoningText, toolCalls),
    );
  }

  // Streaming
  const encoder = new TextEncoder();
  const id = `msg_${Date.now()}`;

  const sse = new ReadableStream<Uint8Array>({
    async start(controller) {
      let terminated = false;
      const emit = (data: string) => { if (!terminated) controller.enqueue(encoder.encode(data)); };
      const terminate = () => { terminated = true; try { child.kill(); } catch { /* */ } };
      const heartbeatMs = getSseHeartbeatMs();
      const heartbeatTimer = heartbeatMs > 0
        ? setInterval(() => emit(": keep-alive\n\n"), heartbeatMs)
        : null;

      try {
        const reader = (child.stdout as ReadableStream<Uint8Array>).getReader();
        const converter = new StreamToAnthropicConverter(model, { id });
        const lineBuffer = new LineBuffer();

        const processLine = (line: string) => {
          if (terminated) return;
          const event = parseStreamJsonLine(line);
          if (!event) return;
          for (const s of converter.handleEvent(event as StreamJsonEvent)) emit(s);
        };

        while (!terminated) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value || value.length === 0) continue;
          for (const line of lineBuffer.push(value)) processLine(line);
        }
        if (!terminated) for (const line of lineBuffer.flush()) processLine(line);

        if (!terminated) {
          const exitCode = await child.exited;
          if (exitCode !== 0) {
            const stderrText = await new Response(child.stderr).text();
            const agentErr = parseAgentError(stripAnsi(stderrText || "").trim() || `exit code ${exitCode}`);
            // Emit error as text content block before finishing
            for (const s of converter.handleEvent({
              type: "assistant",
              message: { role: "assistant", content: [{ type: "text", text: formatErrorForUser(agentErr) }] },
            })) emit(s);
          }
          for (const s of converter.finish()) emit(s);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        // Best effort: emit error in Anthropic SSE format
        const errConverter = new StreamToAnthropicConverter(model, { id });
        for (const s of errConverter.handleEvent({
          type: "assistant",
          message: { role: "assistant", content: [{ type: "text", text: msg }] },
        })) emit(s);
        for (const s of errConverter.finish()) emit(s);
      } finally {
        if (heartbeatTimer) clearInterval(heartbeatTimer);
        controller.close();
      }
    },
    cancel() { try { child.kill(); } catch { /* */ } },
  });

  return new Response(sse, {
    status: 200,
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" },
  });
}

export async function startStandaloneProxy(opts: StandaloneProxyOptions = {}): Promise<StandaloneProxyInstance> {
  const bunAny = globalThis as any;
  if (!bunAny.Bun?.serve || !bunAny.Bun?.spawn) {
    throw new Error("Standalone proxy requires Bun runtime.");
  }

  if (opts.logLevel) process.env.CURSOR_ACP_LOG_LEVEL = opts.logLevel;
  if (typeof opts.sseHeartbeatMs === "number" && Number.isFinite(opts.sseHeartbeatMs) && opts.sseHeartbeatMs >= 0) {
    process.env.PROXY_SSE_HEARTBEAT_MS = String(Math.floor(opts.sseHeartbeatMs));
  }

  const host = opts.host ?? DEFAULT_HOST;
  const requestedPort = typeof opts.port === "number" && Number.isFinite(opts.port) ? opts.port : DEFAULT_PORT;
  const idleTimeoutSeconds = parseNonNegativeInt(
    opts.idleTimeoutSeconds ?? process.env.PROXY_IDLE_TIMEOUT_SEC,
    DEFAULT_IDLE_TIMEOUT_SECONDS,
  );

  const handler = async (req: Request): Promise<Response> => {
    const url = new URL(req.url);
    const path = url.pathname;
    try {
      if (path === "/health" && req.method === "GET") return Response.json({ ok: true });
      if ((path === "/v1/models" || path === "/models") && req.method === "GET") return await handleModels();
      if (path === "/v1/chat/completions" || path === "/chat/completions") {
        if (req.method !== "POST") return Response.json({ error: "Method not allowed" }, { status: 405 });
        return await handleChatCompletions(req);
      }
      if (path === "/v1/messages" || path === "/messages") {
        if (req.method !== "POST") return Response.json(formatAnthropicError("invalid_request_error", "Method not allowed"), { status: 405 });
        return await handleAnthropicMessages(req);
      }
      return Response.json({ error: `Unsupported path: ${path}` }, { status: 404 });
    } catch (err) {
      return Response.json({ error: err instanceof Error ? err.message : String(err) }, { status: 500 });
    }
  };

  let server: any;
  let port = requestedPort;
  try {
    server = bunAny.Bun.serve({ hostname: host, port, fetch: handler, idleTimeout: idleTimeoutSeconds });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!msg.includes("EADDRINUSE") && !msg.toLowerCase().includes("address already in use")) throw err;
    port = await findAvailablePort(host);
    log.warn("port in use, falling back", { requested: requestedPort, fallback: port });
    server = bunAny.Bun.serve({ hostname: host, port, fetch: handler, idleTimeout: idleTimeoutSeconds });
  }

  const actualPort: number = server.port ?? port;
  const url = `http://${host}:${actualPort}`;
  const baseURL = `${url}/v1`;
  log.info("standalone proxy started", { host, port: actualPort, baseURL, idleTimeoutSeconds, sseHeartbeatMs: getSseHeartbeatMs() });

  return {
    host,
    port: actualPort,
    url,
    baseURL,
    stop: async () => { try { server.stop(true); } catch { /* */ } },
  };
}
