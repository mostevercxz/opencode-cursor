import type {
  AnthropicMessagesRequest,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicTool,
  AnthropicMessagesResponse,
  AnthropicResponseContentBlock,
  AnthropicStopReason,
  AnthropicErrorResponse,
} from "./anthropic-types.js";

type OpenAiMessage = {
  role: string;
  content?: string | null;
  tool_calls?: Array<{ id: string; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
};

type OpenAiTool = {
  type: "function";
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
};

/**
 * Parse an Anthropic Messages API request body into the format
 * that buildPromptFromMessages expects (OpenAI-style messages + tools).
 */
export function parseAnthropicRequest(body: any): {
  model: string;
  messages: OpenAiMessage[];
  tools: OpenAiTool[];
  stream: boolean;
  maxTokens: number;
} {
  if (!body || typeof body !== "object") {
    throw new Error("Request body must be a JSON object");
  }
  const req = body as AnthropicMessagesRequest;
  if (typeof req.model !== "string" || req.model.trim().length === 0) {
    throw new Error("Missing required field: model");
  }
  if (typeof req.max_tokens !== "number" || !Number.isFinite(req.max_tokens) || req.max_tokens <= 0) {
    throw new Error("Missing or invalid required field: max_tokens");
  }
  if (!Array.isArray(req.messages)) {
    throw new Error("Missing required field: messages");
  }
  const model = normalizeModel(req.model);
  const stream = req.stream === true;
  const maxTokens = req.max_tokens;

  const messages: OpenAiMessage[] = [];

  // System message
  if (req.system) {
    const systemText = typeof req.system === "string"
      ? req.system
      : req.system.map((b) => b.text).join("\n");
    messages.push({ role: "system", content: systemText });
  }

  // Convert Anthropic messages to OpenAI-style
  for (const msg of req.messages ?? []) {
    convertMessage(msg, messages);
  }

  // Convert tools
  const tools: OpenAiTool[] = (req.tools ?? []).map(convertTool);

  return { model, messages, tools, stream, maxTokens };
}

function normalizeModel(model: string): string {
  const raw = (model || "").trim();
  if (!raw) return "auto";
  if (raw.startsWith("cursor-acp/")) return raw.slice("cursor-acp/".length) || "auto";
  return raw;
}

function convertMessage(msg: AnthropicMessage, out: OpenAiMessage[]): void {
  if (typeof msg.content === "string") {
    out.push({ role: msg.role, content: msg.content });
    return;
  }

  if (!Array.isArray(msg.content)) {
    out.push({ role: msg.role, content: "" });
    return;
  }

  if (msg.role === "assistant") {
    // Collect text and tool_use blocks
    const textParts: string[] = [];
    const toolCalls: Array<{ id: string; function: { name: string; arguments: string } }> = [];

    for (const block of msg.content) {
      if (block.type === "text") {
        textParts.push(block.text);
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id,
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      }
    }

    const assistantMsg: OpenAiMessage = {
      role: "assistant",
      content: textParts.length > 0 ? textParts.join("\n") : null,
    };
    if (toolCalls.length > 0) assistantMsg.tool_calls = toolCalls;
    out.push(assistantMsg);
    return;
  }

  // User messages — may contain text and tool_result blocks
  for (const block of msg.content) {
    if (block.type === "text") {
      out.push({ role: "user", content: block.text });
    } else if (block.type === "tool_result") {
      const resultContent = typeof block.content === "string"
        ? block.content
        : Array.isArray(block.content)
          ? block.content.map((b) => b.text).join("\n")
          : "";
      out.push({ role: "tool", content: resultContent, tool_call_id: block.tool_use_id });
    } else if (block.type === "image") {
      out.push({ role: "user", content: "[image omitted]" });
    }
  }
}

function convertTool(tool: AnthropicTool): OpenAiTool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  };
}

/**
 * Format a non-streaming Anthropic Messages API response.
 */
export function formatAnthropicResponse(
  model: string,
  assistantText: string,
  reasoningText: string,
  toolCalls: Array<{ id: string; name: string; args: string }>,
): AnthropicMessagesResponse {
  const content: AnthropicResponseContentBlock[] = [];

  if (reasoningText) {
    content.push({ type: "thinking", thinking: reasoningText });
  }

  if (assistantText) {
    content.push({ type: "text", text: assistantText });
  }

  let stopReason: AnthropicStopReason = "end_turn";

  for (const tc of toolCalls) {
    let input: Record<string, unknown> = {};
    try { input = JSON.parse(tc.args); } catch { /* */ }
    content.push({ type: "tool_use", id: tc.id, name: tc.name, input });
    stopReason = "tool_use";
  }

  // Ensure at least one content block
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }

  return {
    id: `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    content,
    model,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 },
  };
}

/**
 * Format an Anthropic error response.
 */
export function formatAnthropicError(
  errorType: AnthropicErrorResponse["error"]["type"],
  message: string,
): AnthropicErrorResponse {
  return {
    type: "error",
    error: { type: errorType, message },
  };
}
