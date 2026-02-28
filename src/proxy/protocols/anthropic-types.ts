// Anthropic Messages API type definitions

// --- Request types ---

export type AnthropicContentBlockText = {
  type: "text";
  text: string;
};

export type AnthropicContentBlockImage = {
  type: "image";
  source: { type: "base64"; media_type: string; data: string };
};

export type AnthropicContentBlockToolUse = {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
};

export type AnthropicContentBlockToolResult = {
  type: "tool_result";
  tool_use_id: string;
  content?: string | AnthropicContentBlockText[];
  is_error?: boolean;
};

export type AnthropicContentBlock =
  | AnthropicContentBlockText
  | AnthropicContentBlockImage
  | AnthropicContentBlockToolUse
  | AnthropicContentBlockToolResult;

export type AnthropicMessage = {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
};

export type AnthropicToolInputSchema = {
  type: "object";
  properties?: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
};

export type AnthropicTool = {
  name: string;
  description?: string;
  input_schema: AnthropicToolInputSchema;
};

export type AnthropicToolChoice =
  | { type: "auto" }
  | { type: "any" }
  | { type: "tool"; name: string };

export type AnthropicMessagesRequest = {
  model: string;
  max_tokens: number;
  messages: AnthropicMessage[];
  system?: string | AnthropicContentBlockText[];
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  metadata?: Record<string, unknown>;
};

// --- Response types ---

export type AnthropicResponseContentBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };

export type AnthropicUsage = {
  input_tokens: number;
  output_tokens: number;
};

export type AnthropicStopReason = "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" | null;

export type AnthropicMessagesResponse = {
  id: string;
  type: "message";
  role: "assistant";
  content: AnthropicResponseContentBlock[];
  model: string;
  stop_reason: AnthropicStopReason;
  stop_sequence: string | null;
  usage: AnthropicUsage;
};

// --- Streaming SSE event types ---

export type AnthropicSseMessageStart = {
  type: "message_start";
  message: AnthropicMessagesResponse;
};

export type AnthropicSseContentBlockStart = {
  type: "content_block_start";
  index: number;
  content_block:
    | { type: "text"; text: string }
    | { type: "thinking"; thinking: string }
    | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };
};

export type AnthropicSseContentBlockDelta = {
  type: "content_block_delta";
  index: number;
  delta:
    | { type: "text_delta"; text: string }
    | { type: "thinking_delta"; thinking: string }
    | { type: "input_json_delta"; partial_json: string };
};

export type AnthropicSseContentBlockStop = {
  type: "content_block_stop";
  index: number;
};

export type AnthropicSseMessageDelta = {
  type: "message_delta";
  delta: {
    stop_reason: AnthropicStopReason;
    stop_sequence?: string | null;
  };
  usage: { output_tokens: number };
};

export type AnthropicSseMessageStop = {
  type: "message_stop";
};

export type AnthropicSsePing = {
  type: "ping";
};

export type AnthropicSseEvent =
  | AnthropicSseMessageStart
  | AnthropicSseContentBlockStart
  | AnthropicSseContentBlockDelta
  | AnthropicSseContentBlockStop
  | AnthropicSseMessageDelta
  | AnthropicSseMessageStop
  | AnthropicSsePing;

// --- Error types ---

export type AnthropicErrorResponse = {
  type: "error";
  error: {
    type: "invalid_request_error" | "authentication_error" | "overloaded_error" | "api_error" | "not_found_error";
    message: string;
  };
};
