import {
  extractText,
  extractThinking,
  inferToolName,
  isAssistantText,
  isThinking,
  isToolCall,
  type StreamJsonEvent,
} from "./types.js";
import { DeltaTracker } from "./delta-tracker.js";
import type { AnthropicStopReason } from "../proxy/protocols/anthropic-types.js";

function formatAnthropicSse(eventType: string, data: object): string {
  return `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
}

type BlockType = "text" | "thinking" | "tool_use" | null;

/**
 * cursor-agent partial events are usually deltas, but some backends emit
 * cumulative text (or a final full text event with timestamp_ms). This helper
 * normalizes both shapes and returns only the truly new delta.
 */
function mergePartialChunk(current: string, chunk: string): { next: string; delta: string } {
  if (!chunk) return { next: current, delta: "" };

  // cumulative stream (or repeated final full text)
  if (chunk.startsWith(current)) {
    return { next: chunk, delta: chunk.slice(current.length) };
  }

  // duplicate resend of last suffix chunk
  if (current.endsWith(chunk)) {
    return { next: current, delta: "" };
  }

  // regular delta chunk
  return { next: current + chunk, delta: chunk };
}

/**
 * Converts cursor-agent stream-json events into Anthropic Messages API SSE strings.
 *
 * Manages content block lifecycle (start/delta/stop) with proper indexing.
 * Parallel to StreamToSseConverter for OpenAI format.
 */
export class StreamToAnthropicConverter {
  private readonly model: string;
  private readonly id: string;
  private messageStarted = false;
  private currentBlockIndex = -1;
  private currentBlockType: BlockType = null;
  private sawToolCall = false;
  private readonly tracker = new DeltaTracker();
  private sawAssistantPartials = false;
  private sawThinkingPartials = false;
  private assistantPartialText = "";
  private thinkingPartialText = "";

  constructor(model: string, options?: { id?: string }) {
    this.model = model;
    this.id = options?.id ?? `msg_${Date.now()}`;
  }

  handleEvent(event: StreamJsonEvent): string[] {
    const output: string[] = [];

    // Emit message_start on first event
    if (!this.messageStarted) {
      this.messageStarted = true;
      output.push(formatAnthropicSse("message_start", {
        type: "message_start",
        message: {
          id: this.id,
          type: "message",
          role: "assistant",
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }));
    }

    if (isThinking(event)) {
      const isPartial = typeof event.timestamp_ms === "number";
      if (isPartial) {
        const text = extractThinking(event);
        if (text) {
          this.sawThinkingPartials = true;
          const merged = mergePartialChunk(this.thinkingPartialText, text);
          this.thinkingPartialText = merged.next;
          if (merged.delta) {
            output.push(...this.ensureBlock("thinking"));
            output.push(formatAnthropicSse("content_block_delta", {
              type: "content_block_delta",
              index: this.currentBlockIndex,
              delta: { type: "thinking_delta", thinking: merged.delta },
            }));
          }
        }
      } else if (!this.sawThinkingPartials) {
        const delta = this.tracker.nextThinking(extractThinking(event));
        if (delta) {
          output.push(...this.ensureBlock("thinking"));
          output.push(formatAnthropicSse("content_block_delta", {
            type: "content_block_delta",
            index: this.currentBlockIndex,
            delta: { type: "thinking_delta", thinking: delta },
          }));
        }
      }
    }

    if (isAssistantText(event)) {
      const isPartial = typeof event.timestamp_ms === "number";
      if (isPartial) {
        const text = extractText(event);
        if (text) {
          this.sawAssistantPartials = true;
          const merged = mergePartialChunk(this.assistantPartialText, text);
          this.assistantPartialText = merged.next;
          if (merged.delta) {
            output.push(...this.ensureBlock("text"));
            output.push(formatAnthropicSse("content_block_delta", {
              type: "content_block_delta",
              index: this.currentBlockIndex,
              delta: { type: "text_delta", text: merged.delta },
            }));
          }
        }
      } else if (!this.sawAssistantPartials) {
        const delta = this.tracker.nextText(extractText(event));
        if (delta) {
          output.push(...this.ensureBlock("text"));
          output.push(formatAnthropicSse("content_block_delta", {
            type: "content_block_delta",
            index: this.currentBlockIndex,
            delta: { type: "text_delta", text: delta },
          }));
        }
      }
    }

    if (isToolCall(event)) {
      this.sawToolCall = true;
      const callId = event.call_id ?? `toolu_${Date.now()}`;
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
      const argsJson = args === undefined ? "{}" : JSON.stringify(args);

      // Close current block, start tool_use block
      output.push(...this.closeCurrentBlock());
      this.currentBlockIndex++;
      this.currentBlockType = "tool_use";

      output.push(formatAnthropicSse("content_block_start", {
        type: "content_block_start",
        index: this.currentBlockIndex,
        content_block: { type: "tool_use", id: callId, name: toolName, input: {} },
      }));

      output.push(formatAnthropicSse("content_block_delta", {
        type: "content_block_delta",
        index: this.currentBlockIndex,
        delta: { type: "input_json_delta", partial_json: argsJson },
      }));

      output.push(formatAnthropicSse("content_block_stop", {
        type: "content_block_stop",
        index: this.currentBlockIndex,
      }));

      // Reset so next block can start fresh
      this.currentBlockType = null;

      return output;
    }

    return output;
  }

  /**
   * Finalize the stream — emit content_block_stop + message_delta + message_stop.
   */
  finish(): string[] {
    const output: string[] = [];

    if (!this.messageStarted) {
      // No events received at all — emit minimal message
      output.push(formatAnthropicSse("message_start", {
        type: "message_start",
        message: {
          id: this.id,
          type: "message",
          role: "assistant",
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }));
    }

    output.push(...this.closeCurrentBlock());

    const stopReason: AnthropicStopReason = this.sawToolCall ? "tool_use" : "end_turn";

    output.push(formatAnthropicSse("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: 0 },
    }));

    output.push(formatAnthropicSse("message_stop", { type: "message_stop" }));

    return output;
  }

  private ensureBlock(type: "text" | "thinking"): string[] {
    if (this.currentBlockType === type) return [];

    const output: string[] = [];
    output.push(...this.closeCurrentBlock());

    this.currentBlockIndex++;
    this.currentBlockType = type;

    if (type === "text") {
      output.push(formatAnthropicSse("content_block_start", {
        type: "content_block_start",
        index: this.currentBlockIndex,
        content_block: { type: "text", text: "" },
      }));
    } else {
      output.push(formatAnthropicSse("content_block_start", {
        type: "content_block_start",
        index: this.currentBlockIndex,
        content_block: { type: "thinking", thinking: "" },
      }));
    }

    return output;
  }

  private closeCurrentBlock(): string[] {
    if (this.currentBlockType === null || this.currentBlockIndex < 0) return [];

    const output = [formatAnthropicSse("content_block_stop", {
      type: "content_block_stop",
      index: this.currentBlockIndex,
    })];

    this.currentBlockType = null;
    return output;
  }
}
