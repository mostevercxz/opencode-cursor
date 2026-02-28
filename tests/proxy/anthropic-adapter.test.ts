import { describe, expect, it } from "bun:test";
import {
  parseAnthropicRequest,
  formatAnthropicResponse,
  formatAnthropicError,
} from "../../src/proxy/protocols/anthropic-adapter.js";

describe("parseAnthropicRequest", () => {
  it("parses simple text messages", () => {
    const result = parseAnthropicRequest({
      model: "sonnet-4.5",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hello" }],
    });

    expect(result.model).toBe("sonnet-4.5");
    expect(result.maxTokens).toBe(1024);
    expect(result.stream).toBe(false);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toEqual({ role: "user", content: "Hello" });
  });

  it("strips cursor-acp/ prefix from model", () => {
    const result = parseAnthropicRequest({
      model: "cursor-acp/opus-4.6",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hi" }],
    });
    expect(result.model).toBe("opus-4.6");
  });

  it("handles system string", () => {
    const result = parseAnthropicRequest({
      model: "auto",
      max_tokens: 1024,
      system: "You are helpful",
      messages: [{ role: "user", content: "Hi" }],
    });

    expect(result.messages[0]).toEqual({ role: "system", content: "You are helpful" });
    expect(result.messages[1]).toEqual({ role: "user", content: "Hi" });
  });

  it("handles system content blocks", () => {
    const result = parseAnthropicRequest({
      model: "auto",
      max_tokens: 1024,
      system: [{ type: "text", text: "Part 1" }, { type: "text", text: "Part 2" }],
      messages: [{ role: "user", content: "Hi" }],
    });

    expect(result.messages[0]).toEqual({ role: "system", content: "Part 1\nPart 2" });
  });

  it("converts assistant messages with tool_use blocks", () => {
    const result = parseAnthropicRequest({
      model: "auto",
      max_tokens: 1024,
      messages: [
        { role: "user", content: "Run ls" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Running command" },
            { type: "tool_use", id: "toolu_1", name: "bash", input: { command: "ls" } },
          ],
        },
      ],
    });

    const assistantMsg = result.messages[1];
    expect(assistantMsg.role).toBe("assistant");
    expect(assistantMsg.content).toBe("Running command");
    expect(assistantMsg.tool_calls).toHaveLength(1);
    expect(assistantMsg.tool_calls![0].id).toBe("toolu_1");
    expect(assistantMsg.tool_calls![0].function.name).toBe("bash");
  });

  it("converts tool_result blocks to tool role messages", () => {
    const result = parseAnthropicRequest({
      model: "auto",
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_1", content: "file.txt" },
          ],
        },
      ],
    });

    expect(result.messages[0].role).toBe("tool");
    expect(result.messages[0].content).toBe("file.txt");
    expect(result.messages[0].tool_call_id).toBe("toolu_1");
  });

  it("converts Anthropic tools to OpenAI format", () => {
    const result = parseAnthropicRequest({
      model: "auto",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hi" }],
      tools: [{
        name: "bash",
        description: "Run command",
        input_schema: {
          type: "object",
          properties: { command: { type: "string" } },
          required: ["command"],
        },
      }],
    });

    expect(result.tools).toHaveLength(1);
    expect(result.tools[0].type).toBe("function");
    expect(result.tools[0].function.name).toBe("bash");
    expect(result.tools[0].function.parameters).toEqual({
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    });
  });
});

describe("formatAnthropicResponse", () => {
  it("formats text-only response", () => {
    const resp = formatAnthropicResponse("test-model", "Hello world", "", []);

    expect(resp.type).toBe("message");
    expect(resp.role).toBe("assistant");
    expect(resp.model).toBe("test-model");
    expect(resp.content).toHaveLength(1);
    expect(resp.content[0]).toEqual({ type: "text", text: "Hello world" });
    expect(resp.stop_reason).toBe("end_turn");
  });

  it("includes thinking block", () => {
    const resp = formatAnthropicResponse("test-model", "Answer", "Reasoning here", []);

    expect(resp.content).toHaveLength(2);
    expect(resp.content[0]).toEqual({ type: "thinking", thinking: "Reasoning here" });
    expect(resp.content[1]).toEqual({ type: "text", text: "Answer" });
  });

  it("includes tool_use blocks", () => {
    const resp = formatAnthropicResponse("test-model", "Let me check", "", [
      { id: "toolu_1", name: "bash", args: '{"command":"ls"}' },
    ]);

    expect(resp.content).toHaveLength(2);
    expect(resp.content[0]).toEqual({ type: "text", text: "Let me check" });
    expect(resp.content[1]).toEqual({
      type: "tool_use",
      id: "toolu_1",
      name: "bash",
      input: { command: "ls" },
    });
    expect(resp.stop_reason).toBe("tool_use");
  });

  it("ensures at least one content block for empty response", () => {
    const resp = formatAnthropicResponse("test-model", "", "", []);
    expect(resp.content).toHaveLength(1);
    expect(resp.content[0]).toEqual({ type: "text", text: "" });
  });
});

describe("formatAnthropicError", () => {
  it("formats error response", () => {
    const err = formatAnthropicError("invalid_request_error", "Bad request");
    expect(err.type).toBe("error");
    expect(err.error.type).toBe("invalid_request_error");
    expect(err.error.message).toBe("Bad request");
  });
});
