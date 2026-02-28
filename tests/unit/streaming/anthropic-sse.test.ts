import { describe, expect, it } from "bun:test";
import { StreamToAnthropicConverter } from "../../../src/streaming/anthropic-sse.js";

function parseSseEvents(chunks: string[]): Array<{ event: string; data: any }> {
  const events: Array<{ event: string; data: any }> = [];
  for (const chunk of chunks) {
    const lines = chunk.trim().split("\n");
    let eventType = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) eventType = line.slice(7);
      if (line.startsWith("data: ")) {
        events.push({ event: eventType, data: JSON.parse(line.slice(6)) });
      }
    }
  }
  return events;
}

describe("StreamToAnthropicConverter", () => {
  it("emits correct sequence for text-only stream", () => {
    const converter = new StreamToAnthropicConverter("test-model", { id: "msg_test" });

    const chunks1 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: 1,
      message: { role: "assistant", content: [{ type: "text", text: "Hello" }] },
    });

    const chunks2 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: 2,
      message: { role: "assistant", content: [{ type: "text", text: " world" }] },
    });

    const finish = converter.finish();

    const events = parseSseEvents([...chunks1, ...chunks2, ...finish]);
    const types = events.map((e) => e.event);

    expect(types).toEqual([
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ]);

    // Check message_start
    expect(events[0].data.type).toBe("message_start");
    expect(events[0].data.message.id).toBe("msg_test");
    expect(events[0].data.message.role).toBe("assistant");

    // Check content_block_start
    expect(events[1].data.content_block.type).toBe("text");
    expect(events[1].data.index).toBe(0);

    // Check deltas
    expect(events[2].data.delta.type).toBe("text_delta");
    expect(events[2].data.delta.text).toBe("Hello");
    expect(events[3].data.delta.text).toBe(" world");

    // Check message_delta
    expect(events[5].data.delta.stop_reason).toBe("end_turn");
  });

  it("emits thinking + text blocks with correct indices", () => {
    const converter = new StreamToAnthropicConverter("test-model", { id: "msg_test" });

    const chunks1 = converter.handleEvent({
      type: "thinking",
      subtype: "delta",
      timestamp_ms: 1,
      text: "Let me think",
    });

    const chunks2 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: 2,
      message: { role: "assistant", content: [{ type: "text", text: "Answer" }] },
    });

    const finish = converter.finish();
    const events = parseSseEvents([...chunks1, ...chunks2, ...finish]);
    const types = events.map((e) => e.event);

    expect(types).toEqual([
      "message_start",
      "content_block_start", // thinking
      "content_block_delta", // thinking delta
      "content_block_stop",  // close thinking
      "content_block_start", // text
      "content_block_delta", // text delta
      "content_block_stop",  // close text
      "message_delta",
      "message_stop",
    ]);

    // Thinking block at index 0
    expect(events[1].data.index).toBe(0);
    expect(events[1].data.content_block.type).toBe("thinking");
    expect(events[2].data.delta.type).toBe("thinking_delta");
    expect(events[2].data.delta.thinking).toBe("Let me think");

    // Text block at index 1
    expect(events[4].data.index).toBe(1);
    expect(events[4].data.content_block.type).toBe("text");
  });

  it("emits tool_use block with correct format", () => {
    const converter = new StreamToAnthropicConverter("test-model", { id: "msg_test" });

    const chunks1 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: 1,
      message: { role: "assistant", content: [{ type: "text", text: "Using tool" }] },
    });

    const chunks2 = converter.handleEvent({
      type: "tool_call",
      call_id: "toolu_123",
      tool_call: {
        bash: { args: { command: "ls -la" } },
      },
    });

    const finish = converter.finish();
    const events = parseSseEvents([...chunks1, ...chunks2, ...finish]);

    // Find tool_use events
    const toolStart = events.find(
      (e) => e.event === "content_block_start" && e.data.content_block?.type === "tool_use",
    );
    expect(toolStart).toBeTruthy();
    expect(toolStart!.data.content_block.id).toBe("toolu_123");
    expect(toolStart!.data.content_block.name).toBe("bash");
    expect(toolStart!.data.index).toBe(1); // text was index 0

    const toolDelta = events.find(
      (e) => e.event === "content_block_delta" && e.data.delta?.type === "input_json_delta",
    );
    expect(toolDelta).toBeTruthy();
    const parsedArgs = JSON.parse(toolDelta!.data.delta.partial_json);
    expect(parsedArgs.command).toBe("ls -la");

    // Stop reason should be tool_use
    const msgDelta = events.find((e) => e.event === "message_delta");
    expect(msgDelta!.data.delta.stop_reason).toBe("tool_use");
  });

  it("handles empty stream gracefully", () => {
    const converter = new StreamToAnthropicConverter("test-model", { id: "msg_test" });
    const finish = converter.finish();
    const events = parseSseEvents(finish);

    expect(events.length).toBeGreaterThanOrEqual(3); // message_start, message_delta, message_stop
    expect(events[0].event).toBe("message_start");
    expect(events[events.length - 1].event).toBe("message_stop");
  });

  it("does not duplicate text when a final full event still carries timestamp_ms", () => {
    const converter = new StreamToAnthropicConverter("test-model", { id: "msg_test" });
    const now = Date.now();

    const chunks1 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: now + 1,
      message: { role: "assistant", content: [{ type: "text", text: "Hello" }] },
    } as any);

    const chunks2 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: now + 2,
      message: { role: "assistant", content: [{ type: "text", text: " world" }] },
    } as any);

    // Some backends emit final accumulated text but still include timestamp_ms.
    const chunks3 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: now + 3,
      message: { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
    } as any);

    const finish = converter.finish();
    const events = parseSseEvents([...chunks1, ...chunks2, ...chunks3, ...finish]);
    const textDeltas = events
      .filter((e) => e.event === "content_block_delta" && e.data.delta?.type === "text_delta")
      .map((e) => e.data.delta.text)
      .join("");

    expect(textDeltas).toBe("Hello world");
  });

  it("handles cumulative partial text events without repeating emitted content", () => {
    const converter = new StreamToAnthropicConverter("test-model", { id: "msg_test" });
    const now = Date.now();

    const chunks1 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: now + 1,
      message: { role: "assistant", content: [{ type: "text", text: "Hello" }] },
    } as any);

    // Cumulative chunk instead of delta chunk.
    const chunks2 = converter.handleEvent({
      type: "assistant",
      timestamp_ms: now + 2,
      message: { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
    } as any);

    const finish = converter.finish();
    const events = parseSseEvents([...chunks1, ...chunks2, ...finish]);
    const textDeltas = events
      .filter((e) => e.event === "content_block_delta" && e.data.delta?.type === "text_delta")
      .map((e) => e.data.delta.text)
      .join("");

    expect(textDeltas).toBe("Hello world");
  });
});
