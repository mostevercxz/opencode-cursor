import { startStandaloneProxy } from "./src/proxy/daemon.js";

function parseArgs(argv: string[]) {
  const out: {
    host?: string;
    port?: number;
    logLevel?: string;
    idleTimeoutSeconds?: number;
    sseHeartbeatMs?: number;
    help?: boolean;
  } = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") { out.help = true; continue; }
    if (arg === "--host" && argv[i + 1]) { out.host = argv[++i]; continue; }
    if (arg === "--port" && argv[i + 1]) { const n = parseInt(argv[++i], 10); if (n > 0) out.port = n; continue; }
    if (arg === "--log-level" && argv[i + 1]) { out.logLevel = argv[++i]; continue; }
    if (arg === "--idle-timeout" && argv[i + 1]) {
      const n = parseInt(argv[++i], 10);
      if (n >= 0) out.idleTimeoutSeconds = n;
      continue;
    }
    if (arg === "--heartbeat-ms" && argv[i + 1]) {
      const n = parseInt(argv[++i], 10);
      if (n >= 0) out.sseHeartbeatMs = n;
      continue;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

if (args.help) {
  console.log(`opencode-cursor-proxy — Standalone Cursor proxy server

Usage:
  bun run standalone.ts [options]

Options:
  --host <addr>       Listen address (default: 127.0.0.1, env: PROXY_HOST)
  --port <port>       Listen port (default: 32124, env: PROXY_PORT)
  --log-level <level> Log level: debug|info|warn|error (default: info, env: LOG_LEVEL)
  --idle-timeout <s>  Bun idle timeout seconds (default: 120, env: PROXY_IDLE_TIMEOUT_SEC)
  --heartbeat-ms <n>  SSE keepalive interval in ms (default: 5000, env: PROXY_SSE_HEARTBEAT_MS, 0 disables)
  -h, --help          Show this help

Endpoints:
  GET  /health              Health check
  GET  /v1/models           List available models
  POST /v1/chat/completions OpenAI-compatible chat completions
  POST /v1/messages         Anthropic-compatible messages`);
  process.exit(0);
}

const host = args.host ?? process.env.PROXY_HOST ?? "127.0.0.1";
const port = args.port ?? (process.env.PROXY_PORT ? parseInt(process.env.PROXY_PORT, 10) : 32124);
const logLevel = args.logLevel ?? process.env.LOG_LEVEL ?? "info";
const idleTimeoutSeconds = args.idleTimeoutSeconds
  ?? (process.env.PROXY_IDLE_TIMEOUT_SEC ? parseInt(process.env.PROXY_IDLE_TIMEOUT_SEC, 10) : undefined);
const sseHeartbeatMs = args.sseHeartbeatMs
  ?? (process.env.PROXY_SSE_HEARTBEAT_MS ? parseInt(process.env.PROXY_SSE_HEARTBEAT_MS, 10) : undefined);

const instance = await startStandaloneProxy({ host, port, logLevel, idleTimeoutSeconds, sseHeartbeatMs });

console.log(`Standalone proxy listening: ${instance.url}`);
console.log(`OpenAI-compatible baseURL:  ${instance.baseURL}`);

let shuttingDown = false;
const shutdown = async (signal: string) => {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`\nReceived ${signal}, shutting down...`);
  await instance.stop();
  process.exit(0);
};

process.on("SIGINT", () => void shutdown("SIGINT"));
process.on("SIGTERM", () => void shutdown("SIGTERM"));
