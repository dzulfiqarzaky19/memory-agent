// Stop hook: persists the last user+assistant exchange into memory-agent via POST /add.
const fs = require("fs");
const readline = require("readline");

const USER_ID = "zaky";
const API_URL = "http://localhost:8000/add";
const LOG_PATH = __dirname + "/save_memory.log";

function log(msg) {
  try {
    fs.appendFileSync(LOG_PATH, `[${new Date().toISOString()}] ${msg}\n`);
  } catch {}
}

function textOf(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n")
    .trim();
}

function isRealUserEntry(entry) {
  if (entry.type !== "user") return false;
  if (entry.isMeta || entry.isSidechain) return false;
  const content = entry.message && entry.message.content;
  if (typeof content === "string") {
    const text = content.trim();
    // System-injected notices (task-notification, system-reminder, etc.) are
    // wrapped in a leading XML-ish tag; genuine typed prompts never start with one.
    if (!text || /^</.test(text)) return false;
    return true;
  }
  if (!Array.isArray(content)) return false;
  return content.some((b) => b.type === "text");
}

function isRealAssistantEntry(entry) {
  if (entry.type !== "assistant") return false;
  if (entry.isSidechain) return false;
  const content = entry.message && entry.message.content;
  return Array.isArray(content) && content.some((b) => b.type === "text");
}

async function main() {
  let input = "";
  for await (const chunk of process.stdin) input += chunk;

  let payload;
  try {
    payload = JSON.parse(input);
  } catch (err) {
    log(`bad stdin JSON: ${err.message}`);
    return;
  }

  if (payload.stop_hook_active) {
    log("skipped: stop_hook_active=true (loop guard)");
    return;
  }

  const transcriptPath = payload.transcript_path;
  if (!transcriptPath || !fs.existsSync(transcriptPath)) {
    log(`skipped: no transcript_path (${transcriptPath})`);
    return;
  }

  const lines = fs.readFileSync(transcriptPath, "utf8").split("\n").filter(Boolean);
  const entries = [];
  for (const line of lines) {
    try {
      entries.push(JSON.parse(line));
    } catch {}
  }

  // Walk backward from the end, skipping tool_result/meta/structural entries,
  // collecting assistant text blocks, until the last genuine user text prompt.
  let i = entries.length - 1;
  const assistantChunks = [];
  while (i >= 0 && !isRealUserEntry(entries[i])) {
    if (isRealAssistantEntry(entries[i])) {
      assistantChunks.unshift(textOf(entries[i].message.content));
    }
    i--;
  }
  const assistantText = assistantChunks.filter(Boolean).join("\n").trim();
  const userText = i >= 0 ? textOf(entries[i].message.content) : "";

  if (!userText || !assistantText) {
    log(`skipped: empty exchange (user=${!!userText}, assistant=${!!assistantText})`);
    return;
  }

  const messages = [
    { role: "user", content: userText },
    { role: "assistant", content: assistantText },
  ];

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, messages }),
      signal: AbortSignal.timeout(90000),
    });
    if (!res.ok) {
      log(`store failed: HTTP ${res.status} ${await res.text()}`);
      return;
    }
    const data = await res.json();
    log(`stored ok: memories_added=${data.memories_added}`);
  } catch (err) {
    log(`store error: ${err.message}`);
  }
}

main();
