#!/usr/bin/env node
// Stop hook — Tencent-style auto-capture for memory-agent.
// Reads the latest user+assistant exchange from the transcript and POSTs /capture.
// Fail-soft: never blocks the turn if the sidecar is down.

const fs = require('fs');
const path = require('path');
const os = require('os');
const http = require('http');
const https = require('https');
const crypto = require('crypto');

const API_BASE = (process.env.MEMORY_AGENT_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
// Env wins; else ~/.memory-agent/api-secret (written by setup scripts).
function loadApiSecret() {
  const fromEnv = (process.env.MEMORY_API_SECRET || '').trim();
  if (fromEnv) return fromEnv;
  try {
    const p = path.join(os.homedir(), '.memory-agent', 'api-secret');
    return fs.readFileSync(p, 'utf8').trim();
  } catch {
    return '';
  }
}
const API_SECRET = loadApiSecret();
// Canonical form matches server ids.canonicalize_user_id (trim + lowercase).
const USER_ID = String(process.env.MEMORY_AGENT_USER_ID || os.userInfo().username || 'default')
  .trim()
  .toLowerCase();
const AGENT_ID = process.env.MEMORY_AGENT_AGENT_ID || 'claude-code';
const STATE_DIR = path.join(os.homedir(), '.memory-agent', 'capture');
const LOG_PATH = path.join(os.homedir(), '.claude', 'hooks', 'logs', 'memory-auto-capture.jsonl');

function log(event) {
  try {
    fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
    fs.appendFileSync(LOG_PATH, JSON.stringify({ ts: new Date().toISOString(), ...event }) + '\n');
  } catch { /* never block */ }
}

function readStdin() {
  try {
    return JSON.parse(fs.readFileSync(0, 'utf8') || '{}');
  } catch {
    return {};
  }
}

function resolveTranscript(input) {
  let tp = input.transcript_path;
  if (tp && fs.existsSync(tp)) return tp;
  const cwd = input.cwd || process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const dir = path.join(os.homedir(), '.claude', 'projects', cwd.replace(/[/\\:]/g, '-'));
  if (input.session_id) {
    const byId = path.join(dir, input.session_id + '.jsonl');
    if (fs.existsSync(byId)) return byId;
  }
  if (!fs.existsSync(dir)) return null;
  const newest = fs.readdirSync(dir).filter(f => f.endsWith('.jsonl'))
    .map(f => ({ f, t: fs.statSync(path.join(dir, f)).mtimeMs }))
    .sort((a, b) => b.t - a.t)[0];
  return newest ? path.join(dir, newest.f) : null;
}

const MAX_USER_CHARS = 2000;
const MAX_ASSISTANT_CHARS = 2500;

function hasToolResult(c) {
  return Array.isArray(c) && c.some(b => b && b.type === 'tool_result');
}

function textOfContent(c) {
  if (typeof c === 'string') return c;
  if (!Array.isArray(c)) return '';
  return c.map(b => (b && b.type === 'text' && b.text) || '').join('\n').trim();
}

// Transcript "user" rows that are NOT human input (harness / eval / skill runner).
function isHarnessUserText(text) {
  const t = (text || '').trim();
  if (!t) return true;
  if (/^<(local-command|command-name|command-message|command-args|task-notification)\b/i.test(t)) {
    return true;
  }
  if (/^<\/?(task-id|tool-use-id|output-file|status|summary|result|note|usage)\b/i.test(t)) {
    return true;
  }
  // Background agent completion payload often embeds the whole subagent answer.
  if (t.includes('<task-notification') || t.includes('</task-notification>')) return true;
  if (t.includes('<task-id>') && t.includes('<status>')) return true;
  // skill-audit / eval runners
  if (/^Grade an assistant's answer against each expected behavior/i.test(t)) return true;
  if (/^Base directory for this skill:/i.test(t)) return true;
  // slash commands (short)
  if (/^\//.test(t) && t.length < 80) return true;
  return false;
}

function clip(text, max) {
  const s = (text || '').trim();
  if (s.length <= max) return s;
  return s.slice(0, max).trimEnd() + '…';
}

function extractExchange(entries) {
  // Walk backward for the latest *human* user message (skip harness-as-user rows).
  let start = -1;
  let userText = '';
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i];
    if (e.type !== 'user' || hasToolResult(e.message && e.message.content)) continue;
    const t = textOfContent(e.message && e.message.content);
    if (isHarnessUserText(t)) continue;
    userText = t;
    start = i;
    break;
  }
  if (start < 0 || !userText.trim()) return null;

  // Only the *last* assistant text after that user turn — not mid-turn narration glued together.
  let assistantText = '';
  for (let i = start + 1; i < entries.length; i++) {
    const e = entries[i];
    // A later *human* user message ends this exchange window.
    if (e.type === 'user' && !hasToolResult(e.message && e.message.content)) {
      const t = textOfContent(e.message && e.message.content);
      if (!isHarnessUserText(t)) break;
      continue;
    }
    if (e.type !== 'assistant') continue;
    const t = textOfContent(e.message && e.message.content);
    if (t) assistantText = t; // keep last only
  }
  if (!assistantText.trim()) return null;

  return {
    messages: [
      { role: 'user', content: clip(userText, MAX_USER_CHARS) },
      { role: 'assistant', content: clip(assistantText, MAX_ASSISTANT_CHARS) },
    ],
  };
}

function statePath(sessionKey) {
  const safe = crypto.createHash('sha1').update(sessionKey).digest('hex').slice(0, 16);
  return path.join(STATE_DIR, safe + '.json');
}

function loadState(sessionKey) {
  try {
    return JSON.parse(fs.readFileSync(statePath(sessionKey), 'utf8'));
  } catch {
    return { lastFingerprint: null };
  }
}

function saveState(sessionKey, state) {
  try {
    fs.mkdirSync(STATE_DIR, { recursive: true });
    fs.writeFileSync(statePath(sessionKey), JSON.stringify(state));
  } catch { /* ignore */ }
}

function fingerprint(messages) {
  return crypto.createHash('sha256')
    .update(messages.map(m => m.role + '\0' + m.content).join('\0'))
    .digest('hex');
}

function postJson(urlString, body) {
  return new Promise((resolve) => {
    let u;
    try { u = new URL(urlString); } catch {
      resolve({ ok: false, error: 'bad-url' });
      return;
    }
    const lib = u.protocol === 'https:' ? https : http;
    const data = Buffer.from(JSON.stringify(body), 'utf8');
    const headers = {
      'Content-Type': 'application/json',
      'Content-Length': data.length,
    };
    if (API_SECRET) headers['X-Memory-Key'] = API_SECRET;
    const req = lib.request({
      hostname: u.hostname,
      port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: u.pathname + u.search,
      method: 'POST',
      headers,
      timeout: 8000,
    }, (res) => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        let parsed = null;
        try { parsed = JSON.parse(text); } catch { /* raw */ }
        resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, body: parsed || text });
      });
    });
    req.on('error', (err) => resolve({ ok: false, error: err.message }));
    req.on('timeout', () => { req.destroy(); resolve({ ok: false, error: 'timeout' }); });
    req.write(data);
    req.end();
  });
}

(async () => {
  const input = readStdin();
  // Never force a continuation loop from capture.
  if (input.stop_hook_active === true) process.exit(0);

  const tp = resolveTranscript(input);
  if (!tp) {
    log({ stage: 'no-transcript' });
    process.exit(0);
  }

  let entries = [];
  try {
    entries = fs.readFileSync(tp, 'utf8').split('\n').filter(Boolean)
      .map(l => { try { return JSON.parse(l); } catch { return null; } })
      .filter(Boolean);
  } catch {
    log({ stage: 'unreadable-transcript' });
    process.exit(0);
  }

  const exchange = extractExchange(entries);
  if (!exchange) {
    log({ stage: 'no-exchange' });
    process.exit(0);
  }

  const sessionKey = String(input.session_id || path.basename(tp, '.jsonl'));
  const fp = fingerprint(exchange.messages);
  const st = loadState(sessionKey);
  if (st.lastFingerprint === fp) {
    log({ stage: 'local-dedupe', sessionKey });
    process.exit(0);
  }

  const payload = {
    user_id: USER_ID,
    session_key: sessionKey,
    agent_id: AGENT_ID,
    messages: exchange.messages,
    metadata: {
      cwd: input.cwd || null,
      source: 'claude-code-stop',
    },
  };

  const res = await postJson(API_BASE + '/capture', payload);
  if (!res.ok) {
    log({ stage: 'api-error', sessionKey, error: res.error || res.status, body: res.body });
    process.exit(0);
  }

  st.lastFingerprint = fp;
  st.lastSeen = (res.body && res.body.messages_seen) || st.lastSeen || 0;
  saveState(sessionKey, st);
  log({
    stage: 'ok',
    sessionKey,
    messages_captured: res.body && res.body.messages_captured,
    duplicate: res.body && res.body.duplicate,
    memories_added: res.body && res.body.memories_added,
  });
  process.exit(0);
})().catch((err) => {
  log({ stage: 'crash', error: String(err && err.message || err) });
  process.exit(0);
});
