#!/usr/bin/env node
// SessionStart hook — inject the partner pack into a new session's context.
// Forced load: the pack is present without the model choosing to call a tool.
// Fail-soft: a down sidecar must never block session start (always exit 0).

const fs = require('fs');
const path = require('path');
const os = require('os');
const http = require('http');
const https = require('https');

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
const LOG_PATH = path.join(os.homedir(), '.claude', 'hooks', 'logs', 'memory-partner-pack.jsonl');

function log(event) {
  try {
    fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
    fs.appendFileSync(LOG_PATH, JSON.stringify({ ts: new Date().toISOString(), ...event }) + '\n');
  } catch { /* never block */ }
}

function getJson(url) {
  return new Promise((resolve) => {
    let u;
    try { u = new URL(url); } catch (e) { resolve({ ok: false, error: 'bad-url' }); return; }
    const lib = u.protocol === 'https:' ? https : http;
    const headers = {};
    if (API_SECRET) headers['X-Memory-Key'] = API_SECRET;
    const req = lib.request({
      hostname: u.hostname,
      port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: u.pathname + u.search,
      method: 'GET',
      headers,
      // Short: session start must not wait on a slow sidecar.
      timeout: 5000,
    }, (res) => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        let parsed = null;
        try { parsed = JSON.parse(text); } catch { /* raw */ }
        resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, body: parsed });
      });
    });
    req.on('error', (err) => resolve({ ok: false, error: err.message }));
    req.on('timeout', () => { req.destroy(); resolve({ ok: false, error: 'timeout' }); });
    req.end();
  });
}

function bullets(facts, empty) {
  if (!Array.isArray(facts) || facts.length === 0) return `  (${empty})`;
  return facts.map(f => `  - ${f.text}`).join('\n');
}

function render(pack) {
  const other = pack.other || {};
  const lines = [];
  if (pack.stale) {
    const secs = Number(pack.stale_seconds || 0);
    const age = secs >= 3600 ? `~${Math.floor(secs / 3600)}h`
      : secs >= 60 ? `~${Math.floor(secs / 60)}m` : `${secs}s`;
    lines.push(`(memory is stale by ${age} — extraction pending; absence is not proof)`);
  }
  lines.push(`## Partner pack — ${pack.user_id} + ${pack.agent_id}`);
  lines.push('');
  lines.push(`**Who they are** (${other.memory_count || 0} memories):`);
  lines.push(other.summary ? `  ${other.summary}` : '  (no persona cached yet)');
  if ((other.instructions || []).length) {
    lines.push('');
    lines.push('**Their standing instructions:**');
    lines.push(bullets(other.instructions, 'none'));
  }
  lines.push('');
  lines.push('**How I work:**');
  lines.push(bullets(pack.self, 'none'));
  if ((pack.relation || []).length) {
    lines.push('');
    lines.push('**How we work together:**');
    lines.push(bullets(pack.relation, 'none'));
  }
  return lines.join('\n');
}

(async () => {
  const url = `${API_BASE}/partner/${encodeURIComponent(USER_ID)}?agent_id=${encodeURIComponent(AGENT_ID)}`;
  const res = await getJson(url);
  if (!res.ok || !res.body) {
    log({ stage: 'api-error', status: res.status, error: res.error });
    process.exit(0);
  }
  try {
    process.stdout.write(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: 'SessionStart',
        additionalContext: render(res.body),
      },
    }) + '\n');
    log({
      stage: 'ok',
      user_id: USER_ID,
      agent_id: AGENT_ID,
      self: (res.body.self || []).length,
      relation: (res.body.relation || []).length,
      stale: !!res.body.stale,
    });
  } catch (e) {
    log({ stage: 'render-error', error: String(e && e.message || e) });
  }
  process.exit(0);
})().catch((e) => {
  log({ stage: 'crash', error: String(e && e.message || e) });
  process.exit(0);
});
