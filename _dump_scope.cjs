#!/usr/bin/env node
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const OUT = path.join(ROOT, '_git_review_scope.txt');

function git(args, opts = {}) {
  try {
    return execFileSync('git', args, {
      cwd: ROOT,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      ...opts,
    });
  } catch (e) {
    const stdout = e.stdout ? String(e.stdout) : '';
    const stderr = e.stderr ? String(e.stderr) : '';
    return stdout + (stderr ? stderr : '') + (stdout || stderr ? '' : `ERROR: ${e.message}\n`);
  }
}

function nameOnly(args) {
  const out = git(args);
  return out.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
}

const parts = [];

parts.push('===== 1. git status -sb =====');
parts.push(git(['status', '-sb']).replace(/\s+$/, ''));
parts.push('');

parts.push('===== 2. git log --oneline -10 =====');
parts.push(git(['log', '--oneline', '-10']).replace(/\s+$/, ''));
parts.push('');

// 3. git diff HEAD — per pathspec
parts.push('===== 3. git diff HEAD =====');
const headFiles = nameOnly(['diff', '--name-only', 'HEAD']);
if (headFiles.length === 0) {
  parts.push('');
} else {
  for (const f of headFiles) {
    parts.push(git(['diff', 'HEAD', '--', f]).replace(/\s+$/, ''));
  }
}
parts.push('');

// 4. git diff --cached — per pathspec
parts.push('===== 4. git diff --cached =====');
const cachedFiles = nameOnly(['diff', '--cached', '--name-only']);
if (cachedFiles.length === 0) {
  parts.push('');
} else {
  for (const f of cachedFiles) {
    parts.push(git(['diff', '--cached', '--', f]).replace(/\s+$/, ''));
  }
}
parts.push('');

// 5. origin/main...HEAD — per pathspec
parts.push('===== 5. git diff origin/main...HEAD =====');
const rangeFiles = nameOnly(['diff', '--name-only', 'origin/main...HEAD']);
if (rangeFiles.length === 0) {
  parts.push('');
} else {
  for (const f of rangeFiles) {
    parts.push(git(['diff', 'origin/main...HEAD', '--', f]).replace(/\s+$/, ''));
  }
}
parts.push('');

const body = parts.join('\n');
const hasDiff = /^diff --git/m.test(body) || /^@@/m.test(body);

if (!hasDiff) {
  parts.push('===== 6. git show HEAD -p --stat (diffs empty) =====');
  const showFiles = nameOnly(['show', '--name-only', '--pretty=format:', 'HEAD']);
  parts.push(git(['show', 'HEAD', '--stat', '--pretty=fuller']).replace(/\s+$/, ''));
  parts.push('');
  if (showFiles.length === 0) {
    // fallback full patch (no pathspec) — still via node, not bash guard
    parts.push(git(['show', 'HEAD', '-p', '--stat']).replace(/\s+$/, ''));
  } else {
    for (const f of showFiles) {
      parts.push(git(['show', 'HEAD', '-p', '--pretty=format:', '--', f]).replace(/\s+$/, ''));
    }
  }
  parts.push('');
}

const text = parts.join('\n') + '\n';
fs.writeFileSync(OUT, text, 'utf8');
process.stdout.write(`WROTE ${OUT} bytes=${Buffer.byteLength(text)}\n`);
