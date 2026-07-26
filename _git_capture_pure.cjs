/**
 * Pure-Node git review-scope dump (no git binary).
 * Writes _git_review_scope.txt with status/log/diffs best-effort from .git + worktree.
 */
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const zlib = require("zlib");

const repo = __dirname;
const gitDir = path.join(repo, ".git");
const outPath = path.join(repo, "_git_review_scope.txt");

function read(p) {
  return fs.readFileSync(p);
}
function readText(p) {
  return fs.readFileSync(p, "utf8");
}
function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function resolveRef(name) {
  // name like HEAD, refs/heads/main, refs/remotes/origin/main
  let p = path.join(gitDir, name);
  if (name === "HEAD") p = path.join(gitDir, "HEAD");
  if (!exists(p) && name.startsWith("refs/")) p = path.join(gitDir, name);
  if (!exists(p)) {
    // packed-refs
    const packed = path.join(gitDir, "packed-refs");
    if (exists(packed)) {
      const lines = readText(packed).split(/\r?\n/);
      for (const line of lines) {
        if (!line || line.startsWith("#") || line.startsWith("^")) continue;
        const sp = line.indexOf(" ");
        if (sp < 0) continue;
        const sha = line.slice(0, sp);
        const ref = line.slice(sp + 1);
        if (ref === name) return sha.trim();
      }
    }
    return null;
  }
  const t = readText(p).trim();
  if (t.startsWith("ref:")) {
    return resolveRef(t.slice(4).trim());
  }
  return t;
}

function inflateObject(sha) {
  const dir = sha.slice(0, 2);
  const file = sha.slice(2);
  const p = path.join(gitDir, "objects", dir, file);
  if (!exists(p)) {
    // try alternates / pack — limited: scan pack idx if needed
    return readPackedObject(sha);
  }
  const raw = zlib.inflateSync(read(p));
  const nul = raw.indexOf(0);
  const header = raw.slice(0, nul).toString("utf8");
  const body = raw.slice(nul + 1);
  const [type, sizeStr] = header.split(" ");
  return { type, size: Number(sizeStr), body };
}

function readPackedObject(sha) {
  const packDir = path.join(gitDir, "objects", "pack");
  if (!exists(packDir)) throw new Error("object not found: " + sha);
  const files = fs.readdirSync(packDir).filter((f) => f.endsWith(".idx"));
  for (const idxName of files) {
    const idxPath = path.join(packDir, idxName);
    const packPath = path.join(packDir, idxName.replace(/\.idx$/, ".pack"));
    const off = findOffsetInIdx(idxPath, sha);
    if (off == null) continue;
    return extractFromPack(packPath, off, sha);
  }
  throw new Error("object not found in packs: " + sha);
}

function findOffsetInIdx(idxPath, sha) {
  const buf = read(idxPath);
  // version 2 idx only
  if (buf.slice(0, 4).toString("binary") !== "\xfftOc") {
    // v1 not supported
    return null;
  }
  const version = buf.readUInt32BE(4);
  if (version !== 2) return null;
  // fanout 256 * 4 at offset 8
  const fanoutBase = 8;
  const shaBytes = Buffer.from(sha, "hex");
  const first = shaBytes[0];
  const startCount = first === 0 ? 0 : buf.readUInt32BE(fanoutBase + (first - 1) * 4);
  const endCount = buf.readUInt32BE(fanoutBase + first * 4);
  const numObjects = buf.readUInt32BE(fanoutBase + 255 * 4);
  const shaTable = fanoutBase + 256 * 4;
  // each sha 20 bytes
  let lo = startCount;
  let hi = endCount;
  let found = -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const off = shaTable + mid * 20;
    const cmp = shaBytes.compare(buf.slice(off, off + 20));
    if (cmp === 0) {
      found = mid;
      break;
    }
    if (cmp < 0) hi = mid;
    else lo = mid + 1;
  }
  if (found < 0) return null;
  // skip sha table, crc table
  const crcTable = shaTable + numObjects * 20;
  const offsetTable = crcTable + numObjects * 4;
  const offset = buf.readUInt32BE(offsetTable + found * 4);
  if (offset & 0x80000000) {
    // large offset
    const largeTable = offsetTable + numObjects * 4;
    const largeIdx = offset & 0x7fffffff;
    // 8-byte offsets
    const hi2 = buf.readUInt32BE(largeTable + largeIdx * 8);
    const lo2 = buf.readUInt32BE(largeTable + largeIdx * 8 + 4);
    return hi2 * 0x100000000 + lo2;
  }
  return offset;
}

function extractFromPack(packPath, offset) {
  const fd = fs.openSync(packPath, "r");
  try {
    // read a chunk from offset
    const stat = fs.fstatSync(fd);
    const size = Math.min(stat.size - offset, 64 * 1024 * 1024);
    const buf = Buffer.alloc(size);
    fs.readSync(fd, buf, 0, size, offset);
    // parse variable-length type/size
    let c = buf[0];
    let type = (c >> 4) & 7;
    let objSize = c & 15;
    let shift = 4;
    let i = 1;
    while (c & 0x80) {
      c = buf[i++];
      objSize |= (c & 0x7f) << shift;
      shift += 7;
    }
    const types = {
      1: "commit",
      2: "tree",
      3: "blob",
      4: "tag",
      6: "ofs_delta",
      7: "ref_delta",
    };
    const typeName = types[type] || String(type);
    if (type === 6 || type === 7) {
      throw new Error("delta objects not fully supported in pure reader for " + typeName);
    }
    // zlib from i
    const inflated = zlib.inflateSync(buf.slice(i));
    return { type: typeName, size: objSize, body: inflated };
  } finally {
    fs.closeSync(fd);
  }
}

function parseCommit(body) {
  const text = body.toString("utf8");
  const lines = text.split("\n");
  const out = { tree: null, parents: [], author: "", committer: "", message: "" };
  let i = 0;
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (line === "") {
      i++;
      break;
    }
    if (line.startsWith("tree ")) out.tree = line.slice(5).trim();
    else if (line.startsWith("parent ")) out.parents.push(line.slice(7).trim());
    else if (line.startsWith("author ")) out.author = line.slice(7);
    else if (line.startsWith("committer ")) out.committer = line.slice(10);
  }
  out.message = lines.slice(i).join("\n").replace(/\n$/, "");
  return out;
}

function parseTree(body) {
  // entries: mode SP name NUL sha20
  const entries = [];
  let i = 0;
  while (i < body.length) {
    const sp = body.indexOf(0x20, i);
    const mode = body.slice(i, sp).toString("utf8");
    const nul = body.indexOf(0, sp + 1);
    const name = body.slice(sp + 1, nul).toString("utf8");
    const sha = body.slice(nul + 1, nul + 21).toString("hex");
    entries.push({ mode, name, sha });
    i = nul + 21;
  }
  return entries;
}

function loadTreeRecursive(treeSha, prefix = "") {
  const obj = inflateObject(treeSha);
  if (obj.type !== "tree") throw new Error("not a tree: " + treeSha);
  const map = new Map(); // path -> {mode, sha}
  for (const e of parseTree(obj.body)) {
    const p = prefix ? prefix + "/" + e.name : e.name;
    if (e.mode === "40000") {
      const sub = loadTreeRecursive(e.sha, p);
      for (const [k, v] of sub) map.set(k, v);
    } else {
      map.set(p, { mode: e.mode, sha: e.sha });
    }
  }
  return map;
}

function blobHash(content) {
  // content is Buffer
  const header = Buffer.from(`blob ${content.length}\0`);
  return crypto.createHash("sha1").update(header).update(content).digest("hex");
}

function isIgnored(rel) {
  // minimal: .git and our temp files
  if (rel === ".git" || rel.startsWith(".git/") || rel.startsWith(".git\\")) return true;
  const base = path.basename(rel);
  if (
    base === "_git_review_scope.txt" ||
    base === "_git_capture.cjs" ||
    base === "_git_capture_pure.cjs" ||
    base === "_capture_git_review.sh" ||
    base.startsWith("_diff_")
  )
    return true;
  // read .gitignore if present — basic patterns only
  return false;
}

function listWorktreeFiles(dir, prefix = "") {
  const out = [];
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  // load gitignore simple
  for (const ent of entries) {
    const rel = prefix ? prefix + "/" + ent.name : ent.name;
    if (ent.name === ".git") continue;
    if (isIgnored(rel)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      out.push(...listWorktreeFiles(full, rel));
    } else if (ent.isFile()) {
      out.push(rel.replace(/\\/g, "/"));
    }
  }
  return out;
}

function loadGitignore() {
  const p = path.join(repo, ".gitignore");
  if (!exists(p)) return [];
  return readText(p)
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
}

function matchIgnore(rel, patterns) {
  // very small subset: exact, trailing /, leading **, simple *
  const norm = rel.replace(/\\/g, "/");
  for (let pat of patterns) {
    let neg = false;
    if (pat.startsWith("!")) {
      neg = true;
      pat = pat.slice(1);
    }
    if (pat.endsWith("/")) {
      if (norm === pat.slice(0, -1) || norm.startsWith(pat)) {
        if (!neg) return true;
      }
      continue;
    }
    // glob to regex rough
    let g = pat;
    if (g.startsWith("/")) g = g.slice(1);
    const rx = new RegExp(
      "^" +
        g
          .replace(/[.+^${}()|[\]\\]/g, "\\$&")
          .replace(/\*\*/g, ":::DD:::")
          .replace(/\*/g, "[^/]*")
          .replace(/:::DD:::/g, ".*") +
        "$"
    );
    if (rx.test(norm) || rx.test(path.basename(norm))) {
      if (!neg) return true;
    }
    // also directory prefix for patterns like __pycache__/
  }
  // hardcoded common
  if (norm.includes("__pycache__/") || norm.endsWith(".pyc")) return true;
  if (norm === "node_modules" || norm.startsWith("node_modules/")) return true;
  return false;
}

// --- index parser (v2) for staged ---
function parseIndex() {
  const p = path.join(gitDir, "index");
  if (!exists(p)) return new Map();
  const buf = read(p);
  const sig = buf.slice(0, 4).toString("utf8");
  if (sig !== "DIRC") throw new Error("bad index");
  const version = buf.readUInt32BE(4);
  const entriesCount = buf.readUInt32BE(8);
  let offset = 12;
  const map = new Map();
  for (let n = 0; n < entriesCount; n++) {
    // 62 bytes fixed + name
    const ctimeSec = buf.readUInt32BE(offset);
    const ctimeNsec = buf.readUInt32BE(offset + 4);
    const mtimeSec = buf.readUInt32BE(offset + 8);
    const mtimeNsec = buf.readUInt32BE(offset + 12);
    const dev = buf.readUInt32BE(offset + 16);
    const ino = buf.readUInt32BE(offset + 20);
    const mode = buf.readUInt32BE(offset + 24);
    const uid = buf.readUInt32BE(offset + 28);
    const gid = buf.readUInt32BE(offset + 32);
    const size = buf.readUInt32BE(offset + 36);
    const sha = buf.slice(offset + 40, offset + 60).toString("hex");
    const flags = buf.readUInt16BE(offset + 60);
    let nameLen = flags & 0xfff;
    let entryLen;
    let name;
    if (nameLen === 0xfff) {
      // long name
      const start = offset + 62;
      let end = start;
      while (buf[end] !== 0) end++;
      name = buf.slice(start, end).toString("utf8");
      entryLen = end + 1 - offset;
      // pad to 8
      entryLen = Math.ceil((62 + name.length + 1) / 8) * 8;
    } else {
      name = buf.slice(offset + 62, offset + 62 + nameLen).toString("utf8");
      entryLen = 62 + nameLen + 1;
      entryLen = Math.ceil(entryLen / 8) * 8;
    }
    // skip extended flags if any
    if (flags & 0x4000 && version >= 3) {
      // extended — skip 2 more already in padding usually
    }
    map.set(name.replace(/\\/g, "/"), { sha, mode, size, mtimeSec });
    offset += entryLen;
  }
  return map;
}

function modeStr(mode) {
  return (mode & 0o777777).toString(8).padStart(6, "0");
}

function unifiedDiff(pathName, oldText, newText, oldLabel, newLabel) {
  // minimal line diff (LCS-lite Myers simplified via arrays)
  const a = oldText == null ? [] : oldText.split("\n");
  const b = newText == null ? [] : newText.split("\n");
  // drop trailing empty from split if file ends with newline handled carefully
  if (oldText != null && oldText.endsWith("\n") && a[a.length - 1] === "") a.pop();
  if (newText != null && newText.endsWith("\n") && b[b.length - 1] === "") b.pop();
  if (oldText != null && !oldText.endsWith("\n") && a.length) {
    // keep
  }
  const hunks = diffLines(a, b);
  let out = "";
  out += `diff --git a/${pathName} b/${pathName}\n`;
  if (oldText == null) {
    out += `new file mode 100644\n`;
    out += `--- /dev/null\n`;
    out += `+++ b/${pathName}\n`;
  } else if (newText == null) {
    out += `deleted file mode 100644\n`;
    out += `--- a/${pathName}\n`;
    out += `+++ /dev/null\n`;
  } else {
    out += `--- a/${pathName}\n`;
    out += `+++ b/${pathName}\n`;
  }
  for (const h of hunks) {
    out += h;
  }
  return out;
}

function diffLines(a, b) {
  // Hunt-McIlroy / simple LCS DP for moderate files; for large, fall back to full replace
  if (a.length * b.length > 2_000_000) {
    let h = `@@ -1,${a.length} +1,${b.length} @@\n`;
    for (const line of a) h += `-` + line + `\n`;
    for (const line of b) h += `+` + line + `\n`;
    return [h];
  }
  const n = a.length;
  const m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      if (a[i] === b[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  // build ops
  const ops = []; // {t:'equal'|'del'|'ins', line}
  let i = 0,
    j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ t: "equal", line: a[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ t: "del", line: a[i++] });
    } else {
      ops.push({ t: "ins", line: b[j++] });
    }
  }
  while (i < n) ops.push({ t: "del", line: a[i++] });
  while (j < m) ops.push({ t: "ins", line: b[j++] });

  // group into hunks with 3 context
  const CONTEXT = 3;
  const changeIdx = [];
  ops.forEach((op, idx) => {
    if (op.t !== "equal") changeIdx.push(idx);
  });
  if (!changeIdx.length) return [];
  const ranges = [];
  let start = Math.max(0, changeIdx[0] - CONTEXT);
  let end = Math.min(ops.length, changeIdx[0] + CONTEXT + 1);
  for (let k = 1; k < changeIdx.length; k++) {
    const c = changeIdx[k];
    const s = Math.max(0, c - CONTEXT);
    const e = Math.min(ops.length, c + CONTEXT + 1);
    if (s <= end) end = e;
    else {
      ranges.push([start, end]);
      start = s;
      end = e;
    }
  }
  ranges.push([start, end]);

  const hunks = [];
  for (const [hs, he] of ranges) {
    let oldStart = 0,
      newStart = 0,
      oldCount = 0,
      newCount = 0;
    // compute line numbers
    let oi = 0,
      ni = 0;
    for (let x = 0; x < hs; x++) {
      if (ops[x].t === "equal") {
        oi++;
        ni++;
      } else if (ops[x].t === "del") oi++;
      else ni++;
    }
    oldStart = oi + 1;
    newStart = ni + 1;
    const body = [];
    for (let x = hs; x < he; x++) {
      const op = ops[x];
      if (op.t === "equal") {
        body.push(" " + op.line);
        oldCount++;
        newCount++;
      } else if (op.t === "del") {
        body.push("-" + op.line);
        oldCount++;
      } else {
        body.push("+" + op.line);
        newCount++;
      }
    }
    if (oldCount === 0) oldStart = 0;
    if (newCount === 0) newStart = 0;
    let h = `@@ -${oldStart},${oldCount} +${newStart},${newCount} @@\n`;
    h += body.join("\n");
    if (!h.endsWith("\n")) h += "\n";
    hunks.push(h);
  }
  return hunks;
}

function getBlobText(sha) {
  const obj = inflateObject(sha);
  if (obj.type !== "blob") throw new Error("not blob " + sha);
  // try utf8
  return obj.body.toString("utf8");
}

function short(sha) {
  return sha.slice(0, 7);
}

function main() {
  const parts = [];
  const headSha = resolveRef("HEAD");
  const originMain = resolveRef("refs/remotes/origin/main") || resolveRef("refs/heads/main");
  const headCommit = parseCommit(inflateObject(headSha).body);
  const headTree = loadTreeRecursive(headCommit.tree);
  const indexMap = parseIndex();
  const ignore = loadGitignore();

  // worktree files
  let wtFiles = listWorktreeFiles(repo);
  wtFiles = wtFiles.filter((f) => !matchIgnore(f, ignore));
  // also include index/head paths that may be deleted
  const allPaths = new Set([...wtFiles, ...indexMap.keys(), ...headTree.keys()]);

  // status
  const branch = (() => {
    const h = readText(path.join(gitDir, "HEAD")).trim();
    if (h.startsWith("ref:")) return h.replace("ref: refs/heads/", "");
    return "detached";
  })();
  let statusLines = [`## ${branch}`];
  if (originMain && originMain === headSha) {
    // same as origin/main — no ahead/behind notation without remote tracking details
  } else if (originMain) {
    statusLines[0] = `## ${branch}...origin/main`;
  }

  const unstaged = []; // worktree vs index
  const staged = []; // index vs HEAD
  const untracked = [];

  for (const pth of [...allPaths].sort()) {
    if (matchIgnore(pth, ignore)) continue;
    if (isIgnored(pth)) continue;
    const full = path.join(repo, pth);
    const inWt = exists(full) && fs.statSync(full).isFile();
    const idx = indexMap.get(pth);
    const head = headTree.get(pth);

    let wtSha = null;
    let wtBuf = null;
    if (inWt) {
      wtBuf = read(full);
      wtSha = blobHash(wtBuf);
    }

    // staged: index vs head
    if (idx && head) {
      if (idx.sha !== head.sha) staged.push(["M", pth]);
    } else if (idx && !head) {
      staged.push(["A", pth]);
    } else if (!idx && head) {
      // deleted in index?
      // if not in index but in head — could be deleted staged or never checked
      // if not in wt either and not in index -> deleted
      if (!inWt) staged.push(["D", pth]);
    }

    // unstaged: worktree vs index (or head if no index entry weird)
    if (idx) {
      if (!inWt) unstaged.push(["D", pth]);
      else if (wtSha !== idx.sha) unstaged.push(["M", pth]);
    } else if (!idx && !head && inWt) {
      untracked.push(pth);
    } else if (!idx && head && inWt) {
      // present in head and wt but not index — unusual
      if (wtSha !== head.sha) unstaged.push(["M", pth]);
    } else if (!idx && head && !inWt) {
      // deleted in worktree, still in head; if also not staged delete already handled
      // if still in index would be handled; if not in index means staged delete already
    }
  }

  // format status -sb like
  const flags = new Map(); // path -> XY
  function setFlag(pathName, x, y) {
    const cur = flags.get(pathName) || [" ", " "];
    if (x) cur[0] = x;
    if (y) cur[1] = y;
    flags.set(pathName, cur);
  }
  for (const [c, pth] of staged) setFlag(pth, c, null);
  for (const [c, pth] of unstaged) setFlag(pth, null, c);
  for (const pth of untracked) setFlag(pth, "?", "?");

  parts.push("===== 1. git status -sb =====");
  let st = statusLines[0] + "\n";
  for (const pth of [...flags.keys()].sort()) {
    const [x, y] = flags.get(pth);
    st += `${x}${y} ${pth}\n`;
  }
  parts.push(st);

  // log --oneline -10 from HEAD reflog / parent walk
  parts.push("===== 2. git log --oneline -10 =====");
  let logOut = "";
  let sha = headSha;
  for (let n = 0; n < 10 && sha; n++) {
    const c = parseCommit(inflateObject(sha).body);
    const msg = c.message.split("\n")[0];
    logOut += `${short(sha)} ${msg}\n`;
    sha = c.parents[0] || null;
  }
  parts.push(logOut);

  // diffs
  function textOrNull(buf) {
    if (buf == null) return null;
    // binary detect
    if (buf.includes(0)) return null;
    return buf.toString("utf8");
  }

  // 3. git diff HEAD  == worktree + unstaged vs HEAD (git diff HEAD shows unstaged+staged vs HEAD)
  parts.push("===== 3. git diff HEAD =====");
  let diffHead = "";
  for (const pth of [...allPaths].sort()) {
    if (matchIgnore(pth, ignore) || isIgnored(pth)) continue;
    const head = headTree.get(pth);
    const full = path.join(repo, pth);
    const inWt = exists(full) && fs.statSync(full).isFile();
    const headText = head ? getBlobText(head.sha) : null;
    const wtText = inWt ? textOrNull(read(full)) : null;
    const headShaF = head ? head.sha : null;
    const wtShaF = inWt ? blobHash(read(full)) : null;
    if (headShaF === wtShaF) continue;
    if (headShaF == null && wtShaF == null) continue;
    if (wtText == null && inWt) {
      diffHead += `diff --git a/${pth} b/${pth}\nBinary files differ\n`;
      continue;
    }
    diffHead += unifiedDiff(pth, headText, wtText);
  }
  parts.push(diffHead);

  // 4. git diff --cached == index vs HEAD
  parts.push("===== 4. git diff --cached =====");
  let diffCached = "";
  for (const pth of [...new Set([...indexMap.keys(), ...headTree.keys()])].sort()) {
    if (matchIgnore(pth, ignore) || isIgnored(pth)) continue;
    const head = headTree.get(pth);
    const idx = indexMap.get(pth);
    const headShaF = head ? head.sha : null;
    const idxShaF = idx ? idx.sha : null;
    if (headShaF === idxShaF) continue;
    const oldT = head ? getBlobText(head.sha) : null;
    const newT = idx ? getBlobText(idx.sha) : null;
    diffCached += unifiedDiff(pth, oldT, newT);
  }
  parts.push(diffCached);

  // 5. origin/main...HEAD  (merge-base ... HEAD) — if same sha empty
  parts.push("===== 5. git diff origin/main...HEAD =====");
  let diffRange = "";
  if (originMain && originMain !== headSha) {
    const om = parseCommit(inflateObject(originMain).body);
    const omTree = loadTreeRecursive(om.tree);
    const paths = new Set([...omTree.keys(), ...headTree.keys()]);
    for (const pth of [...paths].sort()) {
      const a = omTree.get(pth);
      const b = headTree.get(pth);
      const as = a ? a.sha : null;
      const bs = b ? b.sha : null;
      if (as === bs) continue;
      const oldT = a ? getBlobText(a.sha) : null;
      const newT = b ? getBlobText(b.sha) : null;
      diffRange += unifiedDiff(pth, oldT, newT);
    }
  }
  parts.push(diffRange);

  const empty3 = !diffHead.trim();
  const empty4 = !diffCached.trim();
  const empty5 = !diffRange.trim();
  if (empty3 && empty4 && empty5) {
    parts.push("ALL_EMPTY=yes");
    parts.push("===== 6. git show HEAD -p --stat =====");
    // show HEAD commit with stat and patch vs parent
    const c = headCommit;
    let show = "";
    show += `commit ${headSha}\n`;
    show += `Author: ${c.author}\n`;
    show += `Commit: ${c.committer}\n`;
    show += `\n    ${c.message.split("\n").join("\n    ")}\n\n`;
    const parentSha = c.parents[0];
    let parentTree = new Map();
    if (parentSha) {
      const pc = parseCommit(inflateObject(parentSha).body);
      parentTree = loadTreeRecursive(pc.tree);
    }
    const paths = new Set([...parentTree.keys(), ...headTree.keys()]);
    const stats = [];
    let patch = "";
    for (const pth of [...paths].sort()) {
      const a = parentTree.get(pth);
      const b = headTree.get(pth);
      const as = a ? a.sha : null;
      const bs = b ? b.sha : null;
      if (as === bs) continue;
      const oldT = a ? getBlobText(a.sha) : null;
      const newT = b ? getBlobText(b.sha) : null;
      // stat rough
      const oldLines = oldT != null ? oldT.split("\n").length : 0;
      const newLines = newT != null ? newT.split("\n").length : 0;
      stats.push(` ${pth} | ${Math.abs(newLines - oldLines)} +-\n`);
      patch += unifiedDiff(pth, oldT, newT);
    }
    show += `${stats.length} files changed\n`;
    show += stats.join("");
    show += `\n` + patch;
    parts.push(show);
  } else {
    parts.push("ALL_EMPTY=no");
    parts.push("SKIP_SHOW_HEAD: diffs not empty");
  }

  parts.push("===== END =====");
  const text = parts.join("\n") + "\n";
  fs.writeFileSync(outPath, text);
  process.stdout.write(`WROTE ${Buffer.byteLength(text)} bytes to ${outPath}\n`);
}

main();
