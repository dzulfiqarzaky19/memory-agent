#!/usr/bin/env python3
"""Pure-Python review-scope dump from .git + worktree (no git binary)."""
from __future__ import annotations

import hashlib
import os
import struct
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent
GIT = REPO / ".git"
OUT = REPO / "_git_review_scope.txt"

# temp / capture files to ignore in status
IGNORE_NAMES = {
    "_git_review_scope.txt",
    "_git_capture.cjs",
    "_git_capture_pure.cjs",
    "_capture_git_review.sh",
    "_rev_scope.cjs",
    "_rev_scope.py",
    "_dump_scope.cjs",
    "_cap.cmd",
    "_t.js",
    "_t2.js",
    "_hdr.bin",
    "_blob_hdr.bin",
    "_g.exe",
}


def read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def resolve_ref(name: str) -> str | None:
    if name == "HEAD":
        p = GIT / "HEAD"
    else:
        p = GIT / name
    if p.is_file():
        t = read_text(p).strip()
        if t.startswith("ref:"):
            return resolve_ref(t[4:].strip())
        return t
    packed = GIT / "packed-refs"
    if packed.is_file():
        for line in read_text(packed).splitlines():
            if not line or line[0] in "#^":
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1].strip() == name:
                return parts[0].strip()
    return None


def inflate_loose(sha: str) -> tuple[str, bytes]:
    p = GIT / "objects" / sha[:2] / sha[2:]
    if p.is_file():
        raw = zlib.decompress(read_bytes(p))
        nul = raw.index(b"\x00")
        header = raw[:nul].decode("ascii")
        typ, _size = header.split(" ")
        return typ, raw[nul + 1 :]
    return inflate_packed(sha)


def find_pack_offset(idx_path: Path, sha: str) -> int | None:
    buf = read_bytes(idx_path)
    if buf[:4] != b"\xfftOc":
        return None
    version = struct.unpack(">I", buf[4:8])[0]
    if version != 2:
        return None
    fanout_base = 8
    sha_b = bytes.fromhex(sha)
    first = sha_b[0]
    start = 0 if first == 0 else struct.unpack(">I", buf[fanout_base + (first - 1) * 4 : fanout_base + first * 4])[0]
    end = struct.unpack(">I", buf[fanout_base + first * 4 : fanout_base + first * 4 + 4])[0]
    nobj = struct.unpack(">I", buf[fanout_base + 255 * 4 : fanout_base + 255 * 4 + 4])[0]
    sha_table = fanout_base + 256 * 4
    lo, hi, found = start, end, -1
    while lo < hi:
        mid = (lo + hi) // 2
        off = sha_table + mid * 20
        cmpv = (sha_b > buf[off : off + 20]) - (sha_b < buf[off : off + 20])
        if cmpv == 0:
            found = mid
            break
        if cmpv < 0:
            hi = mid
        else:
            lo = mid + 1
    if found < 0:
        return None
    crc_table = sha_table + nobj * 20
    offset_table = crc_table + nobj * 4
    offset = struct.unpack(">I", buf[offset_table + found * 4 : offset_table + found * 4 + 4])[0]
    if offset & 0x80000000:
        large_table = offset_table + nobj * 4
        li = offset & 0x7FFFFFFF
        hi2, lo2 = struct.unpack(">II", buf[large_table + li * 8 : large_table + li * 8 + 8])
        return hi2 * 0x100000000 + lo2
    return offset


def extract_pack_obj(pack_path: Path, offset: int) -> tuple[str, bytes]:
    data = read_bytes(pack_path)
    i = offset
    c = data[i]
    i += 1
    typ_n = (c >> 4) & 7
    obj_size = c & 15
    shift = 4
    while c & 0x80:
        c = data[i]
        i += 1
        obj_size |= (c & 0x7F) << shift
        shift += 7
    types = {1: "commit", 2: "tree", 3: "blob", 4: "tag", 6: "ofs_delta", 7: "ref_delta"}
    typ = types.get(typ_n, str(typ_n))
    if typ_n in (6, 7):
        # delta — need base; handle ref_delta and ofs_delta
        if typ_n == 7:
            base_sha = data[i : i + 20].hex()
            i += 20
            delta = zlib.decompress(data[i:])
            base_typ, base = inflate_loose(base_sha)
            return base_typ, apply_delta(base, delta)
        # ofs_delta
        c = data[i]
        i += 1
        base_off = c & 0x7F
        while c & 0x80:
            c = data[i]
            i += 1
            base_off = ((base_off + 1) << 7) | (c & 0x7F)
        base_offset = offset - base_off
        delta = zlib.decompress(data[i:])
        base_typ, base = extract_pack_obj(pack_path, base_offset)
        return base_typ, apply_delta(base, delta)
    body = zlib.decompress(data[i:])
    return typ, body


def apply_delta(base: bytes, delta: bytes) -> bytes:
    def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
        result = 0
        shift = 0
        while True:
            c = buf[pos]
            pos += 1
            result |= (c & 0x7F) << shift
            if not (c & 0x80):
                break
            shift += 7
        return result, pos

    pos = 0
    _src_size, pos = read_varint(delta, pos)
    _dst_size, pos = read_varint(delta, pos)
    out = bytearray()
    while pos < len(delta):
        cmd = delta[pos]
        pos += 1
        if cmd & 0x80:
            cp_off = 0
            cp_size = 0
            if cmd & 0x01:
                cp_off = delta[pos]
                pos += 1
            if cmd & 0x02:
                cp_off |= delta[pos] << 8
                pos += 1
            if cmd & 0x04:
                cp_off |= delta[pos] << 16
                pos += 1
            if cmd & 0x08:
                cp_off |= delta[pos] << 24
                pos += 1
            if cmd & 0x10:
                cp_size = delta[pos]
                pos += 1
            if cmd & 0x20:
                cp_size |= delta[pos] << 8
                pos += 1
            if cmd & 0x40:
                cp_size |= delta[pos] << 16
                pos += 1
            if cp_size == 0:
                cp_size = 0x10000
            out.extend(base[cp_off : cp_off + cp_size])
        elif cmd != 0:
            out.extend(delta[pos : pos + cmd])
            pos += cmd
        else:
            raise ValueError("bad delta cmd 0")
    return bytes(out)


def inflate_packed(sha: str) -> tuple[str, bytes]:
    pack_dir = GIT / "objects" / "pack"
    if not pack_dir.is_dir():
        raise FileNotFoundError(sha)
    for idx in pack_dir.glob("*.idx"):
        off = find_pack_offset(idx, sha)
        if off is None:
            continue
        pack = idx.with_suffix(".pack")
        return extract_pack_obj(pack, off)
    raise FileNotFoundError(sha)


def inflate(sha: str) -> tuple[str, bytes]:
    return inflate_loose(sha)


def parse_commit(body: bytes) -> dict:
    text = body.decode("utf-8", errors="replace")
    lines = text.split("\n")
    out: dict = {"tree": None, "parents": [], "author": "", "committer": "", "message": ""}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "":
            i += 1
            break
        if line.startswith("tree "):
            out["tree"] = line[5:].strip()
        elif line.startswith("parent "):
            out["parents"].append(line[7:].strip())
        elif line.startswith("author "):
            out["author"] = line[7:]
        elif line.startswith("committer "):
            out["committer"] = line[10:]
        i += 1
    out["message"] = "\n".join(lines[i:]).rstrip("\n")
    return out


def parse_tree(body: bytes) -> list[tuple[str, str, str]]:
    entries = []
    i = 0
    while i < len(body):
        sp = body.index(b" ", i)
        mode = body[i:sp].decode("ascii")
        nul = body.index(b"\x00", sp + 1)
        name = body[sp + 1 : nul].decode("utf-8", errors="replace")
        sha = body[nul + 1 : nul + 21].hex()
        entries.append((mode, name, sha))
        i = nul + 21
    return entries


def load_tree(tree_sha: str, prefix: str = "") -> dict[str, tuple[str, str]]:
    typ, body = inflate(tree_sha)
    if typ != "tree":
        raise ValueError(f"not tree {tree_sha}")
    out: dict[str, tuple[str, str]] = {}
    for mode, name, sha in parse_tree(body):
        path = f"{prefix}/{name}" if prefix else name
        if mode == "40000":
            out.update(load_tree(sha, path))
        else:
            out[path] = (mode, sha)
    return out


def blob_hash(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def load_gitignore() -> list[str]:
    p = REPO / ".gitignore"
    if not p.is_file():
        return []
    return [ln.strip() for ln in read_text(p).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def match_ignore(rel: str, patterns: list[str]) -> bool:
    norm = rel.replace("\\", "/")
    if norm == ".git" or norm.startswith(".git/"):
        return True
    base = Path(norm).name
    if base in IGNORE_NAMES or base.startswith("_diff_"):
        return True
    if "/__pycache__/" in f"/{norm}/" or norm.endswith(".pyc"):
        return True
    if norm == "node_modules" or norm.startswith("node_modules/"):
        return True
    if norm == ".pytest_cache" or norm.startswith(".pytest_cache/"):
        return True
    if norm == ".codescratch" or norm.startswith(".codescratch/"):
        return True
    if norm == ".codestructure" or norm.startswith(".codestructure/"):
        return True
    for pat in patterns:
        neg = False
        if pat.startswith("!"):
            neg = True
            pat = pat[1:]
        if pat.endswith("/"):
            root = pat[:-1]
            if norm == root or norm.startswith(root + "/"):
                if not neg:
                    return True
            continue
        g = pat[1:] if pat.startswith("/") else pat
        # rough glob
        import re

        rx = (
            "^"
            + re.escape(g).replace(r"\*\*", ":::DD:::").replace(r"\*", "[^/]*").replace(":::DD:::", ".*")
            + "$"
        )
        if re.match(rx, norm) or re.match(rx, base):
            if not neg:
                return True
    return False


def list_worktree(dir_path: Path, prefix: str = "") -> list[str]:
    out: list[str] = []
    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: p.name)
    except OSError:
        return out
    for ent in entries:
        rel = f"{prefix}/{ent.name}" if prefix else ent.name
        if ent.name == ".git":
            continue
        if ent.is_dir():
            out.extend(list_worktree(ent, rel))
        elif ent.is_file():
            out.append(rel.replace("\\", "/"))
    return out


def parse_index() -> dict[str, dict]:
    p = GIT / "index"
    if not p.is_file():
        return {}
    buf = read_bytes(p)
    if buf[:4] != b"DIRC":
        raise ValueError("bad index")
    version, count = struct.unpack(">II", buf[4:12])
    offset = 12
    out: dict[str, dict] = {}
    for _ in range(count):
        mode = struct.unpack(">I", buf[offset + 24 : offset + 28])[0]
        size = struct.unpack(">I", buf[offset + 36 : offset + 40])[0]
        sha = buf[offset + 40 : offset + 60].hex()
        flags = struct.unpack(">H", buf[offset + 60 : offset + 62])[0]
        name_len = flags & 0xFFF
        if name_len == 0xFFF:
            start = offset + 62
            end = start
            while buf[end] != 0:
                end += 1
            name = buf[start:end].decode("utf-8", errors="replace")
            entry_len = ((62 + (end - start) + 1 + 7) // 8) * 8
        else:
            name = buf[offset + 62 : offset + 62 + name_len].decode("utf-8", errors="replace")
            entry_len = ((62 + name_len + 1 + 7) // 8) * 8
        out[name.replace("\\", "/")] = {"sha": sha, "mode": mode, "size": size}
        offset += entry_len
        if version >= 3 and (flags & 0x4000):
            pass
    return out


def get_blob_text(sha: str) -> str | None:
    typ, body = inflate(sha)
    if typ != "blob":
        raise ValueError("not blob")
    if b"\x00" in body:
        return None
    return body.decode("utf-8", errors="replace")


def diff_lines(a: list[str], b: list[str]) -> list[str]:
    if len(a) * len(b) > 2_000_000:
        h = [f"@@ -1,{len(a)} +1,{len(b)} @@\n"]
        h.extend(f"-{ln}\n" for ln in a)
        h.extend(f"+{ln}\n" for ln in b)
        return h
    n, m = len(a), len(b)
    # LCS DP lengths
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai = a[i]
        row = dp[i]
        nrow = dp[i + 1]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                row[j] = nrow[j + 1] + 1
            else:
                row[j] = max(nrow[j], row[j + 1])
    ops: list[tuple[str, str]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            ops.append(("equal", a[i]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            ops.append(("del", a[i]))
            i += 1
        else:
            ops.append(("ins", b[j]))
            j += 1
    while i < n:
        ops.append(("del", a[i]))
        i += 1
    while j < m:
        ops.append(("ins", b[j]))
        j += 1
    change_idx = [k for k, op in enumerate(ops) if op[0] != "equal"]
    if not change_idx:
        return []
    CTX = 3
    ranges: list[tuple[int, int]] = []
    start = max(0, change_idx[0] - CTX)
    end = min(len(ops), change_idx[0] + CTX + 1)
    for c in change_idx[1:]:
        s = max(0, c - CTX)
        e = min(len(ops), c + CTX + 1)
        if s <= end:
            end = e
        else:
            ranges.append((start, end))
            start, end = s, e
    ranges.append((start, end))
    hunks: list[str] = []
    for hs, he in ranges:
        oi = ni = 0
        for x in range(hs):
            t = ops[x][0]
            if t == "equal":
                oi += 1
                ni += 1
            elif t == "del":
                oi += 1
            else:
                ni += 1
        old_start, new_start = oi + 1, ni + 1
        body_lines: list[str] = []
        old_count = new_count = 0
        for x in range(hs, he):
            t, line = ops[x]
            if t == "equal":
                body_lines.append(" " + line)
                old_count += 1
                new_count += 1
            elif t == "del":
                body_lines.append("-" + line)
                old_count += 1
            else:
                body_lines.append("+" + line)
                new_count += 1
        if old_count == 0:
            old_start = 0
        if new_count == 0:
            new_start = 0
        h = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@\n"
        h += "\n".join(body_lines)
        if not h.endswith("\n"):
            h += "\n"
        hunks.append(h)
    return hunks


def split_lines(text: str | None) -> list[str]:
    if text is None:
        return []
    if text == "":
        return []
    parts = text.split("\n")
    if text.endswith("\n") and parts and parts[-1] == "":
        parts.pop()
    return parts


def unified_diff(path: str, old_text: str | None, new_text: str | None) -> str:
    a = split_lines(old_text)
    b = split_lines(new_text)
    out = [f"diff --git a/{path} b/{path}\n"]
    if old_text is None:
        out.append("new file mode 100644\n")
        out.append("--- /dev/null\n")
        out.append(f"+++ b/{path}\n")
    elif new_text is None:
        out.append("deleted file mode 100644\n")
        out.append(f"--- a/{path}\n")
        out.append("+++ /dev/null\n")
    else:
        out.append(f"--- a/{path}\n")
        out.append(f"+++ b/{path}\n")
    out.extend(diff_lines(a, b))
    return "".join(out)


def short(sha: str) -> str:
    return sha[:7]


def main() -> int:
    head_sha = resolve_ref("HEAD")
    if not head_sha:
        print("no HEAD", file=sys.stderr)
        return 1
    origin_main = resolve_ref("refs/remotes/origin/main") or resolve_ref("refs/heads/main")
    head_commit = parse_commit(inflate(head_sha)[1])
    head_tree = load_tree(head_commit["tree"])
    index_map = parse_index()
    ignore = load_gitignore()

    wt_files = [f for f in list_worktree(REPO) if not match_ignore(f, ignore)]
    all_paths = set(wt_files) | set(index_map) | set(head_tree)

    head_txt = read_text(GIT / "HEAD").strip()
    branch = head_txt.replace("ref: refs/heads/", "") if head_txt.startswith("ref:") else "detached"
    status0 = f"## {branch}"
    if origin_main and origin_main == head_sha:
        status0 = f"## {branch}...origin/{branch}" if branch == "main" else f"## {branch}"
    elif origin_main:
        status0 = f"## {branch}...origin/main"

    flags: dict[str, list[str]] = {}

    def set_flag(path: str, x: str | None, y: str | None) -> None:
        cur = flags.get(path, [" ", " "])
        if x:
            cur[0] = x
        if y:
            cur[1] = y
        flags[path] = cur

    for path in sorted(all_paths):
        if match_ignore(path, ignore):
            continue
        full = REPO / path
        in_wt = full.is_file()
        idx = index_map.get(path)
        head = head_tree.get(path)
        wt_sha = blob_hash(full.read_bytes()) if in_wt else None

        if idx and head:
            if idx["sha"] != head[1]:
                set_flag(path, "M", None)
        elif idx and not head:
            set_flag(path, "A", None)
        elif not idx and head and not in_wt:
            set_flag(path, "D", None)

        if idx:
            if not in_wt:
                set_flag(path, None, "D")
            elif wt_sha != idx["sha"]:
                set_flag(path, None, "M")
        elif not idx and not head and in_wt:
            set_flag(path, "?", "?")
        elif not idx and head and in_wt and wt_sha != head[1]:
            set_flag(path, None, "M")

    parts: list[str] = []
    parts.append("===== 1. git status -sb =====")
    st = status0 + "\n"
    for path in sorted(flags):
        x, y = flags[path]
        st += f"{x}{y} {path}\n"
    parts.append(st)

    parts.append("===== 2. git log --oneline -10 =====")
    log_out = ""
    sha: str | None = head_sha
    for _ in range(10):
        if not sha:
            break
        c = parse_commit(inflate(sha)[1])
        msg = c["message"].split("\n")[0]
        log_out += f"{short(sha)} {msg}\n"
        sha = c["parents"][0] if c["parents"] else None
    parts.append(log_out)

    # 3. worktree vs HEAD
    parts.append("===== 3. git diff HEAD =====")
    diff_head = ""
    for path in sorted(all_paths):
        if match_ignore(path, ignore):
            continue
        head = head_tree.get(path)
        full = REPO / path
        in_wt = full.is_file()
        head_sha_f = head[1] if head else None
        wt_bytes = full.read_bytes() if in_wt else None
        wt_sha_f = blob_hash(wt_bytes) if wt_bytes is not None else None
        if head_sha_f == wt_sha_f:
            continue
        head_text = get_blob_text(head_sha_f) if head_sha_f else None
        if in_wt:
            if b"\x00" in wt_bytes:
                diff_head += f"diff --git a/{path} b/{path}\nBinary files differ\n"
                continue
            wt_text = wt_bytes.decode("utf-8", errors="replace")
        else:
            wt_text = None
        diff_head += unified_diff(path, head_text, wt_text)
    parts.append(diff_head)

    # 4. index vs HEAD
    parts.append("===== 4. git diff --cached =====")
    diff_cached = ""
    for path in sorted(set(index_map) | set(head_tree)):
        if match_ignore(path, ignore):
            continue
        head = head_tree.get(path)
        idx = index_map.get(path)
        hs = head[1] if head else None
        is_ = idx["sha"] if idx else None
        if hs == is_:
            continue
        old_t = get_blob_text(hs) if hs else None
        new_t = get_blob_text(is_) if is_ else None
        diff_cached += unified_diff(path, old_t, new_t)
    parts.append(diff_cached)

    # 5. origin/main...HEAD
    parts.append("===== 5. git diff origin/main...HEAD =====")
    diff_range = ""
    if origin_main and origin_main != head_sha:
        om = parse_commit(inflate(origin_main)[1])
        om_tree = load_tree(om["tree"])
        for path in sorted(set(om_tree) | set(head_tree)):
            a = om_tree.get(path)
            b = head_tree.get(path)
            as_ = a[1] if a else None
            bs = b[1] if b else None
            if as_ == bs:
                continue
            old_t = get_blob_text(as_) if as_ else None
            new_t = get_blob_text(bs) if bs else None
            diff_range += unified_diff(path, old_t, new_t)
    parts.append(diff_range)

    empty3 = not diff_head.strip()
    empty4 = not diff_cached.strip()
    empty5 = not diff_range.strip()
    if empty3 and empty4 and empty5:
        parts.append("ALL_EMPTY=yes")
        parts.append("===== 6. git show HEAD -p --stat =====")
        c = head_commit
        show = f"commit {head_sha}\n"
        show += f"Author: {c['author']}\n"
        show += f"Commit: {c['committer']}\n\n"
        for ln in c["message"].split("\n"):
            show += f"    {ln}\n"
        show += "\n"
        parent_tree: dict[str, tuple[str, str]] = {}
        if c["parents"]:
            pc = parse_commit(inflate(c["parents"][0])[1])
            parent_tree = load_tree(pc["tree"])
        stats: list[str] = []
        patch = ""
        paths = sorted(set(parent_tree) | set(head_tree))
        for path in paths:
            a = parent_tree.get(path)
            b = head_tree.get(path)
            as_ = a[1] if a else None
            bs = b[1] if b else None
            if as_ == bs:
                continue
            old_t = get_blob_text(as_) if as_ else None
            new_t = get_blob_text(bs) if bs else None
            ol = len(split_lines(old_t))
            nl = len(split_lines(new_t))
            stats.append(f" {path} | {abs(nl - ol)} +-\n")
            patch += unified_diff(path, old_t, new_t)
        show += f"{len(stats)} files changed\n"
        show += "".join(stats)
        show += "\n" + patch
        parts.append(show)
    else:
        parts.append("ALL_EMPTY=no")
        parts.append("SKIP_SHOW_HEAD: diffs not empty")

    parts.append("===== END =====")
    text = "\n".join(parts) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(f"WROTE {len(text.encode('utf-8'))} bytes to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
