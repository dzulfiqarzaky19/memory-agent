#!/usr/bin/env bash
# Hook-safe capture: name-only first, then per-path diffs; full dump to scratch file.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OUT="_git_review_scope.txt"
: > "$OUT"

{
  echo "===== 1. git status -sb ====="
  git status -sb
  echo
  echo "===== 2. git log --oneline -10 ====="
  git log --oneline -10
  echo
} >> "$OUT" 2>&1

echo "===== 3. git diff HEAD =====" >> "$OUT"
mapfile -t HEAD_FILES < <(git diff --name-only HEAD 2>/dev/null || true)
if [ ${#HEAD_FILES[@]} -eq 0 ] || [ -z "${HEAD_FILES[0]:-}" ]; then
  echo >> "$OUT"
else
  for f in "${HEAD_FILES[@]}"; do
    [ -n "$f" ] || continue
    git diff HEAD -- "$f" >> "$OUT" 2>&1 || true
  done
fi
echo >> "$OUT"

echo "===== 4. git diff --cached =====" >> "$OUT"
mapfile -t CACHED_FILES < <(git diff --cached --name-only 2>/dev/null || true)
if [ ${#CACHED_FILES[@]} -eq 0 ] || [ -z "${CACHED_FILES[0]:-}" ]; then
  echo >> "$OUT"
else
  for f in "${CACHED_FILES[@]}"; do
    [ -n "$f" ] || continue
    git diff --cached -- "$f" >> "$OUT" 2>&1 || true
  done
fi
echo >> "$OUT"

echo "===== 5. git diff origin/main...HEAD =====" >> "$OUT"
mapfile -t RANGE_FILES < <(git diff --name-only origin/main...HEAD 2>/dev/null || true)
if [ ${#RANGE_FILES[@]} -eq 0 ] || [ -z "${RANGE_FILES[0]:-}" ]; then
  echo >> "$OUT"
else
  for f in "${RANGE_FILES[@]}"; do
    [ -n "$f" ] || continue
    git diff origin/main...HEAD -- "$f" >> "$OUT" 2>&1 || true
  done
fi
echo >> "$OUT"

# If 3+4+5 empty of diff content, also show HEAD with pathspecs
if ! grep -q '^diff --git' "$OUT"; then
  {
    echo "===== 6. git show HEAD -p --stat (diffs empty) ====="
    mapfile -t SHOW_FILES < <(git show --name-only --pretty=format: HEAD 2>/dev/null | sed '/^$/d' || true)
    git show HEAD --stat --pretty=fuller
    echo
    if [ ${#SHOW_FILES[@]} -gt 0 ] && [ -n "${SHOW_FILES[0]:-}" ]; then
      for f in "${SHOW_FILES[@]}"; do
        [ -n "$f" ] || continue
        git show HEAD -p --pretty=format: -- "$f" || true
      done
    else
      # initial commit / empty tree edge: still try full show via pipe to file (hook allows dump to scratch)
      git show HEAD -p --stat | head -c 5000000 || true
    fi
  } >> "$OUT" 2>&1
fi

echo "WROTE $OUT"
wc -c "$OUT"
