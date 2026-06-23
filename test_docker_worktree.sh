#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Docker integration test for the per-session worktree feature (agy bridge).
#
# This is the AUTHORITATIVE test: it proves, inside the REAL container against
# the REAL /app/slitled-platform clone, that each session gets a detached git
# worktree pinned to the CORRECT branch with no cross-branch mixing — by
# inspecting git's actual `worktree list`, independent of whether agy/codex
# auth is healthy (the worktree is created at spawn, before the agent is ready).
#
# WARNING: this rebuilds the image and RESTARTS the bridges container
# (gemini-cli-rest-bridges-1), briefly interrupting the live :8000/:8001 service.
#
# Usage:  bash test_docker_worktree.sh        # build + up + test
#         SKIP_BUILD=1 bash test_docker_worktree.sh   # test the already-running container
# ---------------------------------------------------------------------------
set -uo pipefail

C=gemini-cli-rest-bridges-1
REPO=/app/slitled-platform
BASE=http://localhost:8000
PASS=0; FAIL=0
ok(){ echo "  PASS: $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

dexec(){ docker exec "$C" "$@"; }
wlist(){ dexec git -C "$REPO" worktree list --porcelain 2>/dev/null; }
sha(){ dexec git -C "$REPO" rev-parse "$1^{commit}" 2>/dev/null; }
# safe_name() token the server derives for a key — computed by the REAL helper
# in the container, so the test stays in lockstep with the code.
token(){ dexec python -c "import worktree; print(worktree.safe_name('$1'))" 2>/dev/null; }

echo "=================================================================="
echo " Docker worktree integration test"
echo "=================================================================="

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "== build image + (re)start container (interrupts the live bridge) =="
  docker compose build && docker compose up -d || { echo "build/up failed"; exit 1; }
fi

echo "== wait for agy /health (proves worktree.py imported & both servers booted) =="
up=0
for _ in $(seq 1 60); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then up=1; break; fi
  sleep 2
done
[ "$up" = 1 ] && ok "/health responds — container booted with worktree.py present" \
              || { no "/health never came up (likely import/boot failure)"; echo "FAIL=$FAIL"; exit 1; }

# Branch tips we will pin to (bare names resolve to origin/* inside the server).
DEV_SHA=$(sha origin/dev); MAIN_SHA=$(sha origin/main)
echo "  origin/dev  = $DEV_SHA"
echo "  origin/main = $MAIN_SHA"
[ -n "$DEV_SHA" ] && [ -n "$MAIN_SHA" ] || { no "could not resolve origin/dev & origin/main in the clone"; }

echo "== drive two concurrent sessions on DIFFERENT branches =="
# Fire and forget — we assert on the worktrees git creates, not the agent reply.
curl -fsS -m 200 -X POST "$BASE/chat/qdev@dev"   -H 'content-type: application/json' \
     -d '{"prompt":"reply with the single word ok"}' >/dev/null 2>&1 &
curl -fsS -m 200 -X POST "$BASE/chat/qmain@main" -H 'content-type: application/json' \
     -d '{"prompt":"reply with the single word ok"}' >/dev/null 2>&1 &

QDEV_TOK=$(token "qdev@dev"); QMAIN_TOK=$(token "qmain@main")

echo "== poll until both worktrees exist (created early in spawn, before agy ready) =="
WL=""
for _ in $(seq 1 30); do
  WL=$(wlist)
  if echo "$WL" | grep -q "HEAD $DEV_SHA" && echo "$WL" | grep -q "HEAD $MAIN_SHA"; then break; fi
  sleep 2
done
echo "--- git worktree list --porcelain ---"; echo "$WL" | grep -E '^(worktree|HEAD|detached|branch)' | sed 's/^/    /'

# A worktree exists at each branch's commit.
echo "$WL" | grep -q "HEAD $DEV_SHA"  && ok "a worktree is checked out at origin/dev's commit"  || no "no worktree at origin/dev's commit"
echo "$WL" | grep -q "HEAD $MAIN_SHA" && ok "a worktree is checked out at origin/main's commit" || no "no worktree at origin/main's commit"

# Per-session: qdev's worktree is on dev's commit and NOT main's (no mixing).
QDEV_BLOCK=$(echo "$WL" | grep -A2 "worktree .*${QDEV_TOK}")
QMAIN_BLOCK=$(echo "$WL" | grep -A2 "worktree .*${QMAIN_TOK}")
echo "$QDEV_BLOCK"  | grep -q "HEAD $DEV_SHA"  && ok "qdev@dev  worktree pinned to origin/dev (correct branch)"   || no "qdev@dev worktree not at dev's commit"
echo "$QDEV_BLOCK"  | grep -q "HEAD $MAIN_SHA" && no "qdev@dev worktree LEAKED main's commit (MIXED!)"            || ok "qdev@dev  worktree is NOT at main's commit (no mixing)"
echo "$QMAIN_BLOCK" | grep -q "HEAD $MAIN_SHA" && ok "qmain@main worktree pinned to origin/main (correct branch)" || no "qmain@main worktree not at main's commit"
echo "$QDEV_BLOCK"  | grep -q "^detached" && ok "qdev@dev worktree is a DETACHED HEAD (no branch attached)" || no "qdev@dev worktree is not detached"

echo "== branchless /chat returns the handshake and creates NO worktree =="
BEFORE=$(wlist | grep -c '^worktree ')
RESP=$(curl -fsS -m 30 -X POST "$BASE/chat/plainnobranch" -H 'content-type: application/json' -d '{"prompt":"hi"}')
echo "$RESP" | grep -qi "which branch" && ok "branchless /chat returns the which-branch handshake" || no "branchless /chat did not return handshake: $RESP"
AFTER=$(wlist | grep -c '^worktree ')
[ "$BEFORE" = "$AFTER" ] && ok "branchless /chat created NO worktree ($BEFORE→$AFTER)" || no "branchless /chat changed the worktree count ($BEFORE→$AFTER)"

echo "== slash base routes (the {name:path} fix): @origin/dev is reachable, not 404 =="
CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST "$BASE/chat/qslash@origin/dev" -H 'content-type: application/json' -d '{"prompt":"reply ok"}')
[ "$CODE" != "404" ] && ok "slash key /chat/qslash@origin/dev routed (HTTP $CODE, not 404)" || no "slash key 404'd — {name:path} not in effect"

echo "== DELETE tears down a session's worktree =="
curl -fsS -m 30 -X DELETE "$BASE/chat/qdev@dev" >/dev/null 2>&1
sleep 3
wlist | grep -q "$QDEV_TOK" && no "qdev@dev worktree still present after DELETE" || ok "DELETE removed qdev@dev's worktree"

echo "=================================================================="
echo " RESULT: $PASS passed, $FAIL failed"
echo "=================================================================="
[ "$FAIL" = 0 ] || exit 1
