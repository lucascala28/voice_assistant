#!/usr/bin/env bash
# Build a throwaway sandbox copy of the project, run the unit tests and the end-to-end harness in it, clean up.
#
#   sandbox/run.sh                    test the committed HEAD (what a push would send)
#   sandbox/run.sh --ref <commit>      test another commit/branch
#   sandbox/run.sh --working-tree     test uncommitted changes (tracked + untracked, minus ignored files)
#   sandbox/run.sh -- --only lamp_on --skip-audio   extra args after -- go to sandbox/harness.py
#
# Note: sandbox/harness.py and sandbox/scenarios.json always come from THIS checkout, not from the ref under test,
#       so the same end-to-end test is applied to every ref. The unit tests (tests/) are the ref's own, including
#       tests/test_sandbox_harness.py, which imports the ref's copy of sandbox/harness.py.
#
# Env: SANDBOX_ENV_FILE (default <repo>/.env, skipped if absent) supplies LLM_HOST/LLM_MODEL; variables already
#      exported (e.g. in CI) win over the file. An explicit SANDBOX_ENV_FILE that does not exist is an error (exit 2).
# SANDBOX_KEEP=1 keeps the sandbox dir;
#      SANDBOX_VENV (default ~/.cache/beemo-sandbox/venv) is the reused virtualenv, so your .venv is never touched.
# Exit: 0 pass, 1 test failure, 2 sandbox/infra failure.
set -uo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REF=HEAD
MODE=ref
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --working-tree) MODE=wt; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

# Drop the previous run's report now so an early exit can never leave a stale one to be read.
rm -rf "$REPO/.sandbox-last"

# Fail early on an explicitly requested env file that is missing; the default .env may legitimately be absent.
if [[ -n "${SANDBOX_ENV_FILE:-}" && ! -f "$SANDBOX_ENV_FILE" ]]; then
  echo "SANDBOX SETUP FAILED: SANDBOX_ENV_FILE=$SANDBOX_ENV_FILE does not exist" >&2
  exit 2
fi
ENV_FILE="${SANDBOX_ENV_FILE:-$REPO/.env}"
[[ -f "$ENV_FILE" ]] || ENV_FILE=""

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/beemo-sandbox.XXXXXX")"
SRC="$ROOT/src"
OUT="$ROOT/out"

# Remove the sandbox on every exit path (early failures included): worktree first, then the dir, then stale metadata.
cleanup() {
  if [[ "${SANDBOX_KEEP:-0}" == 1 ]]; then echo "sandbox kept at $ROOT"; return; fi
  [[ "$MODE" == ref ]] && git -C "$REPO" worktree remove --force "$SRC" >/dev/null 2>&1
  rm -rf "$ROOT"
  git -C "$REPO" worktree prune >/dev/null 2>&1
}
trap cleanup EXIT

if [[ "$MODE" == ref ]]; then
  git -C "$REPO" worktree add --detach --quiet "$SRC" "$REF" || exit 2
  echo "sandbox: $(git -C "$SRC" log -1 --format='%h %s') -> $SRC"
else
  mkdir -p "$SRC"
  (cd "$REPO" && git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ "$SRC/") || exit 2
  echo "sandbox: working tree of $REPO -> $SRC"
fi

# Untracked runtime assets the code expects next to it (never secrets beyond the LLM host in .env)
[[ -f "$REPO/data/beemo.onnx" && ! -f "$SRC/data/beemo.onnx" ]] && mkdir -p "$SRC/data" && cp "$REPO/data/beemo.onnx" "$SRC/data/"

export UV_PROJECT_ENVIRONMENT="${SANDBOX_VENV:-$HOME/.cache/beemo-sandbox/venv}"
cd "$SRC" || exit 2
echo "== uv sync (sandbox venv: $UV_PROJECT_ENVIRONMENT)"
uv sync --frozen --inexact --quiet || { echo "SANDBOX SETUP FAILED: uv sync"; exit 2; }

echo "== unit tests"
uv run --frozen --with pytest pytest -q tests
UNIT=$?

echo "== end-to-end harness"
uv run --frozen python "$REPO/sandbox/harness.py" --src "$SRC" --out "$OUT" \
  ${ENV_FILE:+--env-file "$ENV_FILE"} "$@"
E2E=$?

[[ "${SANDBOX_KEEP:-0}" == 1 ]] || { rm -rf "$REPO/.sandbox-last" && mkdir -p "$REPO/.sandbox-last" && cp -R "$OUT/." "$REPO/.sandbox-last/"; echo "recordings + report copied to $REPO/.sandbox-last"; }

echo "== result: unit=$UNIT e2e=$E2E"
(( E2E == 2 )) && exit 2
(( UNIT != 0 || E2E != 0 )) && exit 1
exit 0
