#!/usr/bin/env bash
# Run one module's tests with readable output.
#   scripts/check.sh 01            # module 01_data
#   scripts/check.sh 04 -k causal  # extra args go to pytest
#   scripts/check.sh debug d04_scaling
#   scripts/check.sh stretch
#   scripts/check.sh all
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

if [ $# -lt 1 ]; then
  echo "usage: scripts/check.sh <module-prefix|all|stretch|debug <exercise>> [pytest args]"; exit 2
fi

if [ "$1" = "all" ]; then
  shift; exec $PY -m pytest "$@"
fi

if [ "$1" = "debug" ]; then
  shift
  ex="${1:?usage: scripts/check.sh debug <exercise-prefix, e.g. d04_scaling>}"; shift
  dir=$(ls -d debug/${ex}* 2>/dev/null | head -1)
  [ -z "$dir" ] && { echo "no debug exercise matching '$ex'"; ls debug; exit 2; }
  echo "=== $dir ==="; exec $PY -m pytest "$dir" -v --tb=short -p no:cacheprovider "$@"
fi

arg="$1"; shift
dir=$(ls -d ${arg}* 2>/dev/null | grep -E '^(0[1-6]_|stretch)' | head -1)
[ -z "$dir" ] && { echo "no module matching '$arg'"; exit 2; }
echo "=== $dir ==="
exec $PY -m pytest "$dir" -v --tb=short "$@"
