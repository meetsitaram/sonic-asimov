#!/usr/bin/env bash
# Play the idle-stand loop in MuJoCo with the SONIC policy holding the
# robot in place. Thin wrapper over play_relaxed_walk.sh --idle; all
# player flags pass through (e.g. --kinematic, --no-viewer, --record).
set -euo pipefail
exec "$(dirname "$0")/play_relaxed_walk.sh" --idle "$@"
