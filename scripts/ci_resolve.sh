#!/usr/bin/env bash
set -euo pipefail

# Resolve SwiftPM dependencies, healing a corrupt restored build cache.
#
# CI restores `.build` from an actions/cache archive. That archive carries
# SwiftPM's dependency git state: bare mirrors under `.build/repositories`,
# working copies under `.build/checkouts`, and the resolved pins in
# `.build/workspace-state.json`. A mirror can come back unusable — e.g. a
# remote-tracking ref naming an object the archive does not contain:
#
#   error: 'mlx-swift-lm': Couldn't check out revision '<sha>':
#       fatal: bad object refs/remotes/origin/<branch>
#
# git then aborts on any operation that scans refs in that mirror, so
# resolution can never succeed no matter what the manifest asks for. The
# failure is also self-perpetuating: the job dies before actions/cache's post
# step runs, so the bad archive is never replaced and every later run restores
# it again. (Observed 2026-08-16: all 13 Nightly E2E shards failed this way on
# an archive saved 2026-08-09.)
#
# Recovery is to discard the dependency git state and resolve from scratch.
# Compiled build products under `.build/<triple>/` are left alone, so a healed
# run still gets the incremental-build value the cache exists for.

if swift package resolve --disable-sandbox; then
  exit 0
fi

echo "::warning::SwiftPM resolve failed — discarding cached dependency git state and retrying"
rm -rf .build/repositories .build/checkouts .build/workspace-state.json
swift package resolve --disable-sandbox
