#!/bin/bash
# Build a self-contained source tarball of this repository *including all submodules*
# at the exact commits pinned by the superproject, for upload to Zenodo.
#
# Why: GitHub release tarballs (and the Zenodo-GitHub integration) do NOT include
# submodules, and this artifact depends on SimCCL / ns-3-alibabacloud / aicb at
# commits that live on non-default branches. The tarball produced here builds offline.
#
# Usage: artifact/make_archive.sh [output.tar.gz]
#   (run from a clean checkout with `git submodule update --init --recursive` done)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -n "$(git status --porcelain --ignore-submodules=none)" ]; then
  echo "Working tree is not clean; commit or stash first so the archive matches a commit." >&2
  git status --short >&2
  exit 1
fi

REV="$(git rev-parse --short HEAD)"
PREFIX="SimAI-MCCL-artifact-${REV}"
OUT="${1:-$ROOT/${PREFIX}.tar.gz}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Archiving superproject at $(git rev-parse HEAD)"
git archive --format=tar --prefix="${PREFIX}/" HEAD > "$WORK/super.tar"

# Append every (recursive) submodule at its checked-out commit.
git submodule foreach --quiet --recursive \
  'echo "$displaypath $(git rev-parse HEAD)"' > "$WORK/submodules.txt"
while read -r path sha; do
  echo "  + $path @ $sha"
  (cd "$ROOT/$path" && git archive --format=tar --prefix="${PREFIX}/${path}/" "$sha") > "$WORK/sub.tar"
  tar --concatenate --file="$WORK/super.tar" "$WORK/sub.tar"
done < "$WORK/submodules.txt"

# Record provenance inside the archive.
mkdir -p "$WORK/${PREFIX}"
{
  echo "superproject $(git rev-parse HEAD) $(git describe --tags --always 2>/dev/null || true)"
  cat "$WORK/submodules.txt"
} > "$WORK/${PREFIX}/ARCHIVE_PROVENANCE.txt"
tar --append --file="$WORK/super.tar" -C "$WORK" "${PREFIX}/ARCHIVE_PROVENANCE.txt"

gzip -9 -c "$WORK/super.tar" > "$OUT"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"
