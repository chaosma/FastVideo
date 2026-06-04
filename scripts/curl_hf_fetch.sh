#!/bin/bash
# Direct-HTTPS fetch of missing files for an HF repo into the hub cache
# layout, bypassing `hf download` (which stalls on this box; see
# memory_prefetch_plan.md Phase 5/6 notes). LFS files are fetched with
# curl (4 in parallel), sha256-verified against the blob name, and
# symlinked into the snapshot. Usage: curl_hf_fetch.sh <org/repo>
set -euo pipefail
REPO="$1"
BASE="/workspace/.hf_home/hub/models--${REPO//\//--}"
COMMIT=$(cat "$BASE/refs/main")
SNAP="$BASE/snapshots/$COMMIT"
URL="https://huggingface.co/$REPO/resolve/$COMMIT"

# Partial blobs from the broken downloader are untrusted
rm -f "$BASE"/blobs/*.incomplete

TREE=$(curl -sSfL "https://huggingface.co/api/models/$REPO/tree/main?recursive=true")

fetch_one() {
  local path="$1" sha="$2"
  curl -sSfL --retry 5 --retry-all-errors -C - \
    -o "$BASE/blobs/$sha.incomplete" "$URL/$path"
  local got
  got=$(sha256sum "$BASE/blobs/$sha.incomplete" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then
    echo "SHA MISMATCH for $path: got $got want $sha"
    return 1
  fi
  mv "$BASE/blobs/$sha.incomplete" "$BASE/blobs/$sha"
  # relative symlink: 2 ups to leave snapshots/<commit>/, +1 per subdir.
  # e.g. top-level file -> ../../blobs/<sha>, foo/bar.bin -> ../../../blobs/<sha>
  local depth rel=""
  depth=$(tr -dc '/' <<<"$path" | wc -c)
  for _ in $(seq 1 $((depth + 2))); do rel+="../"; done
  mkdir -p "$SNAP/$(dirname "$path")"
  ln -sf "${rel}blobs/$sha" "$SNAP/$path"
  echo "verified + linked $path"
}
export -f fetch_one
export BASE SNAP URL

jq -r '.[] | select(.type=="file") | select(.lfs != null) | "\(.path)\t\(.lfs.oid)"' <<<"$TREE" |
while IFS=$'\t' read -r path sha; do
  if [ -e "$SNAP/$path" ]; then continue; fi
  printf '%s\0%s\0' "$path" "$sha"
done |
xargs -0 -n2 -P4 bash -c 'fetch_one "$0" "$1"'

# Non-LFS files should already exist; report any that don't
jq -r '.[] | select(.type=="file") | select(.lfs == null) | .path' <<<"$TREE" |
while read -r path; do
  [ -e "$SNAP/$path" ] || echo "WARNING: non-LFS file missing: $path"
done

broken=$(find "$SNAP" -type l ! -exec test -e {} \; -print | wc -l)
echo "broken links remaining: $broken"
[ "$broken" -eq 0 ] && echo "REPO $REPO COMPLETE AND VERIFIED"
