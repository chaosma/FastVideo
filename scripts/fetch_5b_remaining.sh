#!/bin/bash
# Direct-HTTPS fetch of the 3 missing 5B blobs, with sha256 verification,
# placed into the HF hub cache layout. Bypasses the broken `hf download` path.
set -euo pipefail
BASE=/workspace/.hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers
SNAP=$BASE/snapshots/b8fff7315c768468a5333511427288870b2e9635
URL=https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main

declare -A FILES=(
  [text_encoder/model-00001-of-00003.safetensors]=a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58
  [transformer/diffusion_pytorch_model-00001-of-00005.safetensors]=511bec832a201caa410d09c5ce7dbbf8ad2708c345d82038f684fc74cce982be
  [transformer/diffusion_pytorch_model-00004-of-00005.safetensors]=a331121771790939678db6f585553fd5184609f7d02593c699a4d241b0d834c5
)

# Fresh start: prior partials are untrusted
rm -f "$BASE"/blobs/*.incomplete

pids=()
for f in "${!FILES[@]}"; do
  sha=${FILES[$f]}
  ( curl -sSfL --retry 5 --retry-all-errors -C - \
      -o "$BASE/blobs/$sha.incomplete" "$URL/$f" \
    && echo "downloaded $f" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for f in "${!FILES[@]}"; do
  sha=${FILES[$f]}
  got=$(sha256sum "$BASE/blobs/$sha.incomplete" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then
    echo "SHA MISMATCH for $f: got $got want $sha"; exit 1
  fi
  mv "$BASE/blobs/$sha.incomplete" "$BASE/blobs/$sha"
  ln -sf "../../../blobs/$sha" "$SNAP/$f"
  echo "verified + linked $f"
done
echo "ALL 3 FILES COMPLETE AND VERIFIED"
