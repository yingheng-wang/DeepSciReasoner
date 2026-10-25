#!/bin/bash
# Download the prepared benchmarks from Hugging Face into DATA/.
#
#   bash scripts/download_data.sh                      # all six benchmarks (~500 MB compressed)
#   bash scripts/download_data.sh edos_caltech phdos   # selected benchmarks
#
# Each benchmark is unpacked to DATA/<benchmark>/{manifest.csv,split.pkl,graphs/}.
# Set HF_TOKEN to authenticate if needed; set HF_REPO to use a mirror.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

HF_REPO="${HF_REPO:-yingheng/DeepSciReasoner}"
BASE_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

if [[ $# -gt 0 ]]; then benches=("$@"); else read -r -a benches <<< "$BENCHMARKS"; fi
for b in "${benches[@]}"; do set_benchmark "$b"; done   # validate names before downloading

if command -v sha256sum >/dev/null; then sha256() { sha256sum "$@"; }; else sha256() { shasum -a 256 "$@"; }; fi
fetch() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl -fL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" -o "$2" "$BASE_URL/$1"
    else
        curl -fL --retry 3 -o "$2" "$BASE_URL/$1"
    fi
}

mkdir -p DATA
fetch SHA256SUMS DATA/SHA256SUMS
for b in "${benches[@]}"; do
    echo "==> $b"
    mkdir -p "DATA/$b"
    for f in manifest.csv split.pkl graphs.tar.gz; do
        fetch "$b/$f" "DATA/$b/$f"
    done
    (cd DATA && grep " $b/" SHA256SUMS | sha256 -c -)
    rm -rf "DATA/$b/graphs"
    tar -xzf "DATA/$b/graphs.tar.gz" -C "DATA/$b"
    rm "DATA/$b/graphs.tar.gz"
    echo "$b: $(find "DATA/$b/graphs" -name '*.chkpt' | wc -l | tr -d ' ') graphs"
done
