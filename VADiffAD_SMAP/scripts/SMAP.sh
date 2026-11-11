#!/usr/bin/env bash
set -euo pipefail
# Run from repository root; this script does not download or preprocess SMAP.
python main.py --config config/smap.yaml --data data/SMAP.npz --output runs/SMAP --device "${1:-cpu}"
