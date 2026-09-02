#!/usr/bin/env bash
# Safe TCM probe on Orange Pi RV2 / X60 (cluster 0). Requires root.
set -euo pipefail
HOST="${IME_HOST:-orangepi@192.168.1.37}"
DIR="${IME_DIR:-~/ime-bench}"
BLOCKS="${1:-1}"

ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes "$HOST" bash -s <<EOF
set -euo pipefail
cd $DIR
make board-tcm-probe
echo "--- sudo taskset -c 0 ./ime-tcm-probe $BLOCKS ---"
sudo taskset -c 0 ./ime-tcm-probe $BLOCKS
EOF
