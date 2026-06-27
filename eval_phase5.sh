#!/usr/bin/env bash
set -e

SEED=${1:-42}
MODEL_DIR="models/phase5/v3/seed${SEED}"
mkdir -p results/phase5

for STAGE in 5b 5c 5d 5e; do
    RUN="phase5_${STAGE}_v3_seed${SEED}"
    OUT="results/phase5/${RUN}_heldout.json"

    if [ -f "$OUT" ]; then
        echo "Skipping $STAGE (already done: $OUT)"
        continue
    fi

    CKPT="${MODEL_DIR}/${RUN}/best_model.zip"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: $CKPT not found — did training finish?"; exit 1
    fi

    .venv/bin/python evaluate.py "$CKPT" \
        --tracks held-out --episodes 20 \
        --output "$OUT"

    echo "Evaluated $STAGE → $OUT"
done

echo "All evaluations done for seed ${SEED}."
