#!/usr/bin/env bash
set -e

SEED=${1:-42}
MODEL_DIR="models/phase5/v3/seed${SEED}"
PREV="models/phase4/v2/seed${SEED}/phase4_v2_seed${SEED}/best_model.zip"

for stage_cfg in "5b:200000" "5c:750000" "5d:500000" "5e:1500000"; do
    STAGE="${stage_cfg%%:*}"
    STEPS="${stage_cfg##*:}"
    RUN="phase5_${STAGE}_v3_seed${SEED}"

    CMD=(
        .venv/bin/python train.py
        --curriculum-stage "$STAGE"
        --track-mode random
        --seed "$SEED"
        --load-checkpoint "$PREV"
        --timesteps "$STEPS"
        --n-envs 8
        --run-name "$RUN"
        --log-dir logs/phase5
        --model-dir "$MODEL_DIR"
        --no-progress-bar
    )
    if [ "$STAGE" = "5e" ]; then
        CMD+=(--pool-dir "${MODEL_DIR}/pool")
    fi

    "${CMD[@]}"

    PREV="${MODEL_DIR}/${RUN}/best_model.zip"
    if [ ! -f "$PREV" ]; then
        echo "ERROR: $PREV not found after $STAGE — aborting."
        exit 1
    fi
    echo "Stage $STAGE done."
done

echo "All stages complete for seed ${SEED}."
