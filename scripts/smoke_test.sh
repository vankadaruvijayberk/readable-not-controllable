#!/usr/bin/env bash
# Runs every experiment end to end on CPU with a 135M model and about 100 questions.
# Takes about five minutes. The numbers it prints mean nothing; it only checks that each stage runs.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/smoke.yaml
MODEL=smollm2-135m
OUT=${1:-outputs/smoke}

python scripts/collect_activations.py --config $CONFIG --model $MODEL --out $OUT/activations
python scripts/run_detection.py --config $CONFIG --model $MODEL --activations-dir $OUT/activations \
    --work-dir $OUT/work --out $OUT/detection
python scripts/run_ksweep.py --config $CONFIG --model $MODEL --activations-dir $OUT/activations \
    --work-dir $OUT/work --detection $OUT/detection --out $OUT/ksweep
for DATASET in medqa_usmle medmcqa; do
    python scripts/run_interventions.py --config $CONFIG --model $MODEL --dataset $DATASET \
        --detection $OUT/detection --out $OUT/interventions
done
python scripts/analyze_interventions.py --config $CONFIG --interventions $OUT/interventions \
    --ksweep $OUT/ksweep --out $OUT/analysis
python scripts/make_figures.py --ksweep $OUT/ksweep --out $OUT/figures
echo "smoke test finished: $OUT"
