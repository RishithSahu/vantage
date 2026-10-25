#!/bin/bash
# usage: bench/eval_round.sh NAME   (adapter models/meaning-lora-NAME.gguf): full evaluation + gate
cd "$(dirname "$0")/.."
NAME=$1
log() { echo "[$(date +%H:%M:%S)] $1" >> logs_after_train.txt; }
log "$NAME eval start"
PYTHONUTF8=1 python bench/eval_parallel.py models/meaning-lora-$NAME.gguf $NAME --procs 3 --threads 4 > logs_eval_${NAME}_main.txt 2>&1
FROZEN=$(python - <<'PY'
import sys; sys.path.insert(0, "bench/meaning_eval")
import eval_r10 as E; print(" ".join(E.FROZEN))
PY
)
PYTHONUTF8=1 python bench/eval_parallel.py models/meaning-lora-$NAME.gguf ${NAME}u --no-hint --default-vehicle none --no-vehicle-tag --procs 3 --threads 4 --sets $FROZEN > logs_eval_${NAME}_unknown_main.txt 2>&1
python bench/merge_preds.py ${NAME}_all ${NAME}u ${NAME} >> logs_eval_${NAME}_unknown_main.txt 2>&1
python bench/meaning_eval/gate.py ${NAME}_all frame2w_all > results/gate_${NAME}.txt 2>&1
log "$NAME eval + gate done"
