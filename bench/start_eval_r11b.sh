#!/bin/bash
# after the main r11 evaluation: the frozen sets again with the car type unknown (no hint line), the like-for-like comparison with frame2w, then the merged table and the gate
cd "$(dirname "$0")/.."
while ! grep -q "r11 eval done" logs_after_train.txt; do sleep 30; done
echo "[$(date +%H:%M:%S)] r11 unknown eval start" >> logs_after_train.txt
FROZEN=$(python - <<'PY'
import sys; sys.path.insert(0, "bench/meaning_eval")
import eval_r10 as E; print(" ".join(E.FROZEN))
PY
)
PYTHONUTF8=1 python bench/eval_parallel.py models/meaning-lora-frame2r11.gguf frame2r11u --no-hint --default-vehicle none --no-vehicle-tag --procs 3 --threads 4 --sets $FROZEN > logs_eval_frame2r11_unknown_main.txt 2>&1
python bench/merge_preds.py frame2r11_all frame2r11u frame2r11 >> logs_eval_frame2r11_unknown_main.txt 2>&1
python bench/meaning_eval/gate.py frame2r11_all frame2w_all > results/gate_frame2r11.txt 2>&1
echo "[$(date +%H:%M:%S)] r11 unknown eval + gate done" >> logs_after_train.txt
