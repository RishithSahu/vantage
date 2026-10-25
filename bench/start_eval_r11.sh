#!/bin/bash
# waits for the round-11 adapter, then evaluates it on every set (three processes at once, car type told to the model)
cd "$(dirname "$0")/.."
while ! grep -q "r11 converted" logs_after_train.txt; do sleep 30; done
sleep 5
echo "[$(date +%H:%M:%S)] r11 eval start" >> logs_after_train.txt
PYTHONUTF8=1 python bench/eval_parallel.py models/meaning-lora-frame2r11.gguf frame2r11 --procs 3 --threads 4 > logs_eval_frame2r11_main.txt 2>&1
echo "[$(date +%H:%M:%S)] r11 eval done" >> logs_after_train.txt
