#!/bin/bash
# waits for the GPU (mix fine-tune converted), then trains round 11: the round-10 adapter fine-tuned one epoch on the round-11 data
cd "$(dirname "$0")/.."
while [ ! -f models/meaning-lora-frame2r10_mix.gguf ]; do sleep 30; done
sleep 20
echo "[$(date +%H:%M:%S)] starting r11 training" >> logs_after_train.txt
VGPT_FRAME=2 VGPT_VEHICLE_HINT=1 PYTHONUTF8=1 python training/meaning/train_meaning.py --data training/meaning/data/frame2_r11 --out training/out/meaning_frame2r11 --epochs 1 --bs 16 --lr 1e-4 --rank 32 --alpha 64 --seed 3 --init-adapter training/out/meaning_frame2r10/adapter.safetensors > logs_train_r11.txt 2>&1
echo "[$(date +%H:%M:%S)] r11 training exit $?" >> logs_after_train.txt
python -m training.convert_lora_gguf training/out/meaning_frame2r11 --out models/meaning-lora-frame2r11.gguf > logs_convert_frame2r11.txt 2>&1
echo "[$(date +%H:%M:%S)] r11 converted" >> logs_after_train.txt
