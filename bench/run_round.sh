#!/bin/bash
# usage: bench/run_round.sh NAME DATA_DIR INIT_ADAPTER_DIR LR [EPOCHS]
#   trains NAME (fine-tuned from INIT on DATA with learning rate LR), converts it, evaluates it on every set (car type told) and on the frozen sets again with the car type unknown, merges, and prints the adoption gate against frame2w
cd "$(dirname "$0")/.."
NAME=$1; DATA=$2; INIT=$3; LR=$4; EPOCHS=${5:-1}
log() { echo "[$(date +%H:%M:%S)] $1" >> logs_after_train.txt; }
log "$NAME training start"
VGPT_FRAME=2 VGPT_VEHICLE_HINT=1 PYTHONUTF8=1 python training/meaning/train_meaning.py --data $DATA --out training/out/meaning_$NAME --epochs $EPOCHS --bs 16 --lr $LR --rank 32 --alpha 64 --seed 4 --init-adapter $INIT/adapter.safetensors > logs_train_$NAME.txt 2>&1
log "$NAME training exit $?"
VGPT_FRAME=2 PYTHONUTF8=1 python -m training.convert_lora_gguf training/out/meaning_$NAME --out models/meaning-lora-$NAME.gguf > logs_convert_$NAME.txt 2>&1
log "$NAME converted"
bash bench/eval_round.sh $NAME
