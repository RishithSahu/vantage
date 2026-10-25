#!/bin/bash
# Starts the isolated dev page (port 8794) and, optionally, a car for it. Own store, own pipe name: nothing of the installed app is touched.
#   bench/dev_up.sh [car_model] [car args...]      e.g.  bench/dev_up.sh aster_ev --dataset data/world4/aster_ev/aster_ev_4yr.db --instant --speed 60
cd "$(dirname "$0")/.."
export VGPT_LINK=devlink PYTHONUTF8=1 VEHICLEGPT_HOME="$PWD/../_dev_home"
rm -rf "$VEHICLEGPT_HOME" "$PWD/../_dev_link"; mkdir -p "$VEHICLEGPT_HOME" "$PWD/../_dev_link"
nohup python bench/dev_web.py --port 8794 --linked --link-home "$PWD/../_dev_link" > logs_dev_web.txt 2>&1 &
sleep 25
if [ -n "$1" ]; then nohup python vehiclegpt.py car "$@" > logs_dev_car.txt 2>&1 & fi
