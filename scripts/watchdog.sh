#!/usr/bin/env bash
#tmux 실행 시 GPU 온도와 전력을 모니터링하여 일정 수준 이상이면 워크로드를 종료

MAX_TEMP=85
MAX_POWER=260
CHECK_INTERVAL=5

LOGDIR="$HOME/siyeon/exp_v3/logs"
LOGFILE="$LOGDIR/gpu_watchdog.log"

mkdir -p "$LOGDIR"

echo "GPU watchdog 시작 $(date)" | tee -a "$LOGFILE"

while true
do

    DATA=$(nvidia-smi \
        --query-gpu=temperature.gpu,power.draw \
        --format=csv,noheader,nounits)

    TEMP=$(echo $DATA | awk -F',' '{print $1}')
    POWER=$(echo $DATA | awk -F',' '{print $2}' | awk '{print int($1)}')

    echo "$(date) | temp=${TEMP}C power=${POWER}W" >> "$LOGFILE"

    if [ "$TEMP" -ge "$MAX_TEMP" ]; then
        echo "⚠ GPU 온도 초과: ${TEMP}C" | tee -a "$LOGFILE"
        pkill -f run_experiment.py
        echo "워크로드 종료됨 (temperature)" | tee -a "$LOGFILE"
        exit 1
    fi

    if [ "$POWER" -ge "$MAX_POWER" ]; then
        echo "⚠ GPU 전력 초과: ${POWER}W" | tee -a "$LOGFILE"
        pkill -f run_experiment.py
        echo "워크로드 종료됨 (power)" | tee -a "$LOGFILE"
        exit 1
    fi

    sleep $CHECK_INTERVAL

done