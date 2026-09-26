#!/bin/bash
# helper: run a pipeline stage in the background with a log (usage: ./run_bg.sh module tag)
cd "$(dirname "$0")"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src nohup python3 -m "lts.$1" "$2" > "results/$1_$2.log" 2>&1 &
