#!/bin/bash
export FORCE_REBUILD=1
cd "$HOME"
nohup bash openblas-verify/run-verify-0.3.34.sh </dev/null >logs/openblas-034-nohup2.out 2>&1 &
echo $! > logs/openblas-034.pid
echo started $(cat logs/openblas-034.pid)
