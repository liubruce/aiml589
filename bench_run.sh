#!/bin/sh
# qsub -t 1-2:1 bench_2.sh ed2_on_mujoco Hopper-v3
while read alg; do
  while read problem; do
      qsub -t 1-10:1 bench_2.sh $alg $problem
      # echo "show: $alg $problem ..."
  done < problems.txt
  echo "$alg"
done < algorithms.txt