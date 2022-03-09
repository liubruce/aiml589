#!/bin/sh
while read alg; do
  while read problem; do
       # qsub qsub -t 1-10:1 bench_2.sh $alg $problem
       echo "show: $alg $problem ..."
  done < problems.txt
done < algorithms.txt