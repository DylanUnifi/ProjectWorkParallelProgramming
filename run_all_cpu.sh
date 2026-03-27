#!/bin/bash

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all")

cpu_jobs=0
max_cpu_jobs=6

echo "Starting classical CPU benchmark..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      
      echo "Launching classical run: $ds | $diff | $size"
      
      if [ "$size" == "all" ]; then
        SUBSET_ARG=""
      else
        SUBSET_ARG="--train-subset $size"
      fi
      
      (time python3 train_svm_classical.py \
        --config configs/${ds}_${diff}.yaml \
        $SUBSET_ARG \
        --pca-components 16 \
        --kernel rbf) 2>&1 | tee -a log_${ds}_${diff}_classical_${size}.txt &
      
      cpu_jobs=$((cpu_jobs + 1))
      
      if [ $cpu_jobs -eq $max_cpu_jobs ]; then
        echo "CPU concurrency limit reached. Waiting..."
        wait
        cpu_jobs=0
      fi
      
    done
  done
done

wait
echo "All classical jobs completed."
