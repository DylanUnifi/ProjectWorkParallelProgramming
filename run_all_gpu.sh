#!/bin/bash

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all")

gpu_id=0
max_gpus=8

echo "Starting quantum GPU batch on $max_gpus GPUs..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      
      echo "Launching $ds | $diff | $size on GPU $gpu_id"
      
      if [ "$size" == "all" ]; then
        SUBSET_ARG=""
      else
        SUBSET_ARG="--train-subset $size"
      fi
      
      (time CUDA_VISIBLE_DEVICES=$gpu_id python3 train_svm_qkernel.py \
        --config configs/${ds}_${diff}.yaml \
        --gram-backend torch \
        $SUBSET_ARG \
        --pca-components 16 \
        --embed-mode ryrz \
        --kernel-centering \
        --normalize-kernel) 2>&1 | tee -a log_${ds}_${diff}_torch_${size}.txt &
      
      gpu_id=$(( (gpu_id + 1) % max_gpus ))
      
      if [ $gpu_id -eq 0 ]; then
        echo "All GPUs are busy. Waiting for the current wave to finish..."
        wait
      fi
      
    done
  done
done

wait
echo "All quantum jobs completed."
