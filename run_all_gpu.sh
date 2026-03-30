#!/bin/bash

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all")
backends=("torch" "cuda_states") # 👈 La magie opère ici !

gpu_id=0
max_gpus=8

echo "🔥 Démarrage du cluster Quantique sur $max_gpus GPUs (Torch & CUDA States)..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      for backend in "${backends[@]}"; do
        
        echo "➔ Lancement $ds | $diff | $size | Backend: $backend sur le GPU n°$gpu_id"
        
        # Gestion du paramètre "all"
        if [ "$size" == "all" ]; then
          SUBSET_ARG=""
        else
          SUBSET_ARG="--train-subset $size"
        fi
        
        (time CUDA_VISIBLE_DEVICES=$gpu_id python3 train_svm_qkernel.py \
          --config configs/${ds}_${diff}.yaml \
          --gram-backend $backend \
          $SUBSET_ARG \
          --pca-components 16 \
          --embed-mode ryrz \
          --kernel-centering \
          --normalize-kernel) 2>&1 | tee -a log_${ds}_${diff}_${backend}_${size}.txt &
        
        gpu_id=$(( (gpu_id + 1) % max_gpus ))
        
        # Pause tous les 8 lancements
        if [ $gpu_id -eq 0 ]; then
          echo "⏳ Les 8 GPUs sont occupés. Attente de la fin de la vague..."
          wait
        fi
        
      done
    done
  done
done

wait
echo "✅ TOUTES LES TÂCHES QUANTIQUES SONT TERMINÉES !"