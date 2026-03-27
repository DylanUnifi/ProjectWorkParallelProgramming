#!/bin/bash

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all") # Ajout du "all" !

cpu_jobs=0
max_cpu_jobs=6 # Garde une marge, car les "all" sur SVHN vont prendre beaucoup de RAM classique !

echo "🔥 Démarrage du benchmark Classique sur CPU..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      
      echo "➔ Lancement Classique $ds | $diff | $size"
      
      # Si la taille est "all", on n'envoie pas l'argument --train-subset
      if [ "$size" == "all" ]; then
        SUBSET_ARG=""
      else
        SUBSET_ARG="--train-subset $size"
      fi
      
      (time python3 train_svm_classical.py \
        --config configs/${ds}_${diff}.yaml \
        $SUBSET_ARG \
        --pca-components 16 \
        --kernel rbf) 2>&1 | tee -a log_${ds}_${diff}_classique_${size}.txt &
      
      cpu_jobs=$((cpu_jobs + 1))
      
      if [ $cpu_jobs -eq $max_cpu_jobs ]; then
        echo "⏳ Limite de concurrence CPU atteinte. Attente..."
        wait
        cpu_jobs=0
      fi
      
    done
  done
done

wait
echo "✅ TOUTES LES TÂCHES CLASSIQUES SONT TERMINÉES !"