#!/bin/bash
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -o extract_deep_features_%j.out

date
srun singularity exec -B /data-x/g15/ --nv /app/singularity/deepo/20190624.sif \
for payload in 0.4; do
srun python3 -u get_cnn_fea.py \
--model_type SRNet \
--cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
--stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_$payload\/ \
--adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/SRNet/UNIWARD/payload_$payload\/ \
--cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_$payload\/cover.mat \
--stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_$payload\/S-UNIWARD_$payload\.mat \
--adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_$payload\/ADV-EMB_SRNet_UNIWARD_payload_$payload\.mat \
--load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_$payload\/Model_670000.ckpt \
--batch_size 50 \
--num_workers 8
    date
done

get_cnn_fea.py \
--model_type SRNet \
--cover_dir /data-x/g15/qinchuan/Spatial/imresize-256/cover/ \
--stego_dir /data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_0.4/ \
--adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB/SRNet/UNIWARD/payload_0.4/ \
--cover_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.4/cover.mat \
--stego_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.4/S-UNIWARD_0.4.mat \
--adv_feature_path ~/data/feature/BOSS_BOWS2/spatial/256/SRNet/S-UNIWARD/payload_0.4/ADV-EMB_SRNet_UNIWARD_payload_0.4.mat \
--load_path /public/qinchuan/deep-learning/SRNet/UNIWARD/payload_0.4/Model_670000.ckpt \
--batch_size 50 \
--num_workers 8
