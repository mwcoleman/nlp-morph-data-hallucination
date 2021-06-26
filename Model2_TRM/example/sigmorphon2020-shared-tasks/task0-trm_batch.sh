#!/bin/bash
gpu=0
data_dir=task0-data/processed/$1
ckpt_dir=results/original

lang=$1
arch=transformer

lr=0.001
scheduler=warmupinvsqr
max_steps=$2 # CRH: 1400 steps for 100 epochs. 14steps/epoch.   SWA
warmup=4000
beta2=0.98
label_smooth=0.1
total_eval=30
bs=125

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=0.3     #${2:-0.3}

CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/$lang.trn \
    --dev $data_dir/$lang.dev \
    --test $data_dir/$lang.tst \
    --model $ckpt_dir/$lang \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
